"""End-to-end orchestration: ingest a PDF, extract+ground facts, embed them,
and incrementally compare only the new facts against everything already in
the store (not a full pairwise recompute).

Embeddings and relationship classifications are batched into as few API
calls as possible — the Gemini key this project runs on has a tight
per-minute free-tier request quota, so per-fact/per-pair calls would make a
100-page document take far too long."""
import logging

from app import store
from app.fact_extraction import extract_facts_from_window, verify_grounding
from app.llm_client import LLMError, embed_texts_batch
from app.pdf_ingest import extract_pages, window_pages
from app.relationships import PAIRS_PER_CLASSIFY_CALL, classify_pairs_batch, find_candidate_pairs

logger = logging.getLogger(__name__)

WINDOW_SIZE = 25
WINDOW_OVERLAP = 1


def _embedding_text(fact: dict) -> str:
    parts = [fact.get("subject", ""), fact.get("attribute", ""), fact.get("value", ""), fact.get("fact_text", "")]
    return " | ".join(p for p in parts if p)


def ingest_document(pdf_path: str, filename: str) -> dict:
    pages = extract_pages(pdf_path)
    page_text_by_number = {p.page_number: p.text for p in pages}
    document_id = store.insert_document(filename, len(pages))

    windows = window_pages(pages, window_size=WINDOW_SIZE, overlap=WINDOW_OVERLAP)

    seen_dedup_keys: set[tuple[int, str]] = set()
    inserted_facts: list[dict] = []
    verified_count = 0
    failed_count = 0
    extraction_errors: list[str] = []

    for window in windows:
        page_range = f"{window[0].page_number}-{window[-1].page_number}"
        try:
            raw_facts = extract_facts_from_window(window)
            logger.info("Window pages %s: extracted %d raw facts", page_range, len(raw_facts))
        except LLMError as e:
            extraction_errors.append(str(e))
            logger.warning("Extraction failed for page window %s: %s", page_range, e)
            continue

        window_facts: list[dict] = []
        for raw in raw_facts:
            dedup_key = (raw.get("page_number"), (raw.get("verbatim_quote") or "").strip())
            if dedup_key in seen_dedup_keys:
                continue
            seen_dedup_keys.add(dedup_key)

            grounding_status = verify_grounding(raw, page_text_by_number)
            if grounding_status == "verified":
                verified_count += 1
            else:
                failed_count += 1
            raw["_grounding_status"] = grounding_status
            window_facts.append(raw)

        # Batch-embed every verified fact from this window in one API call.
        verified_in_window = [f for f in window_facts if f["_grounding_status"] == "verified"]
        embeddings_by_idx: dict[int, list[float]] = {}
        if verified_in_window:
            try:
                embs = embed_texts_batch([_embedding_text(f) for f in verified_in_window])
                embeddings_by_idx = {id(f): e for f, e in zip(verified_in_window, embs)}
            except LLMError as e:
                logger.warning("Batch embedding failed for a window, storing facts without embeddings: %s", e)

        for raw in window_facts:
            fact_id = store.insert_fact(
                document_id=document_id,
                fact_text=raw.get("fact_text", ""),
                subject=raw.get("subject", ""),
                attribute=raw.get("attribute", ""),
                value=raw.get("value", ""),
                unit=raw.get("unit", ""),
                time_scope=raw.get("time_scope", ""),
                category=raw.get("category", ""),
                page_number=raw.get("page_number"),
                verbatim_quote=raw.get("verbatim_quote", ""),
                grounding_status=raw["_grounding_status"],
                embedding=embeddings_by_idx.get(id(raw)),
            )
            saved = store.get_fact(fact_id)
            if saved:
                inserted_facts.append(saved)

    new_verified_facts = [f for f in inserted_facts if f["grounding_status"] == "verified" and f.get("embedding")]
    existing_facts = store.facts_excluding_document(document_id)

    relationships_created = 0
    if new_verified_facts and existing_facts:
        doc_name_cache: dict[int, str] = {document_id: filename}
        candidates = find_candidate_pairs(new_verified_facts, existing_facts)

        deduped: list[tuple[dict, dict, float]] = []
        seen_pairs: set[tuple[int, int]] = set()
        for new_fact, existing_fact, sim in candidates:
            pair_key = tuple(sorted((new_fact["id"], existing_fact["id"])))
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)
            deduped.append((new_fact, existing_fact, sim))

        for i in range(0, len(deduped), PAIRS_PER_CLASSIFY_CALL):
            chunk = deduped[i : i + PAIRS_PER_CLASSIFY_CALL]
            batch_input = []
            for new_fact, existing_fact, _sim in chunk:
                other_doc_id = existing_fact["document_id"]
                if other_doc_id not in doc_name_cache:
                    doc = store.get_document(other_doc_id)
                    doc_name_cache[other_doc_id] = doc["filename"] if doc else "unknown"
                batch_input.append((new_fact, filename, existing_fact, doc_name_cache[other_doc_id]))

            try:
                results = classify_pairs_batch(batch_input)
            except LLMError as e:
                logger.warning("Relationship classification failed for a batch: %s", e)
                continue

            for (new_fact, existing_fact, _sim), result in zip(chunk, results):
                relation_type = result.get("relation_type", "unrelated")
                if relation_type == "unrelated":
                    continue
                store.insert_relationship(
                    fact_id_a=new_fact["id"],
                    fact_id_b=existing_fact["id"],
                    relation_type=relation_type,
                    explanation=result.get("explanation", ""),
                    confidence=float(result.get("confidence", 0.0)),
                )
                relationships_created += 1

    return {
        "document_id": document_id,
        "filename": filename,
        "page_count": len(pages),
        "facts_extracted": len(inserted_facts),
        "facts_verified": verified_count,
        "facts_grounding_failed": failed_count,
        "relationships_created": relationships_created,
        "extraction_errors": extraction_errors,
    }
