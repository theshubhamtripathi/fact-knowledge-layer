"""Cross-document relationship detection: embedding similarity narrows the
O(n*m) fact-pair space down to plausible candidates, then the LLM classifies
candidate pairs in batches (several pairs per call) to stay within a tight
per-minute API quota. This avoids O(n^2) LLM calls across the whole fact store."""
import json

import numpy as np

from app.llm_client import generate_json

PAIRS_PER_CLASSIFY_CALL = 15

SIMILARITY_THRESHOLD = 0.72
TOP_K_PER_FACT = 3

RELATION_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "relation_type": {
            "type": "STRING",
            "enum": ["corroborates", "contradicts", "context_explained", "unrelated"],
        },
        "explanation": {"type": "STRING"},
        "confidence": {"type": "NUMBER"},
    },
    "required": ["relation_type", "explanation", "confidence"],
}

PAIR_BLOCK_TEMPLATE = """PAIR {idx}:
FACT A (from "{doc_a}", page {page_a}):
Statement: {text_a}
Subject: {subject_a} | Attribute: {attribute_a} | Value: {value_a} | Unit: {unit_a} | Time scope: {time_a}
Source quote: "{quote_a}"

FACT B (from "{doc_b}", page {page_b}):
Statement: {text_b}
Subject: {subject_b} | Attribute: {attribute_b} | Value: {value_b} | Unit: {unit_b} | Time scope: {time_b}
Source quote: "{quote_b}"
"""

CLASSIFY_BATCH_PROMPT = """You are comparing pairs of factual claims, each pair extracted from two \
DIFFERENT source documents, to determine how the two facts in each pair relate to each other.

For EACH pair below, classify the relationship as exactly one of:
- "corroborates": both facts assert the same underlying truth (even if phrased differently, in a table \
vs prose, etc.).
- "contradicts": the two facts cannot both be true as stated — same subject/attribute/time-scope/unit, \
but different values, with no obvious reconciling explanation.
- "context_explained": the facts LOOK contradictory (e.g. different numbers for what seems like the \
same metric) but are explained by a real difference in time period, scope (e.g. standalone vs \
consolidated), or unit (e.g. INR crore vs INR lakh, absolute vs percentage). Explain exactly what \
context resolves it.
- "unrelated": the facts are not meaningfully comparable (different subjects/attributes entirely), even \
though they were textually similar enough to be shortlisted.

Respond with a JSON array of exactly {n} objects, IN THE SAME ORDER as the pairs below. Each object has \
relation_type, a one-to-two sentence explanation citing the specific evidence from both facts, and a \
confidence score from 0.0 to 1.0.

{pairs_block}
"""


def cosine_similarity(a: list[float], b: list[float]) -> float:
    va, vb = np.array(a), np.array(b)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    if denom == 0:
        return 0.0
    return float(np.dot(va, vb) / denom)


def find_candidate_pairs(new_facts: list[dict], existing_facts: list[dict]) -> list[tuple[dict, dict, float]]:
    """For each new fact, shortlist the most similar existing facts from OTHER
    documents above a similarity threshold."""
    candidates = []
    existing_with_emb = [
        (f, json.loads(f["embedding"])) for f in existing_facts if f.get("embedding")
    ]
    for nf in new_facts:
        if not nf.get("embedding"):
            continue
        nf_emb = json.loads(nf["embedding"]) if isinstance(nf["embedding"], str) else nf["embedding"]
        scored = []
        for ef, ef_emb in existing_with_emb:
            if ef["document_id"] == nf["document_id"]:
                continue
            sim = cosine_similarity(nf_emb, ef_emb)
            if sim >= SIMILARITY_THRESHOLD:
                scored.append((ef, sim))
        scored.sort(key=lambda x: x[1], reverse=True)
        for ef, sim in scored[:TOP_K_PER_FACT]:
            candidates.append((nf, ef, sim))
    return candidates


BATCH_RESPONSE_SCHEMA = {
    "type": "ARRAY",
    "items": RELATION_SCHEMA,
}


def _pair_block(idx: int, fact_a: dict, doc_a_name: str, fact_b: dict, doc_b_name: str) -> str:
    return PAIR_BLOCK_TEMPLATE.format(
        idx=idx,
        doc_a=doc_a_name, page_a=fact_a["page_number"], text_a=fact_a["fact_text"],
        subject_a=fact_a.get("subject", ""), attribute_a=fact_a.get("attribute", ""),
        value_a=fact_a.get("value", ""), unit_a=fact_a.get("unit", ""), time_a=fact_a.get("time_scope", ""),
        quote_a=fact_a["verbatim_quote"],
        doc_b=doc_b_name, page_b=fact_b["page_number"], text_b=fact_b["fact_text"],
        subject_b=fact_b.get("subject", ""), attribute_b=fact_b.get("attribute", ""),
        value_b=fact_b.get("value", ""), unit_b=fact_b.get("unit", ""), time_b=fact_b.get("time_scope", ""),
        quote_b=fact_b["verbatim_quote"],
    )


def classify_pairs_batch(pairs: list[tuple[dict, str, dict, str]]) -> list[dict]:
    """Classify several (fact_a, doc_a_name, fact_b, doc_b_name) tuples in one
    LLM call. Falls back to 'unrelated' entries if the model returns a
    mismatched-length array, so callers can zip results 1:1 with input pairs."""
    if not pairs:
        return []

    blocks = [
        _pair_block(i + 1, fact_a, doc_a_name, fact_b, doc_b_name)
        for i, (fact_a, doc_a_name, fact_b, doc_b_name) in enumerate(pairs)
    ]
    prompt = CLASSIFY_BATCH_PROMPT.format(n=len(pairs), pairs_block="\n".join(blocks))
    result = generate_json(prompt, response_schema=BATCH_RESPONSE_SCHEMA)

    fallback = {"relation_type": "unrelated", "explanation": "Classifier returned unexpected shape.", "confidence": 0.0}
    if not isinstance(result, list) or len(result) != len(pairs):
        return [fallback] * len(pairs)
    return [r if isinstance(r, dict) else fallback for r in result]
