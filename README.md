# Fact Knowledge Layer

Ingests PDFs, extracts atomic factual claims grounded in a verbatim quote + page number from the
source document, and classifies relationships between facts across documents as **corroborating**,
**contradicting**, or **contradiction-explained-by-context** (different time period, scope, or units).

## Setup and Run Instructions

```bash
cd backend
python -m venv .venv && .venv\Scripts\activate   # or: source .venv/bin/activate on macOS/Linux
pip install -r requirements.txt
copy .env.example .env   # then fill in GEMINI_API_KEY (needs billing enabled — see Additional Notes)
python -m uvicorn app.main:app --reload --port 8000
```

Open http://127.0.0.1:8000 — upload a PDF, watch facts and cross-document relationships populate.

API surface:
- `POST /documents` — upload a PDF, run the pipeline, return a summary (facts extracted/verified,
  relationships created).
- `GET /documents` — list ingested documents.
- `GET /facts?document_id=` — list facts with evidence (quote, page, source doc).
- `GET /relationships?type=corroborates|contradicts|context_explained` — list relationships with
  both facts' evidence and the LLM's reasoning.

## Video Demo

[Not recorded in this session — see Additional Notes for what a walkthrough would show.]

## Approach

**Architecture** — `backend/app/`:
- `pdf_ingest.py` — per-page text extraction (pdfplumber), grouped into overlapping page windows so
  large PDFs are processed in bounded chunks rather than one giant LLM call.
- `llm_client.py` — thin REST wrapper around the Gemini API (no SDK dependency). Self-paces requests,
  retries network/5xx errors with backoff, and batches embeddings/classifications into as few calls as
  possible.
- `fact_extraction.py` — prompts the LLM for structured atomic facts per page window, then
  programmatically verifies each fact's `verbatim_quote` actually appears (exact or close fuzzy match)
  on the cited page. Facts that fail this check are stored with `grounding_status='failed'` and
  excluded from relationship comparison rather than silently dropped, so they stay inspectable.
- `relationships.py` — embeds each fact (subject+attribute+value+text), shortlists candidate pairs
  across *different* documents via cosine similarity (top-3 per fact, similarity ≥ 0.72), then asks the
  LLM to classify each candidate batch (15 pairs/call) as corroborating / contradicting /
  context-explained / unrelated with a one-line reasoning.
- `pipeline.py` — orchestrates ingest → extract → ground → embed → incrementally compare only the new
  document's facts against the existing store (not a full pairwise recompute).
- `store.py` — SQLite. `facts.category`/`attribute` are free text chosen by the LLM per fact, not a
  fixed enum, so new kinds of facts don't require a schema change.
- `main.py` + `static/index.html` — FastAPI routes and a minimal UI (upload, facts table, relationships
  table filterable by type, side-by-side evidence). Ingestion runs in a thread pool so the server stays
  responsive to GET requests during a long upload.

**Key decisions / trade-offs**:
- **Page windowing**: 25 pages per LLM call, 1-page overlap (so a fact split across a page boundary
  still lands whole in one window). This was tuned up from an initial 3-page window specifically to cut
  the number of API calls against a tight free-tier quota (see Limitations) — a paid/higher-quota key
  could safely use smaller windows for finer-grained, more accurate extraction on very dense pages.
- **Grounding threshold**: exact substring match first, fuzzy fallback (`rapidfuzz.partial_ratio` ≥ 85)
  to tolerate whitespace/line-break differences introduced by PDF text extraction, without accepting
  loosely-paraphrased "verbatim" quotes.
- **Relationship blocking**: cosine similarity on Gemini embeddings (768-dim) narrows candidates before
  any classification LLM call — avoids O(n²) comparisons across the whole fact store, and only the new
  document's facts are compared against the existing store (incremental, not a full recompute).
- **LLM**: Gemini, called directly over its REST API (`gemini-flash-lite-latest` for extraction/
  classification, `gemini-embedding-001` for embeddings) rather than via the `google-genai` SDK, so the
  project isn't blocked on SDK installability.
- **AI tools used**: Claude Code (Anthropic's coding agent) was used to build this project end-to-end —
  architecture, all backend code, prompts, and this README. Gemini (`gemini-flash-lite-latest` /
  `gemini-embedding-001`) is the LLM doing extraction, embedding, and classification at runtime.

## The four required cases

Pulled directly from a real run over the three Delhivery documents (prospectus 2022, annual report
FY24, Q4 FY24 earnings deck) — not hand-picked afterward. Full detail (all facts, quotes, pages) is in
the SQLite store; summarized here.

**1. Corroborated fact** (confidence 1.00)
- Annual Report FY24, p.37 (prose): *"Our debt-equity ratio was 0.01."*
- Q4 FY24 earnings deck, p.20 (table row): *"Debt/Equity (A/C) 0.02x 0.01x"*
- System's reasoning: both report the same 0.01 debt-equity ratio as of March 31, 2024 — same fact,
  once stated in prose and once buried in a table trend row.

**2. Genuine/likely contradiction** (confidence 0.95)
- Prospectus 2022, p.31: *"Sunil Kumar Bansal is the Company Secretary and Compliance Officer of our
  Company."*
- Q4 FY24 earnings deck, p.1: *"Madhulika Rawat / Company Secretary & Compliance Officer / Membership
  No: F8765"*
- Two different named individuals hold the identical specific title for the same company. The system
  found no explicit resignation/appointment date bridging the two documents to reconcile it, so it
  stands as a likely contradiction — though the most plausible real-world explanation is ordinary
  personnel turnover between 2022 and 2024, which the current pipeline can't confirm without an
  explicit transition record in the source text.

**3. Context-explained apparent contradiction** (confidence 0.95)
- Prospectus 2022, p.51: *"On December 31, 2021, we acquired 34.55% of the share capital (on a
  fully-diluted basis) of Falcon"*
- Annual Report FY24, p.99: *"...taking the total stake to 40.98% (non-diluted basis)."*
- System's reasoning: the two percentages look contradictory at a glance, but they're reconciled by two
  independent factors — roughly 2.5 years of further investment between the two dates, *and* a
  different denominator basis (fully-diluted vs. non-diluted share count).

**4. A real extraction failure — and what it caused**
- Annual Report FY24, p.2 is an infographic: one row of numbers (`>2.8  >4.8  15,065  753  98,135`)
  stacked above a row of labels (`Express parcel shipments | ... | Daily average fleet size | Count of
  46-ft tractors | Workforce strength`) — five numbers, five labels, printed as two disconnected lines
  once pdfplumber linearizes the page.
- The LLM paired `753` with the label *"Daily average fleet size"* — actually the 4th number/label pair
  (`753` = *count of 46-ft tractors*; `15,065` two columns over is the real fleet-size figure). The
  quote `"753 Daily average fleet size"` does appear verbatim-adjacent on the page, so it **passed
  grounding** — grounding only verifies a quote exists on the cited page, not that the LLM paired the
  right number with the right label.
- Downstream effect: this misattributed fact got compared against the earnings deck's correct
  `15,065` daily-average-fleet-size figure and classified as a **false-positive "contradicts"** (both
  claim "daily average fleet size" for the same quarter, off by 20×) — with no reconciling context,
  because there isn't one; the fact itself is wrong.
- How I'd fix it: this is a systematic risk for any infographic/multi-column layout where pdfplumber
  linearizes spatially-separated content into adjacent text lines. The real fix is layout-aware
  extraction (e.g. pdfplumber's `.extract_tables()` / word bounding boxes to preserve column
  alignment) fed to the LLM as structured rows instead of raw linear text, so numbers and labels
  can't be silently miscolumned. Short of that, a cheap mitigation already partially in place: the
  grounding check caught 181 of 525 facts (34%) across all five documents as failing verbatim-quote
  matching — those are excluded from relationship comparison, which is why this particular bad fact
  is the exception that slipped through (its quote happened to be locally accurate) rather than the
  rule.

## Limitations and Next Steps

- **Free-tier LLM quota drove architecture decisions under time pressure.** The Gemini key used here
  is on the free tier, which caps `generateContent` at a small number of requests **per day, per
  model** (not per-minute, despite the API's misleading "retry in Ns" hint on 429s). This forced
  larger page windows and larger classification batches than would be ideal for extraction accuracy on
  very dense pages, and made it impractical to fully process the india-macroeconomy dataset in the
  time available (RBI Annual Report and part of the IMF Article IV excerpt were ingested — see
  `facts.db` — but not all three). With a billed key, smaller windows (3–5 pages) would very likely
  raise extraction completeness and reduce the false-negative rate on dense financial tables.
- **Grounding-failure rate varies a lot by document layout** (3.6% on the text-heavy prospectus vs.
  57.5% on the infographic/table-heavy annual report) — a sign that table and infographic pages are
  where this pipeline is weakest, consistent with the case-4 finding above.
- **Relationship blocking (top-3 by cosine similarity, threshold 0.72) can miss genuine contradictions**
  phrased very differently in each source — it only catches what's semantically close enough to be
  embedded near its counterpart.
- **Next steps**: layout-aware PDF extraction for tables/infographics; a second LLM pass that
  cross-checks a classified "contradicts" pair against surrounding page context before finalizing;
  smaller windows once quota allows; a proper video walkthrough.

## Additional Notes

- This run used `gemini-flash-lite-latest` for generation after `gemini-flash-latest` hit its free-tier
  daily cap partway through testing — both are configurable via `GEMINI_GENERATION_MODEL` in `.env`.
- Generalization check: the pipeline was pointed at the india-macroeconomy dataset (RBI Annual Report
  2024-25, IMF Article IV excerpt) with zero code changes and successfully extracted grounded facts
  from both — confirming nothing in the extraction/relationship logic is Delhivery-specific.
