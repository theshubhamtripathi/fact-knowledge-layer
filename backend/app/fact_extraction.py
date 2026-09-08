"""Extract atomic factual claims from page windows, then verify each claim's
verbatim quote actually appears on the page it's attributed to."""
from rapidfuzz import fuzz

from app.llm_client import generate_json
from app.pdf_ingest import Page

FACT_SCHEMA = {
    "type": "ARRAY",
    "items": {
        "type": "OBJECT",
        "properties": {
            "fact_text": {"type": "STRING"},
            "subject": {"type": "STRING"},
            "attribute": {"type": "STRING"},
            "value": {"type": "STRING"},
            "unit": {"type": "STRING"},
            "time_scope": {"type": "STRING"},
            "category": {"type": "STRING"},
            "page_number": {"type": "INTEGER"},
            "verbatim_quote": {"type": "STRING"},
        },
        "required": ["fact_text", "subject", "attribute", "value", "page_number", "verbatim_quote"],
    },
}

PROMPT_TEMPLATE = """You are extracting atomic factual claims from pages of a real-world document \
(could be a financial filing, annual report, earnings deck, government report, or similar).

For EACH distinct factual claim (numeric or semantic) stated on these pages, output one object with:
- fact_text: a clear, self-contained one-sentence statement of the fact (should make sense without \
seeing the source).
- subject: what/who the fact is about (a company, a person, a metric's owner, etc).
- attribute: the specific property or metric being stated (e.g. "revenue", "employee count", \
"designation", "warehouse count").
- value: the actual value or assertion, as text (e.g. "7225.72", "Managing Director", "resigned").
- unit: unit of measurement if numeric (e.g. "INR crore", "%", "count"), else empty string.
- time_scope: the time period this fact applies to (e.g. "FY2024", "Q4 FY24", "as of March 31, 2022"), \
inferred from surrounding context if not in the exact sentence. Empty string if genuinely unknown.
- category: a short free-text category you choose for this fact (e.g. "financial", "governance", \
"operational", "biographical", "risk"). Do not restrict yourself to a fixed list — pick whatever fits.
- page_number: the exact page number (from the page markers below) this fact appears on.
- verbatim_quote: an EXACT, character-for-character substring COPIED from that page's text below that \
directly supports this fact. Do not paraphrase or summarize this field — it must be findable verbatim \
in the source text.

Rules:
- Only extract meaningful atomic claims a reader would cite as a "fact" — skip page headers, footers, \
page numbers, table-of-contents entries, and pure boilerplate/legal disclaimers.
- Prefer specific, checkable claims (numbers, named people/roles, dates, named entities) over vague \
statements.
- If a table row states a fact, still produce a verbatim_quote copied from the extracted text of that \
row/cell as it appears below (extracted table text may have unusual spacing — copy it as-is).
- It is fine to extract zero facts from a page that is purely a cover/section-divider page.
- Return ONLY the JSON array, no other commentary.

PAGES:
{pages_block}
"""


def _build_pages_block(pages: list[Page]) -> str:
    parts = []
    for p in pages:
        parts.append(f"--- PAGE {p.page_number} ---\n{p.text}")
    return "\n\n".join(parts)


def extract_facts_from_window(pages: list[Page]) -> list[dict]:
    if not any(p.text.strip() for p in pages):
        return []
    prompt = PROMPT_TEMPLATE.format(pages_block=_build_pages_block(pages))
    result = generate_json(prompt, response_schema=FACT_SCHEMA)
    if not isinstance(result, list):
        return []
    valid_pages = {p.page_number for p in pages}
    return [f for f in result if isinstance(f, dict) and f.get("page_number") in valid_pages]


GROUNDING_FUZZY_THRESHOLD = 85


def verify_grounding(fact: dict, page_text_by_number: dict[int, str]) -> str:
    """Return 'verified' if the verbatim_quote is found (exactly or via close
    fuzzy match) on the cited page, else 'failed'."""
    quote = (fact.get("verbatim_quote") or "").strip()
    page_number = fact.get("page_number")
    page_text = page_text_by_number.get(page_number, "")

    if not quote or not page_text:
        return "failed"

    if quote in page_text:
        return "verified"

    # Fuzzy fallback: pdf text extraction can introduce whitespace/line-break
    # differences even when the LLM copied the quote correctly.
    score = fuzz.partial_ratio(quote, page_text)
    return "verified" if score >= GROUNDING_FUZZY_THRESHOLD else "failed"
