"""SQLite persistence. Schema intentionally keeps `category`/`attribute` as
free text (LLM-chosen per fact) rather than a fixed enum, so new kinds of
facts don't require a schema migration."""
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "facts.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT NOT NULL,
    uploaded_at TEXT NOT NULL,
    page_count INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS facts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id INTEGER NOT NULL REFERENCES documents(id),
    fact_text TEXT NOT NULL,
    subject TEXT,
    attribute TEXT,
    value TEXT,
    unit TEXT,
    time_scope TEXT,
    category TEXT,
    page_number INTEGER NOT NULL,
    verbatim_quote TEXT NOT NULL,
    grounding_status TEXT NOT NULL,
    embedding TEXT,
    extracted_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS relationships (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    fact_id_a INTEGER NOT NULL REFERENCES facts(id),
    fact_id_b INTEGER NOT NULL REFERENCES facts(id),
    relation_type TEXT NOT NULL,
    explanation TEXT NOT NULL,
    confidence REAL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_facts_document ON facts(document_id);
CREATE INDEX IF NOT EXISTS idx_relationships_type ON relationships(relation_type);
"""


@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with get_conn() as conn:
        conn.executescript(SCHEMA)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def insert_document(filename: str, page_count: int) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO documents (filename, uploaded_at, page_count) VALUES (?, ?, ?)",
            (filename, now(), page_count),
        )
        return cur.lastrowid


def list_documents() -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute("SELECT * FROM documents ORDER BY id").fetchall()
        return [dict(r) for r in rows]


def get_document(document_id: int) -> dict | None:
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM documents WHERE id = ?", (document_id,)).fetchone()
        return dict(row) if row else None


def insert_fact(
    document_id: int,
    fact_text: str,
    subject: str,
    attribute: str,
    value: str,
    unit: str,
    time_scope: str,
    category: str,
    page_number: int,
    verbatim_quote: str,
    grounding_status: str,
    embedding: list[float] | None,
) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            """INSERT INTO facts
            (document_id, fact_text, subject, attribute, value, unit, time_scope,
             category, page_number, verbatim_quote, grounding_status, embedding, extracted_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                document_id, fact_text, subject, attribute, value, unit, time_scope,
                category, page_number, verbatim_quote, grounding_status,
                json.dumps(embedding) if embedding is not None else None,
                now(),
            ),
        )
        return cur.lastrowid


def list_facts(document_id: int | None = None, grounding_status: str | None = None) -> list[dict]:
    query = "SELECT * FROM facts"
    clauses, params = [], []
    if document_id is not None:
        clauses.append("document_id = ?")
        params.append(document_id)
    if grounding_status is not None:
        clauses.append("grounding_status = ?")
        params.append(grounding_status)
    if clauses:
        query += " WHERE " + " AND ".join(clauses)
    query += " ORDER BY id"
    with get_conn() as conn:
        rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]


def get_fact(fact_id: int) -> dict | None:
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM facts WHERE id = ?", (fact_id,)).fetchone()
        return dict(row) if row else None


def facts_excluding_document(document_id: int) -> list[dict]:
    """Existing grounded facts from every OTHER document — used for incremental
    ingestion so a new upload is only compared against what's already there."""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM facts WHERE document_id != ? AND grounding_status = 'verified' AND embedding IS NOT NULL",
            (document_id,),
        ).fetchall()
        return [dict(r) for r in rows]


def insert_relationship(fact_id_a: int, fact_id_b: int, relation_type: str, explanation: str, confidence: float) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            """INSERT INTO relationships (fact_id_a, fact_id_b, relation_type, explanation, confidence, created_at)
            VALUES (?, ?, ?, ?, ?, ?)""",
            (fact_id_a, fact_id_b, relation_type, explanation, confidence, now()),
        )
        return cur.lastrowid


def list_relationships(relation_type: str | None = None) -> list[dict]:
    query = """
    SELECT r.*,
           fa.fact_text AS fact_a_text, fa.verbatim_quote AS fact_a_quote, fa.page_number AS fact_a_page,
           fa.document_id AS fact_a_document_id, da.filename AS fact_a_filename,
           fb.fact_text AS fact_b_text, fb.verbatim_quote AS fact_b_quote, fb.page_number AS fact_b_page,
           fb.document_id AS fact_b_document_id, db.filename AS fact_b_filename
    FROM relationships r
    JOIN facts fa ON fa.id = r.fact_id_a
    JOIN facts fb ON fb.id = r.fact_id_b
    JOIN documents da ON da.id = fa.document_id
    JOIN documents db ON db.id = fb.document_id
    """
    params = []
    if relation_type is not None:
        query += " WHERE r.relation_type = ?"
        params.append(relation_type)
    query += " ORDER BY r.id DESC"
    with get_conn() as conn:
        rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]
