import logging
import shutil
import tempfile
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app import store
from app.pipeline import ingest_document

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

app = FastAPI(title="Fact Knowledge Layer")

store.init_db()


@app.post("/documents")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        summary = await run_in_threadpool(ingest_document, tmp_path, file.filename)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}") from e
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return summary


@app.get("/documents")
async def get_documents():
    return store.list_documents()


@app.get("/facts")
async def get_facts(document_id: int | None = None):
    return store.list_facts(document_id=document_id)


@app.get("/relationships")
async def get_relationships(type: str | None = None):
    valid_types = {"corroborates", "contradicts", "context_explained"}
    if type is not None and type not in valid_types:
        raise HTTPException(status_code=400, detail=f"type must be one of {valid_types}")
    return store.list_relationships(relation_type=type)


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def root():
    return FileResponse(STATIC_DIR / "index.html")
