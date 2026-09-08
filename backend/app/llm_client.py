"""Thin wrapper around the Gemini REST API — no SDK dependency, so it keeps
working even in environments where installing google-genai isn't possible.

The free-tier key this project runs on has a tight per-minute request quota,
so this client self-paces calls and batches embeddings, rather than issuing
one HTTP request per fact."""
import json
import os
import re
import threading
import time

import requests

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
GENERATION_MODEL = os.environ.get("GEMINI_GENERATION_MODEL", "models/gemini-flash-latest")
EMBEDDING_MODEL = os.environ.get("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001")

_MAX_RETRIES = 10
_MIN_CALL_INTERVAL_SECONDS = float(os.environ.get("GEMINI_MIN_CALL_INTERVAL", "4.5"))

_rate_lock = threading.Lock()
_last_call_at = 0.0


class LLMError(RuntimeError):
    pass


def _throttle():
    global _last_call_at
    with _rate_lock:
        wait = _MIN_CALL_INTERVAL_SECONDS - (time.time() - _last_call_at)
        if wait > 0:
            time.sleep(wait)
        _last_call_at = time.time()


def _parse_retry_delay_seconds(resp_text: str, default: float) -> float:
    match = re.search(r'"retryDelay":\s*"(\d+(?:\.\d+)?)s"', resp_text)
    if match:
        return float(match.group(1)) + 1.0
    match = re.search(r"retry in (\d+(?:\.\d+)?)s", resp_text)
    if match:
        return float(match.group(1)) + 1.0
    return default


def _post_with_retries(url: str, payload: dict) -> dict:
    last_err = None
    for attempt in range(_MAX_RETRIES):
        _throttle()
        resp = requests.post(url, json=payload, timeout=180)
        if resp.status_code == 200:
            return resp.json()
        if resp.status_code in (429, 500, 503):
            last_err = f"{resp.status_code}: {resp.text[:500]}"
            delay = _parse_retry_delay_seconds(resp.text, default=2 ** attempt)
            time.sleep(delay)
            continue
        raise LLMError(f"Gemini API error {resp.status_code}: {resp.text[:1000]}")
    raise LLMError(f"Gemini API failed after {_MAX_RETRIES} retries: {last_err}")


def generate_json(prompt: str, response_schema: dict | None = None, temperature: float = 0.1) -> dict | list:
    """Call generateContent asking for structured JSON output, and parse it."""
    if not GEMINI_API_KEY:
        raise LLMError("GEMINI_API_KEY is not set")

    generation_config: dict = {
        "temperature": temperature,
        "responseMimeType": "application/json",
    }
    if response_schema is not None:
        generation_config["responseSchema"] = response_schema

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": generation_config,
    }
    url = f"{BASE_URL}/{GENERATION_MODEL}:generateContent?key={GEMINI_API_KEY}"
    data = _post_with_retries(url, payload)

    try:
        candidates = data["candidates"]
        parts = candidates[0]["content"]["parts"]
        text = "".join(p.get("text", "") for p in parts)
    except (KeyError, IndexError) as e:
        raise LLMError(f"Unexpected Gemini response shape: {data}") from e

    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise LLMError(f"Gemini did not return valid JSON: {text[:1000]}") from e


def embed_texts_batch(texts: list[str], task_type: str = "SEMANTIC_SIMILARITY") -> list[list[float]]:
    """Embed many texts in a single API call via batchEmbedContents."""
    if not GEMINI_API_KEY:
        raise LLMError("GEMINI_API_KEY is not set")
    texts = [t[:8000] if t.strip() else " " for t in texts]
    if not texts:
        return []

    requests_payload = [
        {
            "model": EMBEDDING_MODEL,
            "content": {"parts": [{"text": t}]},
            "taskType": task_type,
            "outputDimensionality": 768,
        }
        for t in texts
    ]
    url = f"{BASE_URL}/{EMBEDDING_MODEL}:batchEmbedContents?key={GEMINI_API_KEY}"
    data = _post_with_retries(url, {"requests": requests_payload})
    try:
        return [e["values"] for e in data["embeddings"]]
    except KeyError as e:
        raise LLMError(f"Unexpected batch embedding response shape: {data}") from e


def embed_text(text: str, task_type: str = "SEMANTIC_SIMILARITY") -> list[float]:
    return embed_texts_batch([text], task_type=task_type)[0]
