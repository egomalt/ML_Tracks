"""
FastAPI-сервер — отдаёт поисковый API и статичный фронтенд.

Запуск:
    uvicorn app.main:app --reload --port 8000
"""
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from deep_translator import GoogleTranslator

from app.search import SongSearcher

_translator = GoogleTranslator(source="auto", target="ru")

STATIC_DIR = Path(__file__).parent / "static"
searcher: SongSearcher | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global searcher
    searcher = SongSearcher()
    yield


app = FastAPI(title="SongFinder", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class SearchRequest(BaseModel):
    query: str
    top_k: int = 3
    translate: bool = False


@app.get("/", include_in_schema=False)
async def root():
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/search")
async def search(req: SearchRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="query is empty")
    top_k = max(1, min(req.top_k, 10))
    results = searcher.search(req.query.strip(), top_k=top_k)
    if req.translate:
        for r in results:
            try:
                r["matched_text"] = _translator.translate(r["matched_text"])
            except Exception:
                pass  # если перевод упал — оставляем оригинал
    return {"results": results, "query": req.query}


@app.get("/health")
async def health():
    return {"status": "ok", "index_size": searcher._index.ntotal if searcher else 0}
