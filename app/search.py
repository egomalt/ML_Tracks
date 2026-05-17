"""
Ядро поиска — используется и FastAPI-сервером, и Cog-предиктором.

Поиск идёт по вектору всей песни (среднее всех чанков), а не по отдельному куплету.
Это даёт соответствие по настроению трека целиком.
"""
import json
import pickle
import pathlib
from typing import Any

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

INDEX_DIR = pathlib.Path("index")
INDEX_PATH = INDEX_DIR / "song_index.faiss"
SONG_MAP_PATH = INDEX_DIR / "song_map.pkl"
SONGS_PATH = pathlib.Path("data/processed/songs.json")

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"


class SongSearcher:
    def __init__(self, model_name: str = MODEL_NAME) -> None:
        print(f"[searcher] loading model {model_name} ...")
        self._model = SentenceTransformer(model_name)

        print(f"[searcher] loading FAISS index ...")
        self._index = faiss.read_index(str(INDEX_PATH))
        self._index.hnsw.efSearch = 100  # могло быть сохранено в индексе — перезаписываем явно

        with open(SONG_MAP_PATH, "rb") as f:
            # список [{song_id, preview_text}] — позиция = строка в индексе
            self._song_map: list[dict] = pickle.load(f)

        with open(SONGS_PATH, encoding="utf-8") as f:
            self._songs: dict[str, dict] = json.load(f)

        print(f"[searcher] ready — {self._index.ntotal} songs in index")

    def search(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        vec = self._model.encode(
            [query], normalize_embeddings=True
        ).astype("float32")

        distances, indices = self._index.search(vec, top_k)

        results: list[dict] = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx == -1:
                continue
            entry = self._song_map[int(idx)]
            song = self._songs.get(entry["song_id"])
            if not song:
                continue
            results.append({
                "title": song["title"],
                "artist": song["artist"],
                "genre": song.get("genre", "Unknown"),
                "url": song.get("url", ""),
                "matched_text": entry["preview_text"],
                "score": round(float(dist), 4),
            })

        return results
