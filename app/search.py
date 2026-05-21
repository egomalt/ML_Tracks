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

        all_views = [s.get("views", 0) or 0 for s in self._songs.values()]
        self._max_views: int = max(all_views) if all_views else 0

        print(f"[searcher] ready — {self._index.ntotal} songs in index")

    def search(
        self,
        query: str,
        top_k: int = 3,
        genre: str | None = None,
    ) -> list[dict[str, Any]]:
        vec = self._model.encode(
            [query], normalize_embeddings=True
        ).astype("float32")

        # При фильтре по жанру берём больше кандидатов, чтобы было из чего выбрать
        fetch_k = min(top_k * 8 if genre else top_k, self._index.ntotal)
        distances, indices = self._index.search(vec, fetch_k)

        # Находим максимальный views для нормализации популярности
        max_views = self._max_views

        results: list[dict] = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx == -1:
                continue
            entry = self._song_map[int(idx)]
            song = self._songs.get(entry["song_id"])
            if not song:
                continue

            song_genre = song.get("genre", "Unknown")
            if genre and song_genre.lower() != genre.lower():
                continue

            # Взвешивание: cosine similarity + нормализованная популярность
            views = song.get("views", 0) or 0
            popularity = (np.log1p(views) / np.log1p(max_views)) if max_views > 0 else 0.0
            final_score = 0.85 * float(dist) + 0.15 * popularity

            results.append({
                "title": song["title"],
                "artist": song["artist"],
                "genre": song_genre,
                "url": song.get("url", ""),
                "matched_text": entry["preview_text"],
                "score": round(final_score, 4),
            })

            if len(results) >= top_k:
                break

        return results
