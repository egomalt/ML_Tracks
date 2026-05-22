#!/usr/bin/env python3
"""
Векторизация чанков через sentence-transformers и построение FAISS HNSW индекса.

Вектор каждой песни = среднее всех векторов её чанков (нормализованное).

Вход:  data/processed/chunks.jsonl  +  data/processed/songs.json
Выход: index/song_index.faiss   — один вектор на песню
       index/song_map.pkl       — список {song_id, best_chunk_text} по позиции в индексе

Память: чанки читаются потоково батчами; в RAM одновременно хранятся только
        текущий батч + аккумуляторы (sum-вектор на песню). Никаких 10M векторов.
"""
import json
import pathlib
import pickle
import argparse
import time
from collections import defaultdict

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

PROCESSED_DIR = pathlib.Path("data/processed")
INDEX_DIR = pathlib.Path("index")
CHUNKS_PATH = PROCESSED_DIR / "chunks.jsonl"
SONGS_PATH = PROCESSED_DIR / "songs.json"

HNSW_M = 32
HNSW_EF_CONSTRUCTION = 200
HNSW_EF_SEARCH = 100
BATCH_SIZE = 256

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"


# ---------------------------------------------------------------------------
# Потоковое чтение чанков батчами
# ---------------------------------------------------------------------------

def iter_batches(path: pathlib.Path, batch_size: int):
    """Читает chunks.jsonl и отдаёт батчи (chunk_dicts, vectorise_texts)."""
    batch_chunks, batch_texts = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            batch_chunks.append({
                "song_id":     rec["song_id"],
                "chunk_index": rec["chunk_index"],
                "text":        rec["text"],
            })
            batch_texts.append(rec["vectorise_text"])
            if len(batch_chunks) >= batch_size:
                yield batch_chunks, batch_texts
                batch_chunks, batch_texts = [], []
    if batch_chunks:
        yield batch_chunks, batch_texts


def count_lines(path: pathlib.Path) -> int:
    with open(path, "rb") as f:
        return sum(1 for _ in f)


# ---------------------------------------------------------------------------
# Кодирование + агрегация за один проход
# ---------------------------------------------------------------------------

def encode_and_aggregate(
    model: SentenceTransformer,
    chunks_path: pathlib.Path,
    batch_size: int,
) -> dict[str, dict]:
    """
    Читает чанки батчами, кодирует, сразу суммирует векторы по песне.
    Возвращает accum: {song_id -> {"sum": ndarray, "count": int, "text": str}}
    В RAM одновременно: один батч векторов + аккумуляторы.
    """
    print("Считаем чанки …")
    total = count_lines(chunks_path)
    print(f"  {total:,} чанков")

    accum: dict[str, dict] = {}

    with tqdm(total=total, desc="Encoding") as pbar:
        for batch_chunks, batch_texts in iter_batches(chunks_path, batch_size):
            vecs = model.encode(
                batch_texts,
                normalize_embeddings=True,
                show_progress_bar=False,
            ).astype("float32")

            for chunk, vec in zip(batch_chunks, vecs):
                sid = chunk["song_id"]
                if sid not in accum:
                    accum[sid] = {"sum": vec.copy(), "count": 1, "text": chunk["text"]}
                else:
                    accum[sid]["sum"] += vec
                    accum[sid]["count"] += 1

            pbar.update(len(batch_chunks))

    return accum


def build_song_vectors(
    accum: dict[str, dict],
) -> tuple[list[str], np.ndarray, dict[str, str]]:
    """Нормализует средние векторы и собирает матрицу для FAISS."""
    song_ids: list[str] = []
    averaged: list[np.ndarray] = []
    best_chunk: dict[str, str] = {}

    for sid, data in tqdm(accum.items(), desc="Normalizing"):
        mean_vec = data["sum"] / data["count"]
        norm = np.linalg.norm(mean_vec)
        if norm > 0:
            mean_vec /= norm
        song_ids.append(sid)
        averaged.append(mean_vec)
        best_chunk[sid] = data["text"]

    return song_ids, np.vstack(averaged).astype("float32"), best_chunk


def build_hnsw_index(vecs: np.ndarray) -> faiss.IndexHNSWFlat:
    dim = vecs.shape[1]
    index = faiss.IndexHNSWFlat(dim, HNSW_M, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = HNSW_EF_CONSTRUCTION
    index.hnsw.efSearch = HNSW_EF_SEARCH
    print(f"Building HNSW index (dim={dim}, M={HNSW_M}, n={len(vecs):,}) …")
    t0 = time.perf_counter()
    index.add(vecs)
    elapsed = time.perf_counter() - t0
    print(f"Готово за {elapsed:.1f}s  |  ntotal={index.ntotal:,}")
    return index


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_NAME,
                        help="sentence-transformers model name")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Батч-размер для энкодера (уменьши при нехватке RAM)")
    args = parser.parse_args()

    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(
            f"{CHUNKS_PATH} not found. Run 02_preprocess.py first."
        )

    print(f"Loading model: {args.model}")
    model = SentenceTransformer(args.model)

    accum = encode_and_aggregate(model, CHUNKS_PATH, args.batch_size)
    print(f"  {len(accum):,} уникальных песен")

    song_ids, song_vecs, best_chunk = build_song_vectors(accum)
    del accum  # освобождаем ~sum-векторы

    index = build_hnsw_index(song_vecs)

    song_map = [
        {"song_id": sid, "preview_text": best_chunk[sid]}
        for sid in song_ids
    ]

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    index_path = INDEX_DIR / "song_index.faiss"
    song_map_path = INDEX_DIR / "song_map.pkl"

    faiss.write_index(index, str(index_path))
    with open(song_map_path, "wb") as f:
        pickle.dump(song_map, f)

    print(f"\nСохранено:")
    print(f"  {index_path}  ({index_path.stat().st_size / 1e6:.1f} MB)")
    print(f"  {song_map_path}  ({song_map_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
