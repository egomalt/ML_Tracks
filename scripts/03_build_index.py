#!/usr/bin/env python3
"""
Векторизация чанков через sentence-transformers и построение FAISS HNSW индекса.

Вектор каждой песни = среднее всех векторов её чанков (нормализованное).
Это позволяет искать по настроению всего трека, а не по отдельному куплету.

Вход:  data/processed/chunks.jsonl  +  data/processed/songs.json
Выход: index/song_index.faiss   — один вектор на песню
       index/song_map.pkl       — список {song_id, best_chunk_text} по позиции в индексе
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

# M=32: каждый узел имеет до 32 двунаправленных связей на слой
# efConstruction=200: чем выше — тем лучше качество индекса, но дольше построение
HNSW_M = 32
HNSW_EF_CONSTRUCTION = 200
HNSW_EF_SEARCH = 100
BATCH_SIZE = 256

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"


def load_chunks(path: pathlib.Path) -> tuple[list[dict], list[str]]:
    chunks, texts = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            chunks.append({
                "song_id": rec["song_id"],
                "chunk_index": rec["chunk_index"],
                "text": rec["text"],
            })
            texts.append(rec["vectorise_text"])
    return chunks, texts


def encode_in_batches(
    model: SentenceTransformer,
    texts: list[str],
    batch_size: int,
) -> np.ndarray:
    all_vecs = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
        batch = texts[i : i + batch_size]
        vecs = model.encode(
            batch,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        all_vecs.append(vecs)
    return np.vstack(all_vecs).astype("float32")


def aggregate_by_song(
    chunks: list[dict],
    chunk_vecs: np.ndarray,
) -> tuple[list[str], np.ndarray, dict[str, str]]:
    """
    Усредняет векторы чанков по песне -> один вектор на трек.

    Возвращает:
      song_ids   — список song_id в порядке строк индекса
      song_vecs  — матрица (n_songs, dim), нормализованная
      best_chunk — {song_id -> текст первого чанка} для превью
    """
    buckets: dict[str, list[int]] = defaultdict(list)
    for i, chunk in enumerate(chunks):
        buckets[chunk["song_id"]].append(i)

    song_ids: list[str] = []
    averaged: list[np.ndarray] = []
    best_chunk: dict[str, str] = {}

    for song_id, idxs in tqdm(buckets.items(), desc="Averaging"):
        vecs = chunk_vecs[idxs]
        mean_vec = vecs.mean(axis=0)
        # нормализуем среднее — cosine similarity работает только с единичными векторами
        norm = np.linalg.norm(mean_vec)
        if norm > 0:
            mean_vec /= norm
        averaged.append(mean_vec)
        song_ids.append(song_id)
        # сохраняем первый чанк как превью-цитату
        first_idx = idxs[0]
        best_chunk[song_id] = chunks[first_idx]["text"]

    return song_ids, np.vstack(averaged).astype("float32"), best_chunk


def build_hnsw_index(vecs: np.ndarray) -> faiss.IndexHNSWFlat:
    dim = vecs.shape[1]
    # IndexHNSWFlat со скалярным произведением (= косинусное сходство для нормированных векторов)
    index = faiss.IndexHNSWFlat(dim, HNSW_M, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = HNSW_EF_CONSTRUCTION
    index.hnsw.efSearch = HNSW_EF_SEARCH
    print(f"Building HNSW index (dim={dim}, M={HNSW_M}) over {len(vecs)} vectors ...")
    t0 = time.perf_counter()
    index.add(vecs)
    elapsed = time.perf_counter() - t0
    print(f"Index built in {elapsed:.1f}s  |  ntotal={index.ntotal}")
    return index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_NAME,
                        help="sentence-transformers model name")
    args = parser.parse_args()

    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(
            f"{CHUNKS_PATH} not found. Run 02_preprocess.py first."
        )

    print(f"Loading model: {args.model}")
    model = SentenceTransformer(args.model)

    print("Loading chunks ...")
    chunks, texts = load_chunks(CHUNKS_PATH)
    print(f"  {len(chunks)} chunks loaded")

    chunk_vecs = encode_in_batches(model, texts, BATCH_SIZE)

    song_ids, song_vecs, best_chunk = aggregate_by_song(chunks, chunk_vecs)
    print(f"  {len(song_ids)} unique songs")

    index = build_hnsw_index(song_vecs)

    # song_map: список записей в том же порядке, что строки индекса
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

    print(f"\nSaved:")
    print(f"  {index_path}  ({index_path.stat().st_size / 1e6:.1f} MB)")
    print(f"  {song_map_path}  ({song_map_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
