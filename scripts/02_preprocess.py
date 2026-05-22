#!/usr/bin/env python3
"""
Предобработка сырых данных о песнях: нарезка на чанки для векторизации.

Вход:
  data/raw/song_lyrics.csv      (датасет Kaggle: Genius Song Lyrics)
  ИЛИ data/raw/songs_demo.jsonl (демо-вариант)

Выход:
  data/processed/songs.json     {song_id -> метаданные}
  data/processed/chunks.jsonl   один чанк на строку

Алгоритм выборки: reservoir sampling — читает CSV чанками по CHUNK_SIZE строк,
никогда не держит в RAM больше чем CHUNK_SIZE + limit строк.
"""
import argparse
import hashlib
import json
import pathlib
import random
import re
import sys

import pandas as pd
from tqdm import tqdm

RAW_DIR = pathlib.Path("data/raw")
PROCESSED_DIR = pathlib.Path("data/processed")

KAGGLE_CSV = RAW_DIR / "song_lyrics.csv"
DEMO_JSONL = RAW_DIR / "songs_demo.jsonl"

MAX_CHUNK_CHARS = 300
MIN_CHUNK_CHARS = 30
MAX_SONGS  = 100_000
CHUNK_SIZE = 5_000   # строк за один read — столько держим в RAM единовременно


# ---------------------------------------------------------------------------
# Утилиты
# ---------------------------------------------------------------------------

def make_song_id(artist: str, title: str) -> str:
    raw = f"{artist}|{title}".lower()
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def clean_text(text: str) -> str:
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"\(.*?\)", "", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def split_chunks(lyrics: str) -> list[str]:
    paragraphs = re.split(r"\n{2,}", lyrics)
    chunks = []
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(para) <= MAX_CHUNK_CHARS:
            chunks.append(para)
        else:
            lines = [l.strip() for l in para.splitlines() if l.strip()]
            group: list[str] = []
            for line in lines:
                group.append(line)
                joined = "\n".join(group)
                if len(joined) >= MAX_CHUNK_CHARS:
                    if joined:
                        chunks.append(joined)
                    group = []
            if group:
                chunks.append("\n".join(group))
    return [c for c in chunks if len(c) >= MIN_CHUNK_CHARS]


def make_vectorise_text(chunk: str, artist: str, genre: str) -> str:
    parts = []
    if genre and genre.lower() not in ("unknown", ""):
        parts.append(f"Жанр: {genre}.")
    parts.append(f"Исполнитель: {artist}.")
    parts.append(f"Текст: {chunk}")
    return " ".join(parts)


def _row_to_song(row) -> dict:
    genius_id = getattr(row, "id", "")
    url = (
        f"https://genius.com/songs/{int(genius_id)}"
        if pd.notna(genius_id) and str(genius_id).isdigit()
        else ""
    )
    views_raw = getattr(row, "views", None)
    views = int(views_raw) if pd.notna(views_raw) else 0
    return {
        "title":  str(row.title).strip(),
        "artist": str(row.artist).strip(),
        "genre":  str(getattr(row, "tag", "Unknown")).strip(),
        "url":    url,
        "lyrics": str(row.lyrics),
        "views":  views,
    }


# ---------------------------------------------------------------------------
# Загрузчики данных
# ---------------------------------------------------------------------------

def load_kaggle_csv(path: pathlib.Path, limit: int, seed: int) -> list[dict]:
    """Reservoir sampling: читает CSV потоково, в RAM держит не более
    CHUNK_SIZE + limit строк одновременно."""
    rng = random.Random(seed)
    reservoir: list[dict] = []
    n_seen = 0

    cols = ["title", "artist", "tag", "lyrics", "language", "id", "views"]
    reader = pd.read_csv(
        path,
        usecols=cols,
        chunksize=CHUNK_SIZE,
        low_memory=False,
    )

    print(f"Streaming {path}  (target: {limit:,} random songs, seed={seed}) …")
    for chunk in reader:
        chunk = chunk.dropna(subset=["lyrics", "artist", "title"])
        chunk = chunk[chunk["lyrics"].str.len() > 100]
        chunk = chunk[chunk["language"].isin(["en", "ru"])]

        for row in chunk.itertuples(index=False):
            n_seen += 1
            song = _row_to_song(row)
            if n_seen <= limit:
                reservoir.append(song)
            else:
                # reservoir sampling: заменяем случайный элемент
                j = rng.randint(0, n_seen - 1)
                if j < limit:
                    reservoir[j] = song

        print(
            f"\r  подходящих песен: {n_seen:,} | отобрано: {len(reservoir):,}",
            end="", flush=True,
        )

    print(f"\nГотово: отобрано {len(reservoir):,} из {n_seen:,} подходящих (en/ru) песен.")
    return reservoir


def load_demo_jsonl(path: pathlib.Path) -> list[dict]:
    songs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            songs.append(json.loads(line))
    return songs


def detect_source() -> tuple[str, pathlib.Path]:
    if KAGGLE_CSV.exists():
        return "kaggle", KAGGLE_CSV
    if DEMO_JSONL.exists():
        return "demo", DEMO_JSONL
    sys.exit(
        "Данные не найдены. Сначала запусти scripts/01_download_data.py\n"
        "Быстрый старт: python scripts/01_download_data.py --demo"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                        help="Сколько песен отобрать (0 = все)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed для воспроизводимой случайной выборки")
    args = parser.parse_args()

    if args.limit is None:
        print("Сколько песен загрузить? (например: 10000, 50000, 100000 — или 0 для всех)")
        raw = input("  > ").strip()
        try:
            args.limit = int(raw)
        except ValueError:
            sys.exit(f"Ошибка: '{raw}' — не число.")

    source, path = detect_source()
    if source == "kaggle":
        raw_songs = load_kaggle_csv(path, args.limit or 10**9, args.seed)
    else:
        raw_songs = load_demo_jsonl(path)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    songs_meta: dict[str, dict] = {}
    total_chunks = 0

    chunks_path = PROCESSED_DIR / "chunks.jsonl"
    with open(chunks_path, "w", encoding="utf-8") as out:
        for song in tqdm(raw_songs, desc="Нарезка чанков"):
            song_id = make_song_id(song["artist"], song["title"])
            lyrics = clean_text(song["lyrics"])
            chunks = split_chunks(lyrics)
            if not chunks:
                continue

            songs_meta[song_id] = {
                "title":  song["title"],
                "artist": song["artist"],
                "genre":  song.get("genre", "Unknown"),
                "url":    song.get("url", ""),
                "views":  song.get("views", 0),
            }

            for i, chunk in enumerate(chunks):
                record = {
                    "song_id":       song_id,
                    "chunk_index":   i,
                    "text":          chunk,
                    "vectorise_text": make_vectorise_text(
                        chunk, song["artist"], song.get("genre", "Unknown")
                    ),
                }
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                total_chunks += 1

    songs_path = PROCESSED_DIR / "songs.json"
    with open(songs_path, "w", encoding="utf-8") as f:
        json.dump(songs_meta, f, ensure_ascii=False, indent=2)

    print(f"\nГотово: {len(songs_meta)} песен → {total_chunks} чанков")
    print(f"  {songs_path}")
    print(f"  {chunks_path}")


if __name__ == "__main__":
    main()
