#!/usr/bin/env python3
"""
Предобработка сырых данных о песнях: нарезка на чанки для векторизации.

Вход:
  data/raw/song_lyrics.csv      (датасет Kaggle: Genius Song Lyrics)
  ИЛИ data/raw/songs_demo.jsonl (демо-вариант)

Выход:
  data/processed/songs.json     {song_id -> метаданные}
  data/processed/chunks.jsonl   один чанк на строку
"""
import json
import pathlib
import re
import sys
import argparse
import hashlib

import pandas as pd
from tqdm import tqdm

RAW_DIR = pathlib.Path("data/raw")
PROCESSED_DIR = pathlib.Path("data/processed")

KAGGLE_CSV = RAW_DIR / "song_lyrics.csv"
DEMO_JSONL = RAW_DIR / "songs_demo.jsonl"

MAX_CHUNK_CHARS = 300
MIN_CHUNK_CHARS = 30
MAX_SONGS = 10_000   # ограничение по памяти/скорости; 0 = без ограничений


def make_song_id(artist: str, title: str) -> str:
    raw = f"{artist}|{title}".lower()
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def clean_text(text: str) -> str:
    text = re.sub(r"\[.*?\]", "", text)       # убираем [Chorus], [Verse 1] и т.д.
    text = re.sub(r"\(.*?\)", "", text)        # убираем (repeat) и похожие метки
    text = re.sub(r" {2,}", " ", text)
    text = text.strip()
    return text


def split_chunks(lyrics: str) -> list[str]:
    """Разбивает текст песни на куплеты-чанки."""
    paragraphs = re.split(r"\n{2,}", lyrics)
    chunks = []
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(para) <= MAX_CHUNK_CHARS:
            chunks.append(para)
        else:
            # длинные абзацы режем на группы по 3 строки
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
    """Добавляет метаданные перед текстом — жанр и исполнитель «запекаются» в эмбеддинг."""
    parts = []
    if genre and genre.lower() not in ("unknown", ""):
        parts.append(f"Жанр: {genre}.")
    parts.append(f"Исполнитель: {artist}.")
    parts.append(f"Текст: {chunk}")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Загрузчики данных
# ---------------------------------------------------------------------------

def load_kaggle_csv(path: pathlib.Path, limit: int) -> list[dict]:
    print(f"Loading {path} …")
    df = pd.read_csv(path, usecols=["title", "artist", "tag", "lyrics", "language", "id", "views"])
    df = df.dropna(subset=["lyrics", "artist", "title"])
    df = df[df["lyrics"].str.len() > 100]
    # английские и русские треки (модель мультиязычная)
    df = df[df["language"].isin(["en", "ru"])]
    if limit:
        df = df.head(limit)
    songs = []
    for _, row in df.iterrows():
        genius_id = row.get("id", "")
        url = f"https://genius.com/songs/{int(genius_id)}" if pd.notna(genius_id) and str(genius_id).isdigit() else ""
        views = int(row["views"]) if pd.notna(row.get("views")) else 0
        songs.append({
            "title": str(row["title"]).strip(),
            "artist": str(row["artist"]).strip(),
            "genre": str(row.get("tag", "Unknown")).strip(),
            "url": url,
            "lyrics": str(row["lyrics"]),
            "views": views,
        })
    return songs


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
        "No data found. Run scripts/01_download_data.py first.\n"
        "Quick start: python scripts/01_download_data.py --demo"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=MAX_SONGS,
                        help="Max number of songs to process (0 = all)")
    args = parser.parse_args()

    source, path = detect_source()
    if source == "kaggle":
        raw_songs = load_kaggle_csv(path, args.limit)
    else:
        raw_songs = load_demo_jsonl(path)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    songs_meta: dict[str, dict] = {}
    total_chunks = 0

    chunks_path = PROCESSED_DIR / "chunks.jsonl"
    with open(chunks_path, "w", encoding="utf-8") as out:
        for song in tqdm(raw_songs, desc="Processing songs"):
            song_id = make_song_id(song["artist"], song["title"])
            lyrics = clean_text(song["lyrics"])
            chunks = split_chunks(lyrics)
            if not chunks:
                continue

            songs_meta[song_id] = {
                "title": song["title"],
                "artist": song["artist"],
                "genre": song.get("genre", "Unknown"),
                "url": song.get("url", ""),
                "views": song.get("views", 0),
            }

            for i, chunk in enumerate(chunks):
                record = {
                    "song_id": song_id,
                    "chunk_index": i,
                    "text": chunk,
                    "vectorise_text": make_vectorise_text(
                        chunk, song["artist"], song.get("genre", "Unknown")
                    ),
                }
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                total_chunks += 1

    songs_path = PROCESSED_DIR / "songs.json"
    with open(songs_path, "w", encoding="utf-8") as f:
        json.dump(songs_meta, f, ensure_ascii=False, indent=2)

    print(f"\nDone. {len(songs_meta)} songs -> {total_chunks} chunks")
    print(f"  {songs_path}")
    print(f"  {chunks_path}")


if __name__ == "__main__":
    main()
