#!/usr/bin/env python3
"""
Загрузка датасета с песнями.

Основной источник:  Kaggle 'carlosgdcj/genius-song-lyrics-with-language-information'
                    Колонки: title, artist, tag, lyrics, language, id
Запасной вариант:   флаг --demo генерирует маленький тестовый датасет
"""
import argparse
import json
import pathlib
import zipfile

RAW_DIR = pathlib.Path("data/raw")
KAGGLE_DATASET = "carlosgdcj/genius-song-lyrics-with-language-information"
KAGGLE_FILE = "song_lyrics.csv"

# ---------------------------------------------------------------------------
# Демо-данные — публичный домен / вымышленные отрывки, без проблем с авторством
# ---------------------------------------------------------------------------
DEMO_SONGS = [
    {"title": "Bohemian Rhapsody", "artist": "Queen", "genre": "Rock",
     "url": "https://www.youtube.com/watch?v=fJ9rUzIMcZQ",
     "lyrics": (
         "Is this the real life? Is this just fantasy?\n"
         "Caught in a landslide, no escape from reality\n\n"
         "Open your eyes, look up to the skies and see\n"
         "I'm just a poor boy, I need no sympathy\n\n"
         "Easy come, easy go, will you let me go?\n"
         "Bismillah! No, we will not let you go"
     )},
    {"title": "Hotel California", "artist": "Eagles", "genre": "Rock",
     "url": "https://www.youtube.com/watch?v=BciS5krYL80",
     "lyrics": (
         "On a dark desert highway, cool wind in my hair\n"
         "Warm smell of colitas rising up through the air\n\n"
         "Up ahead in the distance I saw a shimmering light\n"
         "My head grew heavy and my sight grew dim\n\n"
         "Welcome to the Hotel California\n"
         "Such a lovely place, such a lovely face"
     )},
    {"title": "Смерть на верхних этажах", "artist": "Земляне", "genre": "Pop",
     "url": "https://www.youtube.com/watch?v=example",
     "lyrics": (
         "Я люблю тебя, жизнь, что само по себе и не ново\n"
         "Я люблю тебя, жизнь, я люблю тебя снова и снова\n\n"
         "Вот уж осень пришла, И короче стал день\n"
         "Потому что уходит, Медленно тень"
     )},
    {"title": "Что такое осень", "artist": "ДДТ", "genre": "Rock",
     "url": "https://www.youtube.com/watch?v=example2",
     "lyrics": (
         "Что такое осень — это небо\n"
         "Плачущее небо под ногами\n\n"
         "В лужах разбивается на стёкла\n"
         "На чёрные стёкла\n\n"
         "Осень, я давно с тобою не был\n"
         "Осень, мне тебя не жаль"
     )},
    {"title": "Shape of You", "artist": "Ed Sheeran", "genre": "Pop",
     "url": "https://www.youtube.com/watch?v=JGwWNGJdvx8",
     "lyrics": (
         "The club isn't the best place to find a lover\n"
         "So the bar is where I go\n\n"
         "Me and my friends at the table doing shots\n"
         "Drinking fast and then we talk slow\n\n"
         "I'm in love with your body\n"
         "Every day discovering something brand new"
     )},
    {"title": "Lose Yourself", "artist": "Eminem", "genre": "Hip-Hop",
     "url": "https://www.youtube.com/watch?v=_Yhyp-_hX2s",
     "lyrics": (
         "Look, if you had one shot, or one opportunity\n"
         "To seize everything you ever wanted, one moment\n"
         "Would you capture it, or just let it slip?\n\n"
         "His palms are sweaty, knees weak, arms are heavy\n"
         "There's vomit on his sweater already, mom's spaghetti"
     )},
    {"title": "Кукушка", "artist": "Кино", "genre": "Rock",
     "url": "https://www.youtube.com/watch?v=example3",
     "lyrics": (
         "Прогулки по воде, прогулки по воде\n"
         "Прогулки по воде с Тобой\n\n"
         "Мне хотелось бы остаться\n"
         "Мне хотелось бы остаться\n\n"
         "Сколько раз упасть и встать\n"
         "Сколько раз начать опять"
     )},
    {"title": "Someone Like You", "artist": "Adele", "genre": "Pop",
     "url": "https://www.youtube.com/watch?v=hLQl3WQQoQ0",
     "lyrics": (
         "I heard that you're settled down\n"
         "That you found a girl and you're married now\n\n"
         "I heard that your dreams came true\n"
         "Guess she gave you things I didn't give to you\n\n"
         "Never mind, I'll find someone like you\n"
         "I wish nothing but the best for you too"
     )},
    {"title": "Creep", "artist": "Radiohead", "genre": "Alternative",
     "url": "https://www.youtube.com/watch?v=XFkzRNyygfk",
     "lyrics": (
         "When you were here before\n"
         "Couldn't look you in the eye\n\n"
         "You're just like an angel\n"
         "Your skin makes me cry\n\n"
         "But I'm a creep, I'm a weirdo\n"
         "What the hell am I doing here?\n"
         "I don't belong here"
     )},
    {"title": "Дорогой длинною", "artist": "Борис Фомин", "genre": "Romance",
     "url": "https://www.youtube.com/watch?v=example4",
     "lyrics": (
         "Дорогой длинною да ночью лунною\n"
         "Да с песней той, что вдаль летит\n\n"
         "Со старой скрипкою да с думой грустною\n"
         "О той любви, что не горит\n\n"
         "Эх, дороги, пыль да туман\n"
         "Холода, тревоги да степной бурьян"
     )},
]


def download_kaggle() -> bool:
    """Скачивает датасет через Kaggle API v1 напрямую (без kaggle SDK)."""
    try:
        import httpx

        creds_path = pathlib.Path.home() / ".kaggle" / "kaggle.json"
        creds = json.loads(creds_path.read_text(encoding="utf-8"))
        key = creds["key"]

        owner, slug = KAGGLE_DATASET.split("/")
        api_url = (
            f"https://www.kaggle.com/api/v1/datasets/download"
            f"/{owner}/{slug}/{KAGGLE_FILE}"
        )

        RAW_DIR.mkdir(parents=True, exist_ok=True)
        zip_path = RAW_DIR / "dataset_tmp.zip"

        print(f"Downloading {KAGGLE_DATASET} …")
        with httpx.Client(follow_redirects=True, timeout=600.0) as client:
            with client.stream(
                "GET", api_url, headers={"Authorization": f"Bearer {key}"}
            ) as r:
                r.raise_for_status()
                total = int(r.headers.get("content-length", 0))
                done = 0
                with open(zip_path, "wb") as f:
                    for chunk in r.iter_bytes(chunk_size=1024 * 1024):
                        f.write(chunk)
                        done += len(chunk)
                        if total:
                            print(
                                f"\r  {done // 1_000_000} / {total // 1_000_000} MB"
                                f"  ({100 * done // total}%)",
                                end="", flush=True,
                            )
        print()

        dst = RAW_DIR / KAGGLE_FILE
        print("Extracting …")
        with zipfile.ZipFile(zip_path) as z:
            # файл может лежать в корне архива или в подпапке
            names = z.namelist()
            match = next((n for n in names if n.endswith(KAGGLE_FILE)), names[0])
            with z.open(match) as src, open(dst, "wb") as out:
                out.write(src.read())
        zip_path.unlink()

        print(f"Saved to {dst}")
        return True
    except Exception as exc:
        print(f"[warn] Kaggle download failed: {exc}")
        return False


def write_demo() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out = RAW_DIR / "songs_demo.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for song in DEMO_SONGS:
            f.write(json.dumps(song, ensure_ascii=False) + "\n")
    print(f"Demo dataset written -> {out}  ({len(DEMO_SONGS)} songs)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--demo", action="store_true",
        help="Skip Kaggle, write tiny demo dataset instead"
    )
    args = parser.parse_args()

    if args.demo:
        write_demo()
        return

    ok = download_kaggle()
    if not ok:
        print("Falling back to demo dataset.")
        write_demo()


if __name__ == "__main__":
    main()
