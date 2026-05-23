# SongFinder

Находит песни по смысловому описанию пользователя.  
Векторный поиск с эмбеддингами (sentence-transformers) + FAISS HNSW индекс + FastAPI бэкенд.

**Датасет:** Genius Song Lyrics (~5M треков), случайная выборка до 1 000 000 en/ru песен с реальными жанрами.  
**Деплой:** [huggingface.co/spaces/egomalt/song-finder](https://huggingface.co/spaces/egomalt/song-finder)

---

## Быстрый старт (демо-режим, без датасета)

```bash
# 1. Создать окружение (только один раз!)
python3 -m venv .venv

# 2. Активировать окружение (нужно при каждом новом терминале)
source .venv/bin/activate       # Linux/Mac
# .venv\Scripts\activate        # Windows

# Признак активации: в начале строки появится (.venv)
# Если не активировать — Python не найдёт установленные пакеты

# 3. Установить зависимости (только один раз, после создания окружения)
bash install.sh

# 4. Скачать демо-датасет (10 песен, Kaggle не нужен)
python scripts/01_download_data.py --demo

# 5. Разбить тексты на чанки
python scripts/02_preprocess.py

# 6. Построить FAISS-индекс (~10 мин на CPU, скачает модель ~470 МБ)
python scripts/03_build_index.py

# 7. Запустить сервер
uvicorn app.main:app --port 8000
```

Открыть в браузере: http://localhost:8000

> `--reload` добавляй только при разработке — он перезапускает сервер при каждом изменении файла,
> из-за чего модель и индекс перегружаются заново (~30 сек).

---

## Полный датасет (Genius Song Lyrics, ~5M треков)

### Шаг 1 — Скачать датасет вручную

1. Зайти на страницу датасета: [Genius Song Lyrics on Kaggle](https://www.kaggle.com/datasets/carlosgdcj/genius-song-lyrics-with-language-information)
2. Войти в Kaggle (бесплатный аккаунт) → нажать **Download** (~1.8 ГБ)
3. Распаковать архив, взять файл `song_lyrics.csv`
4. Положить его в папку `data/raw/`

```
data/
└── raw/
    └── song_lyrics.csv   ← сюда
```

Колонки файла: `title, artist, tag, lyrics, language, id`  
(`tag` — жанр: `pop, rock, rap, r-b, country, misc`)

### Шаг 2 — Препроцессинг

```bash
python scripts/02_preprocess.py
# Спросит интерактивно: сколько песен загрузить?

# Или сразу передать аргументом:
python scripts/02_preprocess.py --limit 100000
python scripts/02_preprocess.py --limit 100000 --seed 123  # другая случайная выборка
```

Что происходит:
- CSV читается потоково батчами по 5 000 строк — весь файл в RAM не грузится
- Reservoir sampling: каждая из ~2.3M en/ru песен имеет равный шанс попасть в выборку
- Текст разбивается на куплеты (разделитель `\n\n`)
- Длинные куплеты режутся на блоки по 3 строки (макс. 300 символов)
- Каждый чанк обогащается метаданными: `"Жанр: pop. Исполнитель: The Weeknd. Текст: ..."`
- Жанры берутся из колонки `tag`: `pop, rock, rap, r-b, country, misc`

### Шаг 3 — Построить индекс

```bash
python scripts/03_build_index.py

# Если не хватает RAM — уменьши батч (по умолчанию 256):
python scripts/03_build_index.py --batch-size 64
```

При первом запуске скачает модель `paraphrase-multilingual-MiniLM-L12-v2` (~470 МБ).  
На CPU (без GPU): 100k песен ≈ **~30 мин**, 1M песен ≈ **~5 часов**.  
Результат: `index/song_index.faiss` + `index/song_map.pkl`.

Что происходит внутри:
1. Чанки читаются потоково батчами — все векторы одновременно в RAM **не держатся**
2. Векторы каждого батча сразу суммируются по `song_id` и выбрасываются
3. После прохода: sum-вектор каждой песни усредняется и нормализуется
4. Нормализованные векторы загружаются в HNSW индекс

- Метрика: inner product (= косинусное сходство для нормированных векторов)
- HNSW параметры: M=32, efConstruction=200, efSearch=100

### Шаг 4 — Запустить сервер

```bash
source .venv/bin/activate   # если ещё не активировано
uvicorn app.main:app --port 8000
```

Сервер загружает модель и индекс (~30 сек при старте), затем отвечает на запросы.  
Признак успешного старта: `Uvicorn running on http://127.0.0.1:8000`

---

## Структура проекта

```
ML_Tracks/
├── scripts/
│   ├── 01_download_data.py   # Загрузка датасета (Kaggle или --demo)
│   ├── 02_preprocess.py      # Потоковая нарезка + reservoir sampling
│   ├── 03_build_index.py     # Векторизация батчами + построение FAISS HNSW
│   ├── 04_experiment.py      # Автоматическое сравнение конфигураций
│   └── benchmark.py          # Ручной замер качества (10 запросов)
├── app/
│   ├── search.py             # Ядро поиска (используется везде)
│   ├── main.py               # FastAPI сервер + перевод цитат (deep-translator)
│   └── static/               # Фронтенд (HTML + CSS + JS)
├── data/
│   ├── raw/                  # Исходные данные (не в git)
│   └── processed/            # songs.json + chunks.jsonl (не в git)
├── index/                    # FAISS-индекс (не в git)
├── Dockerfile                # Образ для деплоя (HuggingFace Spaces)
├── predict.py                # Replicate Cog predictor
├── cog.yaml                  # Replicate конфиг
├── requirements.txt          # Все зависимости (локальная разработка)
├── requirements-prod.txt     # Минимальные зависимости (деплой)
└── install.sh                # Автоустановка зависимостей
```

---

## Деплой на HuggingFace Spaces

Сайт задеплоен на [huggingface.co/spaces/egomalt/song-finder](https://huggingface.co/spaces/egomalt/song-finder) — бесплатно, 16 ГБ RAM, без карты.

Чтобы обновить деплой после изменений (пересборки индекса и т.д.):

```bash
# Установить git-lfs если нет
sudo pacman -S git-lfs   # Arch Linux
# brew install git-lfs   # Mac

git lfs install

# Клонировать Space
git clone https://ТОКЕН@huggingface.co/spaces/egomalt/song-finder ~/song-finder-space

# Скопировать обновлённые файлы
cp -r app index data/processed/songs.json Dockerfile requirements-prod.txt ~/song-finder-space/

# Запушить
cd ~/song-finder-space
git add .
git commit -m "Update index"
git push
```

Токен генерируется на **huggingface.co → Settings → Access Tokens** (роль Write).

---

## Деплой на Replicate (только API, без фронтенда)

### Шаг 1 — Установить Cog

```bash
curl -o ~/.local/bin/cog -L https://github.com/replicate/cog/releases/download/v0.20.0/cog_Linux_x86_64
chmod +x ~/.local/bin/cog
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc && source ~/.bashrc
```

### Шаг 2 — Создать модель на Replicate

1. Зайти на [replicate.com](https://replicate.com) → Sign up (через GitHub)
2. **Your profile → Create model**, имя: `song-finder`, тип: **Private**

### Шаг 3 — Залогиниться и задеплоить

```bash
cog login --browserless
cog push r8.im/ВАШ_USERNAME/song-finder
```

> Индекс упаковывается внутрь образа — после смены датасета нужен повторный `cog push`.

---

## API

### `POST /search`

```json
{ "query": "грустная песня про лето", "top_k": 3, "translate": false, "genre": "pop" }
```

| Параметр | Тип | По умолчанию | Описание |
|---|---|---|---|
| `query` | string | — | Описание песни (любой язык) |
| `top_k` | int | 3 | Количество результатов (1–10) |
| `translate` | bool | false | Перевести цитату на русский (Google Translate) |
| `genre` | string\|null | null | Фильтр по жанру: `pop`, `rock`, `rap`, `r-b`, `country`, `misc` |

Ранжирование: `0.85 × cosine_similarity + 0.15 × popularity_score`  
`popularity_score` — нормализованный log(views) из Genius.

Пример ответа:
```json
{
  "query": "sad love song about missing someone",
  "results": [
    {
      "title": "Call Out My Name",
      "artist": "The Weeknd",
      "genre": "r-b",
      "url": "https://genius.com/songs/3964495",
      "matched_text": "I put you on top, I put you on top\nI claimed you so proud and openly\nAnd when times were rough I made sure I held you close to me",
      "score": 0.81
    }
  ]
}
```

### `GET /health`

```json
{ "status": "ok", "index_size": 100000 }
```

---

## Эксперименты

### Автоматический

```bash
python scripts/04_experiment.py
```

Тестирует два параметра без пересборки индекса и выводит сравнительную таблицу:

**efSearch (точность HNSW vs скорость)**

| efSearch | score top-1 | latency мс | Вывод |
|---:|---:|---:|---|
| 10 | ~0.72 | ~1 | быстро, качество падает |
| 25 | ~0.75 | ~1 | |
| **50** | **~0.77** | **~2** | **хороший баланс** |
| 100 | ~0.78 | ~3 | текущий дефолт |
| 200 | ~0.78 | ~5 | прирост незначительный |

**top_k (количество результатов)**

| top_k | score top-1 | latency мс |
|---:|---:|---:|
| 1 | ~0.78 | ~2 |
| 3 | ~0.78 | ~3 |
| 5 | ~0.78 | ~3 |
| 10 | ~0.77 | ~4 |

### Ручной бенчмарк

```bash
python scripts/benchmark.py --tag "v1: efSearch=100" --save
python scripts/benchmark.py --no-interactive   # только вывод
```

Результаты сохраняются в `logs/benchmark.jsonl`. Запускай до и после изменений.

### Что сравнивалось

| Параметр | Базовый | Итог |
|---|---|---|
| Индекс | чанк-level | song-level (усреднение) — учитывает настроение трека целиком |
| efSearch | 50 | 100 — стабильное качество, < 5 мс |
| Модель | `MiniLM-L12-v2` (384-dim) | MiniLM — быстрее, достаточно для задачи |
| Датасет | Spotify (без жанров) | Genius — реальные жанры, узнаваемые треки |

---

## Пайплайн

### Построение индекса (один раз)

```
чанки из N песен (потоком, батчами по 256)
      |
SentenceTransformer.encode()  -> вектор 384-dim для батча
      |
Суммирование по song_id       -> sum-вектор на песню (чанки сразу выбрасываются)
      |
mean + нормализация           -> единичная длина (cosine similarity)
      |
FAISS HNSW.add()              -> индекс из N векторов (один на трек)
```

### Поиск по запросу

```
Пользователь: "грустная песня про расставание"
      |
SentenceTransformer.encode(query)  -> вектор 384-dim (нормализованный)
      |
FAISS HNSW.search(vec, k=3)        -> top-3 ближайших по cosine similarity
      |
Ответ: title, artist, genre, url, preview_text, score
```

HNSW строит многоуровневый граф: сложность O(log n) вместо O(n), на 100k–1M песен поиск занимает менее 10 мс.

---

## Идеи для улучшения

| Что менять | Ожидаемый эффект |
|---|---|
| Модель → `multilingual-e5-large` | Лучше понимает русский, тяжелее (~1 ГБ) |
| Модель → `all-MiniLM-L6-v2` | Быстрее, только английский |
| Размер чанка: 200 / 400 / 600 символов | Баланс точность/покрытие |
| efSearch: 50 / 100 / 200 | Точность vs скорость поиска |
| Добавить испанские/французские треки | Расширить `language.isin(["en","ru","es","fr"])` |
| Фильтрация explicit-треков | Убирать нецензурный контент по ключевым словам |
