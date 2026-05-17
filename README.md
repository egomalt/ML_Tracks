# SongFinder — Semantic Song Search

Находит песни по смысловому описанию пользователя.  
Векторный поиск с эмбеддингами (sentence-transformers) + FAISS HNSW индекс + FastAPI бэкенд.

**Датасет:** Genius Song Lyrics (~5M треков), отфильтровано 10 000 английских песен с реальными жанрами.

---

## Быстрый старт (демо-режим, без датасета)

```bash
# 1. Создать окружение
python -m venv .venv

# 2. Активировать окружение (обязательно перед всеми остальными командами)
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/Mac

# 3. Установить зависимости
pip install -r requirements.txt

# 4. Скачать демо-датасет (10 песен, Kaggle не нужен)
python scripts/01_download_data.py --demo

# 5. Разбить тексты на чанки
python scripts/02_preprocess.py

# 6. Построить FAISS-индекс (~10 мин на CPU, скачает модель ~120 МБ)
python scripts/03_build_index.py

# 7. Запустить сервер
uvicorn app.main:app --port 8000
```

Открыть в браузере: http://localhost:8000

> `--reload` добавляй только при разработке — он перезапускает сервер при каждом изменении файла,
> из-за чего модель и индекс перегружаются заново (~30 сек).

---

## Полный датасет (Genius Song Lyrics, ~5M треков)

### Шаг 1 — Kaggle API

1. Зайти на https://www.kaggle.com → Account → Settings → API → **Create New API Token**
2. Скачается файл `kaggle.json` — положить в:
   - Windows: `C:\Users\<user>\.kaggle\kaggle.json`
   - Linux/Mac: `~/.kaggle/kaggle.json`
3. Формат файла: `{"username":"your_username","key":"your_api_key"}`

> Папку `.kaggle` может потребоваться создать вручную.

### Шаг 2 — Скачать датасет

```bash
python scripts/01_download_data.py
```

Скачивает `carlosgdcj/genius-song-lyrics-with-language-information` (~1.8 ГБ) через Kaggle API v1 напрямую (без SDK — обход бага Python 3.13 + urllib3).  
Файл сохраняется в `data/raw/song_lyrics.csv` с колонками: `title, artist, tag, lyrics, language, id`.

### Шаг 3 — Препроцессинг

```bash
python scripts/02_preprocess.py --limit 10000   # 10k английских песен
```

Что происходит:
- Из ~5M треков берутся только `language == "en"` (первые 10 000)
- Текст разбивается на куплеты (разделитель `\n\n`)
- Длинные куплеты режутся на блоки по 3 строки (макс. 300 символов)
- Каждый чанк обогащается метаданными: `"Жанр: pop. Исполнитель: The Weeknd. Текст: ..."`
- Жанры берутся из колонки `tag`: `pop, rock, rap, r-b, country, misc`

### Шаг 4 — Построить индекс

```bash
python scripts/03_build_index.py
```

При первом запуске скачает модель `paraphrase-multilingual-MiniLM-L12-v2` (~120 МБ).  
На CPU (без GPU) 10k песен занимает **~10 минут**.  
Результат: `index/song_index.faiss` + `index/song_map.pkl`.

Что происходит внутри:
1. Каждый чанк кодируется в вектор 384-dim
2. Векторы усредняются по `song_id` → один вектор на трек
3. Усреднённые векторы нормализуются и загружаются в HNSW индекс

- Метрика: inner product (= косинусное сходство для нормированных векторов)
- HNSW параметры: M=32, efConstruction=200, efSearch=100

### Шаг 5 — Запустить сервер

**Важно:** все команды запускать с активированным виртуальным окружением.  
Если окружение не активировано — Python не найдёт faiss, sentence-transformers и другие пакеты.

```bash
# Сначала активировать окружение (один раз за сессию терминала)
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/Mac

# Затем запустить сервер
uvicorn app.main:app --port 8000
```

Или одной строкой без активации:
```bash
.venv\Scripts\uvicorn app.main:app --port 8000
```

Сервер загружает модель и индекс (~30 сек при старте), затем отвечает на запросы.  
Признак успешного старта: `Uvicorn running on http://127.0.0.1:8000`

---

## Структура проекта

```
ml-song-search/
├── scripts/
│   ├── 01_download_data.py   # Загрузка датасета (Kaggle или --demo)
│   ├── 02_preprocess.py      # Нарезка чанков + добавление метаданных
│   ├── 03_build_index.py     # Векторизация + построение FAISS HNSW
│   └── benchmark.py          # Замер качества (10 фиксированных запросов)
├── app/
│   ├── search.py             # Ядро поиска (используется везде)
│   ├── main.py               # FastAPI сервер + перевод цитат (deep-translator)
│   └── static/               # Фронтенд (HTML + CSS + JS)
├── data/
│   ├── raw/                  # Исходные данные (не в git)
│   └── processed/            # songs.json + chunks.jsonl (не в git)
├── index/                    # FAISS-индекс (не в git)
├── predict.py                # Replicate Cog predictor
├── cog.yaml                  # Replicate конфиг
└── requirements.txt
```

---

## Эксперименты

Систематическое сравнение конфигураций — **базовый вариант + 2 группы улучшений**.

### Запуск автоматических экспериментов

```bash
python scripts/04_experiment.py
```

Тестирует два параметра без пересборки индекса и выводит сравнительную таблицу:

**Эксперимент 1 — efSearch (точность HNSW vs скорость)**

| efSearch | score top-1 | latency мс | Вывод |
|---:|---:|---:|---|
| 10 | ~0.72 | ~1 | быстро, качество падает |
| 25 | ~0.75 | ~1 | |
| **50** | **~0.77** | **~2** | **хороший баланс** |
| 100 | ~0.78 | ~3 | текущий дефолт |
| 200 | ~0.78 | ~5 | прирост незначительный |

> При росте efSearch качество растёт быстро до ~50, потом плато. Латентность линейная.

**Эксперимент 2 — top_k (количество результатов)**

| top_k | score top-1 | latency мс |
|---:|---:|---:|
| 1 | ~0.78 | ~2 |
| 3 | ~0.78 | ~3 |
| 5 | ~0.78 | ~3 |
| 10 | ~0.77 | ~4 |

> top-1 score не зависит от top_k — индекс один и тот же. Латентность растёт незначительно.

*Числа в таблицах — ориентировочные; реальные значения зависят от индекса и железа.*

### Ручной бенчмарк качества

```bash
# С оценками 0-3 вручную (сколько из 3 результатов оказались релевантны)
python scripts/benchmark.py --tag "v1: efSearch=100, chunk=300" --save

# Только вывод без оценок
python scripts/benchmark.py --no-interactive
```

Результаты сохраняются в `logs/benchmark.jsonl`.  
Запускайте перед и после каждого изменения — сравнивайте итоговый балл 0–100.

### Что сравнивалось и что выбрали

| Параметр | Базовый | Альтернативы | Итог |
|---|---|---|---|
| Индекс | чанк-level (37k векторов) | **song-level (10k, усреднение)** | song-level лучше — учитывает настроение трека, а не куплета |
| efSearch | 50 | 10 / 25 / **100** / 200 | 100 — стабильное качество, < 5 мс |
| Модель | `MiniLM-L12-v2` (384-dim) | `e5-large` (1024-dim) | MiniLM — быстрее, достаточно для задачи |
| Датасет | Spotify (без жанров) | **Genius (с жанрами, SOTA треки)** | Genius — реальные жанры, узнаваемые треки |

---

## Пайплайн поиска

### Построение индекса (один раз)

```
чанки из 10k песен
      |
SentenceTransformer.encode()  -> вектор 384-dim для каждого чанка
      |
Усреднение по song_id         -> один вектор на песню (= «вектор настроения трека»)
      |
Нормализация                  -> единичная длина (нужна для cosine similarity)
      |
FAISS HNSW.add()              -> индекс из 10k векторов (один на трек)
```

Каждая песня представлена **средним** всех векторов своих куплетов.  
Это значит поиск учитывает настроение всего трека, а не отдельного куплета.

### Поиск по запросу

```
Пользователь: "грустная песня про расставание"
      |
SentenceTransformer.encode(query)  -> вектор 384-dim (нормализованный)
      |
FAISS HNSW.search(vec, k=3)        -> top-3 ближайших песни по cosine similarity
      |
Ответ: title, artist, genre, url, preview_text, score
```

Дедупликация не нужна — в индексе по одному вектору на трек.

### Почему HNSW быстрее полного перебора

HNSW строит многоуровневый граф: на верхних уровнях мало вершин — быстро находим «район» поиска; спускаемся на нижние уровни, уточняем среди ближайших соседей. Сложность O(log n) вместо O(n), на 10k песен поиск занимает менее 5 мс.

---

## Деплой на Replicate

### Шаг 1 — Установить Cog

```powershell
# Через winget (PowerShell от имени администратора)
winget install --id Replicate.Cog
# или через pip (без прав администратора)
.venv\Scripts\pip install cog
```

### Шаг 2 — Создать модель на Replicate

1. Зайти на [replicate.com](https://replicate.com) → Sign up (через GitHub)
2. **Your profile → Create model**, имя: `song-finder`, тип: **Private**
3. Запомнить свой username (виден в URL профиля)

### Шаг 3 — Залогиниться из терминала

```powershell
cog login
```

Откроется браузер → нажать **Authorize** → вернуться в терминал.

### Шаг 4 — Убедиться что индекс построен

Папка `index/` должна содержать `song_index.faiss` и `song_map.pkl`.  
Без них деплой сломается — Cog упаковывает индекс внутрь Docker-образа.

### Шаг 5 — Задеплоить

```powershell
cog push r8.im/ВАШ_USERNAME/song-finder
```

Соберёт Docker-образ с моделью и индексом, загрузит на Replicate (~10–15 минут первый раз).

### Шаг 6 — Открыть

После загрузки: `https://replicate.com/ВАШ_USERNAME/song-finder`

Там будет веб-интерфейс — вводишь запрос, получаешь JSON с треками.

> **Важно**: перед `cog push` пересобери индекс на актуальном датасете (Genius).  
> Индекс упаковывается внутрь образа — после смены датасета нужен повторный `cog push`.

---

## API

### `POST /search`

```json
{ "query": "грустная песня про лето", "top_k": 3, "translate": false }
```

Параметр `translate: true` — переводит цитату из результата на русский (Google Translate, без API-ключа).

Пример реального ответа:
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
{ "status": "ok", "index_size": 10000 }
```

---

## Эксперименты (идеи для улучшения)

| Что менять | Ожидаемый эффект |
|---|---|
| Модель -> `multilingual-e5-large` | Лучше понимает русский, тяжелее (~1 ГБ) |
| Модель -> `all-MiniLM-L6-v2` | Быстрее, только английский |
| Размер чанка: 200 / 400 / 600 символов | Баланс точность/покрытие |
| Фильтр по жанру в поиске | Добавить параметр `genre` в `/search` |
| efSearch: 50 / 100 / 200 | Точность vs скорость поиска |
| Включить не только `en` в фильтре языка | Добавить русские/испанские треки |

Запускайте `benchmark.py --save` перед и после каждого изменения — сравнивайте баллы.
