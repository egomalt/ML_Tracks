#!/usr/bin/env python3
"""
Замер качества поиска на 10 фиксированных запросах.

Использование:
    python scripts/benchmark.py
    python scripts/benchmark.py --tag "эксперимент-1: базовая модель"
    python scripts/benchmark.py --save   # дописать результаты в logs/benchmark.jsonl

Оценка: для каждого запроса вводите 0-3 (сколько из топ-3 результатов оказались релевантны).
Итоговый балл = среднее × 100 / 3.
"""
import argparse
import json
import pathlib
import sys
import time
from datetime import datetime

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from app.search import SongSearcher

LOGS_DIR = pathlib.Path("logs")

BENCHMARK_QUERIES = [
    # --- Мультиязычные / семантические задачи ---
    {
        "id": "q01",
        "query": "Sad song about missing someone you love",
        "note": "Классическая эмоция — много подходящих песен",
    },
    {
        "id": "q02",
        "query": "Happy upbeat summer song with friends",
        "note": "Позитивное настроение + социальная тема",
    },
    {
        "id": "q03",
        "query": "Motivational rap about achieving goals and success",
        "note": "Жанр + тема",
    },
    {
        "id": "q04",
        "query": "Quiet acoustic folk song about home and childhood",
        "note": "Мягкий жанр + ностальгическая тема",
    },
    {
        "id": "q05",
        "query": "Hard rock song about freedom and rebellion against society",
        "note": "Жанр + тема",
    },
    {
        "id": "q06",
        "query": "Грустная русская песня про расставание с любимым человеком",
        "note": "RU: проверка мультиязычных эмбеддингов",
    },
    {
        "id": "q07",
        "query": "Весёлая детская песня про конфеты и сладости",
        "note": "RU: конкретная тема — проверка фильтрации по метаданным",
    },
    {
        "id": "q08",
        "query": "Романтическая баллада о любви под звёздами",
        "note": "RU: абстрактные образы",
    },
    {
        "id": "q09",
        "query": "Song about loneliness in a big city at night",
        "note": "Абстрактная городская атмосфера",
    },
    {
        "id": "q10",
        "query": "Epic anthem about war and survival",
        "note": "Драматическая + историческая тема",
    },
]


def run_benchmark(searcher: SongSearcher, interactive: bool, top_k: int = 3):
    results = []
    total_score = 0.0

    for q in BENCHMARK_QUERIES:
        print(f"\n{'─'*64}")
        print(f"[{q['id']}] {q['query']}")
        if q.get("note"):
            print(f"     ↳ {q['note']}")

        t0 = time.perf_counter()
        hits = searcher.search(q["query"], top_k=top_k)
        elapsed = time.perf_counter() - t0

        print(f"\n  Результаты  (задержка: {elapsed*1000:.0f}мс):")
        for i, h in enumerate(hits, 1):
            print(f"  {i}. {h['title']} — {h['artist']}  [{h['genre']}]  "
                  f"score={h['score']:.3f}")
            print(f"     \"{h['matched_text'][:90].replace(chr(10), ' ')}…\"")

        score = None
        if interactive:
            while True:
                raw = input(
                    f"\n  Оцените релевантность 0-{top_k} "
                    f"(сколько из {top_k} результатов оказались полезны?): "
                ).strip()
                try:
                    score = int(raw)
                    if 0 <= score <= top_k:
                        break
                except ValueError:
                    pass
                print(f"  Введите число от 0 до {top_k}.")
            total_score += score / top_k

        results.append({
            "query_id": q["id"],
            "query": q["query"],
            "latency_ms": round(elapsed * 1000),
            "hits": hits,
            "score": score,
        })

    return results, total_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tag", default="",
        help="Метка для этого запуска, например 'эксперимент-2: размер чанка 200'"
    )
    parser.add_argument(
        "--save", action="store_true",
        help="Дописать результаты в logs/benchmark.jsonl"
    )
    parser.add_argument(
        "--no-interactive", dest="interactive", action="store_false",
        help="Пропустить ручную оценку (только вывести результаты)"
    )
    parser.set_defaults(interactive=True)
    args = parser.parse_args()

    print("Загрузка поисковика …")
    searcher = SongSearcher()

    results, total_score = run_benchmark(searcher, args.interactive)

    if args.interactive:
        final = total_score / len(BENCHMARK_QUERIES) * 100
        print(f"\n{'═'*64}")
        print(f"  Итоговый балл бенчмарка: {final:.1f} / 100")
        print(f"  (средняя релевантность по {len(BENCHMARK_QUERIES)} запросам)")
    else:
        final = None
        print(f"\nГотово — выполнено {len(BENCHMARK_QUERIES)} запросов.")

    if args.save:
        LOGS_DIR.mkdir(exist_ok=True)
        log_path = LOGS_DIR / "benchmark.jsonl"
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "tag": args.tag,
            "final_score": final,
            "results": results,
        }
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(f"\n  Saved → {log_path}")


if __name__ == "__main__":
    main()
