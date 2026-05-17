#!/usr/bin/env python3
"""
Систематическое сравнение конфигураций поисковика.

Что тестируется:
  Эксперимент 1 — efSearch (точность HNSW vs скорость): 10 / 25 / 50 / 100 / 200
  Эксперимент 2 — top_k (количество результатов): 1 / 3 / 5 / 10

Метрика: средний cosine-score первого результата по 10 запросам (прокси-качество).
  Выше = модель находит более уверенные совпадения.

Использование:
    python scripts/04_experiment.py
    python scripts/04_experiment.py --no-save   # только вывод, без записи в файл

Результаты сохраняются в logs/experiments.jsonl.
"""
import argparse
import json
import pathlib
import sys
import time
from datetime import datetime

# добавляем корень проекта и папку scripts в sys.path
ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from app.search import SongSearcher  # noqa: E402
from benchmark import BENCHMARK_QUERIES  # noqa: E402

LOGS_DIR = ROOT / "logs"


def run_config(searcher: SongSearcher, ef_search: int, top_k: int) -> dict:
    """Прогоняет все запросы бенчмарка, возвращает агрегированные метрики."""
    searcher._index.hnsw.efSearch = ef_search

    top1_scores: list[float] = []
    latencies: list[float] = []

    for q in BENCHMARK_QUERIES:
        t0 = time.perf_counter()
        hits = searcher.search(q["query"], top_k=top_k)
        latencies.append((time.perf_counter() - t0) * 1000)
        if hits:
            top1_scores.append(hits[0]["score"])

    return {
        "ef_search": ef_search,
        "top_k": top_k,
        "mean_top1_score": round(sum(top1_scores) / len(top1_scores), 4) if top1_scores else 0.0,
        "mean_latency_ms": round(sum(latencies) / len(latencies), 1),
        "n_queries": len(BENCHMARK_QUERIES),
    }


def print_table(title: str, rows: list[dict], col_ef: bool = True) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    header = f"{'efSearch':>10}  {'top_k':>6}  {'score top-1':>12}  {'latency ms':>12}"
    print(header)
    print("-" * 46)
    for r in rows:
        print(
            f"{r['ef_search']:>10}  {r['top_k']:>6}  "
            f"{r['mean_top1_score']:>12.4f}  {r['mean_latency_ms']:>10.1f}"
        )
    best = max(rows, key=lambda x: x["mean_top1_score"])
    fastest = min(rows, key=lambda x: x["mean_latency_ms"])
    print(f"\n  Лучший score:   efSearch={best['ef_search']}, top_k={best['top_k']} "
          f"-> score={best['mean_top1_score']:.4f}")
    print(f"  Быстрейший:     efSearch={fastest['ef_search']}, top_k={fastest['top_k']} "
          f"-> {fastest['mean_latency_ms']:.1f} мс")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-save", action="store_true", help="Не сохранять результаты")
    args = parser.parse_args()

    print("Загрузка поисковика (модель + индекс) …")
    searcher = SongSearcher()

    all_results: list[dict] = []

    # ── Эксперимент 1: efSearch ──────────────────────────────────────────────
    print("\nЭксперимент 1: варьируем efSearch, top_k=3 …")
    ef_rows = []
    for ef in [10, 25, 50, 100, 200]:
        r = run_config(searcher, ef_search=ef, top_k=3)
        ef_rows.append(r)
        all_results.append({"experiment": "efSearch", **r})
        print(f"  efSearch={ef:>3}  score={r['mean_top1_score']:.4f}  "
              f"latency={r['mean_latency_ms']:.1f}мс")

    print_table("Эксперимент 1 — efSearch vs качество/скорость", ef_rows)

    # ── Эксперимент 2: top_k ─────────────────────────────────────────────────
    print("\nЭксперимент 2: варьируем top_k, efSearch=100 …")
    topk_rows = []
    for k in [1, 3, 5, 10]:
        r = run_config(searcher, ef_search=100, top_k=k)
        topk_rows.append(r)
        all_results.append({"experiment": "top_k", **r})
        print(f"  top_k={k:>2}  score={r['mean_top1_score']:.4f}  "
              f"latency={r['mean_latency_ms']:.1f}мс")

    print_table("Эксперимент 2 — top_k vs качество/скорость", topk_rows)

    # ── Сохранение ───────────────────────────────────────────────────────────
    if not args.no_save:
        LOGS_DIR.mkdir(exist_ok=True)
        log_path = LOGS_DIR / "experiments.jsonl"
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "index_size": searcher._index.ntotal,
            "results": all_results,
        }
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(f"\n  Результаты сохранены -> {log_path}")

    print("\nПримечание:")
    print("  score top-1 = средний cosine similarity первого результата")
    print("  Диапазон 0–1; выше = модель увереннее находит совпадение.")
    print("  Дополни оценку вручную: python scripts/benchmark.py --save")


if __name__ == "__main__":
    main()
