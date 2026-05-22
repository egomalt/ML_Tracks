#!/bin/bash
set -e

python3 -m pip install --upgrade pip

# torch тащится как зависимость sentence-transformers — ставим отдельно,
# чтобы pip не пытался собирать из исходников и не съедал всю память
pip install --only-binary=:all: torch --index-url https://download.pytorch.org/whl/cpu

# тяжёлые пакеты — только готовые wheels, без компиляции
pip install --only-binary=:all: \
    numpy==2.2.6 \
    faiss-cpu==1.13.2 \
    sentence-transformers==5.5.1

# остальные зависимости
pip install \
    fastapi==0.136.1 \
    "uvicorn[standard]==0.47.0" \
    python-multipart==0.0.29 \
    pandas==2.2.3 \
    tqdm==4.67.3 \
    kaggle==2.1.2 \
    lyricsgenius==3.12.2 \
    httpx==0.28.1 \
    deep-translator==1.11.4

echo "Done."
