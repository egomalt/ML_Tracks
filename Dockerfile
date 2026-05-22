FROM python:3.11-slim

WORKDIR /app

# зависимости отдельным слоем — кэшируются при повторных деплоях
COPY requirements-prod.txt .
RUN pip install --no-cache-dir \
    sentence-transformers==5.5.1 \
    faiss-cpu==1.13.2 \
    numpy==2.2.6 \
    fastapi==0.136.1 \
    "uvicorn[standard]==0.47.0" \
    deep-translator==1.11.4 \
    python-multipart==0.0.29

# скачиваем модель заранее, чтобы не тормозить при первом запросе
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')"

# исходники и данные
COPY app/ app/
COPY index/ index/
COPY data/processed/songs.json data/processed/songs.json

EXPOSE 8000

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
