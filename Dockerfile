FROM --platform=linux/amd64 python:3.12.10-slim@sha256:97983fa8cc88343512862c62307159a82261c3528dc025f79e5a3f7af43e50b4 AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1
WORKDIR /build
COPY requirements-lock.txt ./
RUN pip install --prefix=/install --require-hashes -r requirements-lock.txt

FROM --platform=linux/amd64 python:3.12.10-slim@sha256:97983fa8cc88343512862c62307159a82261c3528dc025f79e5a3f7af43e50b4 AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    JUDICIAL_AI_CORPUS_PATH=/data/analysis.csv \
    JUDICIAL_AI_BERT_CACHE_DIR=/models
RUN groupadd --gid 10001 app && useradd --uid 10001 --gid app --no-create-home app
WORKDIR /app
COPY --from=builder /install /usr/local
COPY . .
RUN mkdir -p /data /models && chown -R app:app /models
USER 10001:10001
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/v1/health', timeout=3).read()" || exit 1
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
