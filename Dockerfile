FROM python:3.12.10-slim@sha256:fd95fa221297a88e1cf49c55ec1828edd7c5a428187e67b5d1805692d11588db AS builder

ARG TARGETARCH
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1
WORKDIR /build
COPY requirements-lock.txt requirements-arm64-lock.txt ./
RUN case "$TARGETARCH" in \
      amd64) requirements=requirements-lock.txt ;; \
      arm64) requirements=requirements-arm64-lock.txt ;; \
      *) echo "Unsupported architecture: $TARGETARCH" >&2; exit 1 ;; \
    esac && \
    pip install --prefix=/install --only-binary=:all: --require-hashes -r "$requirements"

FROM python:3.12.10-slim@sha256:fd95fa221297a88e1cf49c55ec1828edd7c5a428187e67b5d1805692d11588db AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    JUDICIAL_AI_CORPUS_PATH=/app/fixtures/synthetic_analysis.csv \
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
