FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/vendor

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
      curl \
      libgl1 \
      libglib2.0-0 \
      libheif1 \
      libreoffice \
      pandoc \
      poppler-utils \
      tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md ./
COPY document_agent ./document_agent
COPY migrations ./migrations
COPY vendor ./vendor

RUN python -m pip install --upgrade pip \
    && python -m pip install .

EXPOSE 8080

CMD ["document-agent", "serve"]
