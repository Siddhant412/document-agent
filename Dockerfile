FROM node:20-bookworm-slim AS ui-build

WORKDIR /ui

COPY frontend/package*.json ./
RUN npm ci
COPY frontend ./
RUN npm run build

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
COPY --from=ui-build /ui/dist ./document_agent/ui_dist

RUN python -m pip install --upgrade pip \
    && python -m pip install .

EXPOSE 8080

CMD ["document-agent", "serve"]
