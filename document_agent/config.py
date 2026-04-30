from __future__ import annotations

from functools import lru_cache
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_env: str = "local"
    public_base_url: str = "http://localhost:8080"
    api_base_url: str = "http://localhost:8080"
    log_level: str = "INFO"

    api_host: str = "0.0.0.0"
    api_port: int = 8080
    api_key: Optional[str] = None
    api_key_header: str = "X-API-Key"
    max_upload_bytes: int = 100 * 1024 * 1024
    max_batch_files: int = 100
    max_batch_bytes: int = 1024 * 1024 * 1024

    database_url: str = "postgresql://document_agent:document_agent@localhost:5432/document_agent"
    db_pool_min_size: int = 1
    db_pool_max_size: int = 10

    minio_endpoint: str = "localhost:9000"
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin"
    minio_secure: bool = False
    minio_region: Optional[str] = None
    minio_bucket: str = "document-agent"
    minio_auto_create_bucket: bool = True

    worker_id: Optional[str] = None
    worker_poll_interval_seconds: float = 2.0
    worker_lease_seconds: int = 300
    worker_lease_heartbeat_seconds: int = 30
    worker_metrics_host: str = "0.0.0.0"
    worker_metrics_port: int = 8081
    job_max_attempts: int = 3
    job_timeout_seconds: int = 3600
    batch_timeout_seconds: int = 14400
    staging_ttl_seconds: int = 86400

    ocr_server_url: Optional[str] = None
    ocr_api_key: Optional[str] = None
    ocr_model: str = "allenai/olmOCR-2-7B-1025-FP8"
    ocr_page_max_tokens: int = 8000
    ocr_target_longest_image_dim: int = 1288
    ocr_max_concurrent_requests: int = 2

    max_pdf_pages: int = 500
    pandoc_bin: str = "pandoc"
    libreoffice_bin: str = "soffice"
    pdf_use_vendor_extractor: bool = True

    upload_spool_chunk_bytes: int = Field(default=1024 * 1024)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
