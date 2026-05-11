CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS document_batches (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  status TEXT NOT NULL DEFAULT 'queued',
  batch_name TEXT,
  idempotency_key TEXT UNIQUE,
  total_files INTEGER NOT NULL DEFAULT 0,
  succeeded_count INTEGER NOT NULL DEFAULT 0,
  failed_count INTEGER NOT NULL DEFAULT 0,
  cancelled_count INTEGER NOT NULL DEFAULT 0,
  result_archive_asset_id UUID,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  started_at TIMESTAMPTZ,
  finished_at TIMESTAMPTZ,
  cancelled_at TIMESTAMPTZ,
  metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS document_jobs (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  batch_id UUID REFERENCES document_batches(id) ON DELETE SET NULL,
  library_item_id UUID,
  input_index INTEGER,
  idempotency_key TEXT,
  status TEXT NOT NULL DEFAULT 'queued',
  filename TEXT NOT NULL,
  content_type TEXT,
  detected_type TEXT,
  sha256 TEXT NOT NULL,
  size_bytes BIGINT NOT NULL,
  source_bucket TEXT NOT NULL,
  source_object_key TEXT NOT NULL,
  source_asset_id UUID,
  source_deleted_at TIMESTAMPTZ,
  stage TEXT NOT NULL DEFAULT 'queued',
  percent INTEGER NOT NULL DEFAULT 0,
  attempt_count INTEGER NOT NULL DEFAULT 0,
  max_attempts INTEGER NOT NULL DEFAULT 3,
  lease_owner TEXT,
  lease_expires_at TIMESTAMPTZ,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  started_at TIMESTAMPTZ,
  finished_at TIMESTAMPTZ,
  cancelled_at TIMESTAMPTZ,
  result_markdown_asset_id UUID,
  error_code TEXT,
  error_message TEXT,
  error_details_json JSONB NOT NULL DEFAULT '{}'::jsonb,
  metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
  CHECK (percent >= 0 AND percent <= 100)
);

ALTER TABLE document_jobs
  ADD COLUMN IF NOT EXISTS library_item_id UUID;

ALTER TABLE document_jobs
  ADD COLUMN IF NOT EXISTS source_asset_id UUID;

CREATE UNIQUE INDEX IF NOT EXISTS document_jobs_single_idempotency_uq
  ON document_jobs(idempotency_key)
  WHERE idempotency_key IS NOT NULL AND batch_id IS NULL;

CREATE INDEX IF NOT EXISTS document_jobs_queue_idx
  ON document_jobs(status, created_at);

CREATE INDEX IF NOT EXISTS document_jobs_batch_idx
  ON document_jobs(batch_id, input_index, created_at);

CREATE INDEX IF NOT EXISTS document_jobs_library_item_idx
  ON document_jobs(library_item_id, created_at);

CREATE INDEX IF NOT EXISTS document_jobs_lease_idx
  ON document_jobs(status, lease_expires_at)
  WHERE status = 'running';

CREATE TABLE IF NOT EXISTS document_assets (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  library_item_id UUID,
  batch_id UUID REFERENCES document_batches(id) ON DELETE SET NULL,
  job_id UUID REFERENCES document_jobs(id) ON DELETE SET NULL,
  role TEXT NOT NULL,
  bucket TEXT NOT NULL,
  object_key TEXT NOT NULL,
  mime_type TEXT NOT NULL,
  size_bytes BIGINT NOT NULL,
  sha256 TEXT NOT NULL,
  public_url TEXT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb
);

ALTER TABLE document_assets
  ADD COLUMN IF NOT EXISTS library_item_id UUID;

CREATE UNIQUE INDEX IF NOT EXISTS document_assets_object_uq
  ON document_assets(bucket, object_key);

CREATE INDEX IF NOT EXISTS document_assets_job_idx
  ON document_assets(job_id, role);

CREATE INDEX IF NOT EXISTS document_assets_batch_idx
  ON document_assets(batch_id, role);

CREATE INDEX IF NOT EXISTS document_assets_library_item_idx
  ON document_assets(library_item_id, role);

CREATE TABLE IF NOT EXISTS document_library_items (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  batch_id UUID REFERENCES document_batches(id) ON DELETE SET NULL,
  current_job_id UUID REFERENCES document_jobs(id) ON DELETE SET NULL,
  input_index INTEGER,
  original_filename TEXT NOT NULL,
  display_filename TEXT NOT NULL,
  content_type TEXT,
  detected_type TEXT,
  sha256 TEXT NOT NULL,
  size_bytes BIGINT NOT NULL,
  status TEXT NOT NULL DEFAULT 'queued',
  stage TEXT NOT NULL DEFAULT 'queued',
  percent INTEGER NOT NULL DEFAULT 0,
  preview_status TEXT NOT NULL DEFAULT 'pending',
  original_asset_id UUID,
  preview_asset_id UUID,
  thumbnail_asset_id UUID,
  latest_markdown_asset_id UUID,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  uploaded_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  processed_at TIMESTAMPTZ,
  deleted_at TIMESTAMPTZ,
  error_code TEXT,
  error_message TEXT,
  error_details_json JSONB NOT NULL DEFAULT '{}'::jsonb,
  metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
  CHECK (percent >= 0 AND percent <= 100)
);

ALTER TABLE document_library_items
  ADD COLUMN IF NOT EXISTS preview_status TEXT NOT NULL DEFAULT 'pending';

CREATE INDEX IF NOT EXISTS document_library_items_status_idx
  ON document_library_items(status, created_at DESC)
  WHERE deleted_at IS NULL;

CREATE INDEX IF NOT EXISTS document_library_items_batch_idx
  ON document_library_items(batch_id, input_index, created_at);

DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_constraint WHERE conname = 'document_batches_archive_fk'
  ) THEN
    ALTER TABLE document_batches
      ADD CONSTRAINT document_batches_archive_fk
      FOREIGN KEY (result_archive_asset_id)
      REFERENCES document_assets(id)
      DEFERRABLE INITIALLY DEFERRED;
  END IF;
END $$;

DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_constraint WHERE conname = 'document_jobs_markdown_fk'
  ) THEN
    ALTER TABLE document_jobs
      ADD CONSTRAINT document_jobs_markdown_fk
      FOREIGN KEY (result_markdown_asset_id)
      REFERENCES document_assets(id)
      DEFERRABLE INITIALLY DEFERRED;
  END IF;
END $$;

CREATE TABLE IF NOT EXISTS job_events (
  id BIGSERIAL PRIMARY KEY,
  library_item_id UUID,
  batch_id UUID REFERENCES document_batches(id) ON DELETE SET NULL,
  job_id UUID REFERENCES document_jobs(id) ON DELETE SET NULL,
  event_type TEXT NOT NULL,
  stage TEXT,
  percent INTEGER,
  message TEXT,
  payload_json JSONB NOT NULL DEFAULT '{}'::jsonb,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

ALTER TABLE job_events
  ADD COLUMN IF NOT EXISTS library_item_id UUID;

CREATE INDEX IF NOT EXISTS job_events_job_idx
  ON job_events(job_id, id);

CREATE INDEX IF NOT EXISTS job_events_batch_idx
  ON job_events(batch_id, id);

CREATE INDEX IF NOT EXISTS job_events_library_item_idx
  ON job_events(library_item_id, id);

CREATE INDEX IF NOT EXISTS job_events_created_idx
  ON job_events(created_at);

CREATE TABLE IF NOT EXISTS document_search_entries (
  library_item_id UUID PRIMARY KEY,
  job_id UUID REFERENCES document_jobs(id) ON DELETE CASCADE,
  asset_id UUID REFERENCES document_assets(id) ON DELETE CASCADE,
  filename TEXT NOT NULL,
  detected_type TEXT,
  content TEXT NOT NULL,
  search_vector TSVECTOR GENERATED ALWAYS AS (
    setweight(to_tsvector('simple', coalesce(filename, '')), 'A') ||
    setweight(to_tsvector('simple', coalesce(content, '')), 'B')
  ) STORED,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

ALTER TABLE document_search_entries
  ADD COLUMN IF NOT EXISTS detected_type TEXT;

CREATE INDEX IF NOT EXISTS document_search_entries_vector_idx
  ON document_search_entries USING GIN(search_vector);

CREATE INDEX IF NOT EXISTS document_search_entries_type_idx
  ON document_search_entries(detected_type);

CREATE INDEX IF NOT EXISTS document_search_entries_job_idx
  ON document_search_entries(job_id);

CREATE TABLE IF NOT EXISTS document_search_chunks (
  id BIGSERIAL PRIMARY KEY,
  library_item_id UUID NOT NULL,
  job_id UUID REFERENCES document_jobs(id) ON DELETE CASCADE,
  asset_id UUID REFERENCES document_assets(id) ON DELETE CASCADE,
  chunk_index INTEGER NOT NULL,
  filename TEXT NOT NULL,
  detected_type TEXT,
  content TEXT NOT NULL,
  embedding vector(768),
  search_vector TSVECTOR GENERATED ALWAYS AS (
    setweight(to_tsvector('simple', coalesce(filename, '')), 'A') ||
    setweight(to_tsvector('simple', coalesce(content, '')), 'B')
  ) STORED,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE(library_item_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS document_search_chunks_vector_idx
  ON document_search_chunks USING hnsw (embedding vector_cosine_ops)
  WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS document_search_chunks_fts_idx
  ON document_search_chunks USING GIN(search_vector);

CREATE INDEX IF NOT EXISTS document_search_chunks_library_idx
  ON document_search_chunks(library_item_id, chunk_index);

CREATE INDEX IF NOT EXISTS document_search_chunks_type_idx
  ON document_search_chunks(detected_type);
