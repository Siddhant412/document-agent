CREATE EXTENSION IF NOT EXISTS pgcrypto;

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

CREATE UNIQUE INDEX IF NOT EXISTS document_jobs_single_idempotency_uq
  ON document_jobs(idempotency_key)
  WHERE idempotency_key IS NOT NULL AND batch_id IS NULL;

CREATE INDEX IF NOT EXISTS document_jobs_queue_idx
  ON document_jobs(status, created_at);

CREATE INDEX IF NOT EXISTS document_jobs_batch_idx
  ON document_jobs(batch_id, input_index, created_at);

CREATE INDEX IF NOT EXISTS document_jobs_lease_idx
  ON document_jobs(status, lease_expires_at)
  WHERE status = 'running';

CREATE TABLE IF NOT EXISTS document_assets (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
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

CREATE UNIQUE INDEX IF NOT EXISTS document_assets_object_uq
  ON document_assets(bucket, object_key);

CREATE INDEX IF NOT EXISTS document_assets_job_idx
  ON document_assets(job_id, role);

CREATE INDEX IF NOT EXISTS document_assets_batch_idx
  ON document_assets(batch_id, role);

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
  batch_id UUID REFERENCES document_batches(id) ON DELETE SET NULL,
  job_id UUID REFERENCES document_jobs(id) ON DELETE SET NULL,
  event_type TEXT NOT NULL,
  stage TEXT,
  percent INTEGER,
  message TEXT,
  payload_json JSONB NOT NULL DEFAULT '{}'::jsonb,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS job_events_job_idx
  ON job_events(job_id, id);

CREATE INDEX IF NOT EXISTS job_events_batch_idx
  ON job_events(batch_id, id);

CREATE INDEX IF NOT EXISTS job_events_created_idx
  ON job_events(created_at);
