export type LibraryStatus = "queued" | "running" | "succeeded" | "failed" | "cancelled" | "deleted";

export interface LibraryItem {
  library_item_id: string;
  current_job_id: string | null;
  batch_id: string | null;
  input_index: number | null;
  filename: string;
  content_type: string | null;
  detected_type: string | null;
  sha256: string;
  size_bytes: number;
  status: LibraryStatus;
  stage: string;
  percent: number;
  preview_status: string;
  created_at: string;
  updated_at: string;
  uploaded_at: string;
  processed_at: string | null;
  error_code: string | null;
  error_message: string | null;
  original_url: string;
  preview_url: string;
  markdown_url: string;
  events_url: string | null;
  has_markdown: boolean;
  has_preview: boolean;
  metadata: Record<string, unknown>;
}

export interface LibraryListResponse {
  items: LibraryItem[];
  limit: number;
  offset: number;
}

export interface JobCreatedResponse {
  library_item_id?: string | null;
  job_id: string;
  status: LibraryStatus;
  library_url?: string | null;
}

export interface BatchCreatedResponse {
  batch_id: string;
  status: string;
  child_jobs: Array<{
    library_item_id?: string | null;
    job_id: string;
    input_index: number;
    filename: string;
    status: LibraryStatus;
  }>;
}

export interface MarkdownResponse {
  library_item_id: string;
  job_id: string | null;
  status: LibraryStatus;
  markdown_url: string | null;
  asset_id: string | null;
  markdown: string | null;
  metadata: Record<string, unknown>;
}

export interface SearchHit {
  library_item_id: string;
  job_id: string;
  asset_id: string;
  filename: string;
  detected_type: string | null;
  score: number;
  keyword_score: number;
  semantic_score: number;
  chunk_index: number | null;
  snippet: string;
  markdown_url: string;
  preview_url: string;
  processed_at: string | null;
}

export interface SearchResponse {
  query: string;
  hits: SearchHit[];
  limit: number;
  offset: number;
  total: number;
}

export interface ApiOptions {
  apiKey?: string;
  apiKeyHeader?: string;
}

export class ApiError extends Error {
  status: number;
  detail: string;

  constructor(status: number, detail: string) {
    super(detail);
    this.status = status;
    this.detail = detail;
  }
}

const DEFAULT_KEY_HEADER = "X-API-Key";

export function authHeaders(options: ApiOptions): HeadersInit {
  if (!options.apiKey) {
    return {};
  }
  return { [options.apiKeyHeader || DEFAULT_KEY_HEADER]: options.apiKey };
}

export async function apiJson<T>(path: string, init: RequestInit, options: ApiOptions): Promise<T> {
  const headers = new Headers(init.headers);
  const auth = authHeaders(options);
  for (const [key, value] of Object.entries(auth)) {
    headers.set(key, value);
  }
  const response = await fetch(path, { ...init, headers });
  if (!response.ok) {
    throw await errorFromResponse(response);
  }
  return (await response.json()) as T;
}

export async function apiBlob(path: string, options: ApiOptions): Promise<Blob> {
  const response = await fetch(path, { headers: authHeaders(options) });
  if (!response.ok) {
    throw await errorFromResponse(response);
  }
  return response.blob();
}

export async function listLibrary(
  params: { q?: string; status?: string; detectedType?: string },
  options: ApiOptions
): Promise<LibraryListResponse> {
  const query = new URLSearchParams({ limit: "100", offset: "0" });
  if (params.q) query.set("q", params.q);
  if (params.status && params.status !== "all") query.set("status", params.status);
  if (params.detectedType && params.detectedType !== "all") {
    query.set("detected_type", params.detectedType);
  }
  return apiJson<LibraryListResponse>(`/v1/library?${query}`, { method: "GET" }, options);
}

export async function getLibraryItem(id: string, options: ApiOptions): Promise<LibraryItem> {
  return apiJson<LibraryItem>(`/v1/library/${id}`, { method: "GET" }, options);
}

export async function getMarkdown(id: string, options: ApiOptions): Promise<MarkdownResponse> {
  return apiJson<MarkdownResponse>(
    `/v1/library/${id}/markdown?include_markdown=true`,
    { method: "GET" },
    options
  );
}

export async function searchLibraryContent(
  params: { q: string; detectedType?: string; limit?: number; mode?: "keyword" | "semantic" | "hybrid" },
  options: ApiOptions
): Promise<SearchResponse> {
  const query = new URLSearchParams({
    q: params.q,
    limit: String(params.limit ?? 20),
    offset: "0",
  });
  if (params.detectedType && params.detectedType !== "all") {
    query.set("detected_type", params.detectedType);
  }
  if (params.mode) query.set("mode", params.mode);
  return apiJson<SearchResponse>(`/v1/search?${query}`, { method: "GET" }, options);
}

export async function reindexSearch(options: ApiOptions): Promise<{ indexed: number; skipped: number; limit: number }> {
  return apiJson<{ indexed: number; skipped: number; limit: number }>(
    "/v1/search/reindex",
    { method: "POST" },
    options
  );
}

export async function deleteLibraryItem(id: string, options: ApiOptions): Promise<void> {
  await apiJson(`/v1/library/${id}`, { method: "DELETE" }, options);
}

export async function reprocessLibraryItem(id: string, options: ApiOptions): Promise<void> {
  await apiJson(`/v1/library/${id}/reprocess`, { method: "POST" }, options);
}

export function uploadFiles(
  files: File[],
  options: ApiOptions,
  onProgress: (percent: number) => void
): Promise<JobCreatedResponse | BatchCreatedResponse> {
  const form = new FormData();
  const path = files.length === 1 ? "/v1/jobs" : "/v1/batches";
  if (files.length === 1) {
    form.append("file", files[0], files[0].name);
  } else {
    for (const file of files) {
      form.append("files", file, file.name);
    }
  }
  return xhrJson(path, form, options, onProgress);
}

function xhrJson<T>(
  path: string,
  body: FormData,
  options: ApiOptions,
  onProgress: (percent: number) => void
): Promise<T> {
  return new Promise((resolve, reject) => {
    const request = new XMLHttpRequest();
    request.open("POST", path);
    request.setRequestHeader("Idempotency-Key", crypto.randomUUID());
    for (const [key, value] of Object.entries(authHeaders(options))) {
      request.setRequestHeader(key, value);
    }
    request.upload.onprogress = (event) => {
      if (event.lengthComputable) {
        onProgress(Math.round((event.loaded / event.total) * 100));
      }
    };
    request.onload = () => {
      if (request.status >= 200 && request.status < 300) {
        onProgress(100);
        resolve(JSON.parse(request.responseText) as T);
        return;
      }
      reject(new ApiError(request.status, responseDetail(request.responseText)));
    };
    request.onerror = () => reject(new ApiError(0, "Network error while uploading files."));
    request.send(body);
  });
}

// ---------------------------------------------------------------------------
// Observability
// ---------------------------------------------------------------------------

export interface ObsStatsResponse {
  total_jobs: number;
  jobs_by_status: Record<string, number>;
  success_rate_pct: number | null;
  avg_duration_seconds: number | null;
  p95_duration_seconds: number | null;
  total_batches: number;
  active_jobs: number;
  throughput_by_hour: Array<{ hour: string; succeeded: number; failed: number }>;
  jobs_by_type: Array<{ detected_type: string | null; count: number }>;
  health: Record<string, string>;
}

export interface ObsEventRow {
  id: number;
  library_item_id: string | null;
  batch_id: string | null;
  job_id: string | null;
  event_type: string;
  stage: string | null;
  percent: number | null;
  message: string | null;
  payload: Record<string, unknown>;
  created_at: string;
}

export interface ObsEventsResponse {
  events: ObsEventRow[];
  has_more: boolean;
  next_before_id: number | null;
}

export interface ObsErrorItem {
  job_id: string;
  library_item_id: string | null;
  filename: string;
  detected_type: string | null;
  error_code: string;
  error_message: string | null;
  failed_at: string | null;
  attempt_count: number;
}

export interface ObsErrorsResponse {
  errors: ObsErrorItem[];
  error_code_counts: Array<{ error_code: string; count: number }>;
  total_failed: number;
}

export interface ObsLogRecord {
  seq: number;
  ts: string;
  level: string;
  logger: string;
  message: string;
}

export interface ObsLogsResponse {
  logs: ObsLogRecord[];
  max_seq: number;
  buffer_capacity: number;
  buffer_used: number;
}

export async function fetchObsStats(options: ApiOptions): Promise<ObsStatsResponse> {
  return apiJson<ObsStatsResponse>("/v1/observability/stats", { method: "GET" }, options);
}

export async function fetchObsEvents(
  params: { limit?: number; before_id?: number; since_id?: number; event_type?: string; q?: string },
  options: ApiOptions
): Promise<ObsEventsResponse> {
  const query = new URLSearchParams();
  if (params.limit != null) query.set("limit", String(params.limit));
  if (params.before_id != null) query.set("before_id", String(params.before_id));
  if (params.since_id != null) query.set("since_id", String(params.since_id));
  if (params.event_type) query.set("event_type", params.event_type);
  if (params.q) query.set("q", params.q);
  return apiJson<ObsEventsResponse>(`/v1/observability/events?${query}`, { method: "GET" }, options);
}

export async function fetchObsErrors(
  params: { limit?: number; error_code?: string; q?: string },
  options: ApiOptions
): Promise<ObsErrorsResponse> {
  const query = new URLSearchParams();
  if (params.limit != null) query.set("limit", String(params.limit));
  if (params.error_code) query.set("error_code", params.error_code);
  if (params.q) query.set("q", params.q);
  return apiJson<ObsErrorsResponse>(`/v1/observability/errors?${query}`, { method: "GET" }, options);
}

export async function fetchObsLogs(
  params: { limit?: number; level?: string; q?: string; since_seq?: number },
  options: ApiOptions
): Promise<ObsLogsResponse> {
  const query = new URLSearchParams();
  if (params.limit != null) query.set("limit", String(params.limit));
  if (params.level) query.set("level", params.level);
  if (params.q) query.set("q", params.q);
  if (params.since_seq != null) query.set("since_seq", String(params.since_seq));
  return apiJson<ObsLogsResponse>(`/v1/observability/logs?${query}`, { method: "GET" }, options);
}

async function errorFromResponse(response: Response): Promise<ApiError> {
  const text = await response.text();
  return new ApiError(response.status, responseDetail(text));
}

function responseDetail(text: string): string {
  if (!text) {
    return "Request failed.";
  }
  try {
    const payload = JSON.parse(text);
    if (typeof payload.detail === "string") {
      return payload.detail;
    }
    return JSON.stringify(payload.detail || payload);
  } catch {
    return text;
  }
}
