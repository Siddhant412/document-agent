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
