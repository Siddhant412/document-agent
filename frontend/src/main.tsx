import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import {
  Activity,
  AlertCircle,
  CheckCircle2,
  Copy,
  Download,
  File,
  FileText,
  Image as ImageIcon,
  Loader2,
  MoreVertical,
  Play,
  RefreshCcw,
  Search,
  Settings,
  SplitSquareHorizontal,
  Trash2,
  UploadCloud,
  X
} from "lucide-react";
import {
  ApiError,
  ApiOptions,
  LibraryItem,
  SearchHit,
  apiBlob,
  deleteLibraryItem,
  getLibraryItem,
  getMarkdown,
  listLibrary,
  reindexSearch,
  reprocessLibraryItem,
  searchLibraryContent,
  uploadFiles
} from "./api";
import "./styles.css";
import { ObservabilityApp } from "./ObservabilityApp";

type PreviewState =
  | { kind: "idle" }
  | { kind: "loading" }
  | { kind: "pending"; message: string }
  | { kind: "error"; message: string }
  | { kind: "text"; text: string }
  | { kind: "image"; url: string }
  | { kind: "pdf"; url: string }
  | { kind: "download"; url: string; label: string };

type MarkdownState =
  | { kind: "idle" }
  | { kind: "loading" }
  | { kind: "pending"; message: string }
  | { kind: "error"; message: string }
  | { kind: "ready"; markdown: string };

const TYPE_FILTERS = ["all", "pdf", "docx", "doc", "txt", "png", "jpg", "jpeg", "heic"];
const STATUS_FILTERS = ["all", "queued", "running", "succeeded", "failed", "cancelled"];

function App() {
  const [items, setItems] = useState<LibraryItem[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [selectedItem, setSelectedItem] = useState<LibraryItem | null>(null);
  const [query, setQuery] = useState("");
  const [statusFilter, setStatusFilter] = useState("all");
  const [typeFilter, setTypeFilter] = useState("all");
  const [searchMode, setSearchMode] = useState<"hybrid" | "keyword" | "semantic">("hybrid");
  const [apiKey, setApiKey] = useLocalStorage("document-agent-api-key", "");
  const [showSettings, setShowSettings] = useState(false);
  const [uploadProgress, setUploadProgress] = useState<number | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [searchHits, setSearchHits] = useState<SearchHit[]>([]);
  const [searchTotal, setSearchTotal] = useState(0);
  const [searchLoading, setSearchLoading] = useState(false);
  const [preview, setPreview] = useState<PreviewState>({ kind: "idle" });
  const [markdown, setMarkdown] = useState<MarkdownState>({ kind: "idle" });
  const [markdownMode, setMarkdownMode] = useState<"preview" | "raw">("preview");
  const [dragActive, setDragActive] = useState(false);

  const apiOptions = useMemo<ApiOptions>(() => ({ apiKey: apiKey.trim() || undefined }), [apiKey]);
  const selected = useMemo(() => {
    const visibleItem = items.find((item) => item.library_item_id === selectedId);
    if (visibleItem) return visibleItem;
    if (selectedItem?.library_item_id === selectedId) return selectedItem;
    return null;
  }, [items, selectedId, selectedItem]);
  const previewObjectUrl = useRef<string | null>(null);

  const clearPreviewObjectUrl = useCallback(() => {
    if (previewObjectUrl.current) {
      URL.revokeObjectURL(previewObjectUrl.current);
      previewObjectUrl.current = null;
    }
  }, []);

  const refreshLibrary = useCallback(
    async (preferredId?: string | null) => {
      try {
        const response = await listLibrary(
          { q: query, status: statusFilter, detectedType: typeFilter },
          apiOptions
        );
        setItems(response.items);
        setNotice(null);
        const targetId = preferredId !== undefined ? preferredId : selectedId;
        if (targetId) {
          const visibleItem = response.items.find((item) => item.library_item_id === targetId);
          setSelectedId(targetId);
          if (visibleItem) {
            setSelectedItem(visibleItem);
          } else if (selectedItem?.library_item_id !== targetId) {
            setSelectedItem(await getLibraryItem(targetId, apiOptions));
          }
        } else {
          const nextItem = response.items[0] || null;
          setSelectedId(nextItem?.library_item_id || null);
          setSelectedItem(nextItem);
        }
      } catch (error) {
        setNotice(messageFromError(error));
      }
    },
    [apiOptions, query, selectedId, selectedItem?.library_item_id, statusFilter, typeFilter]
  );

  const selectLibraryItem = useCallback(
    async (libraryItemId: string | null) => {
      setSelectedId(libraryItemId);
      if (!libraryItemId) {
        setSelectedItem(null);
        return;
      }
      const visibleItem = items.find((item) => item.library_item_id === libraryItemId);
      if (visibleItem) {
        setSelectedItem(visibleItem);
        return;
      }
      try {
        setSelectedItem(await getLibraryItem(libraryItemId, apiOptions));
        setNotice(null);
      } catch (error) {
        setNotice(messageFromError(error));
      }
    },
    [apiOptions, items]
  );

  useEffect(() => {
    refreshLibrary().catch(() => undefined);
  }, [query, statusFilter, typeFilter]);

  useEffect(() => {
    const trimmed = query.trim();
    if (trimmed.length < 2) {
      setSearchHits([]);
      setSearchTotal(0);
      setSearchLoading(false);
      return;
    }
    let cancelled = false;
    setSearchLoading(true);
    const timer = window.setTimeout(() => {
      searchLibraryContent({ q: trimmed, detectedType: typeFilter, limit: 10, mode: searchMode }, apiOptions)
        .then((response) => {
          if (cancelled) return;
          setSearchHits(response.hits);
          setSearchTotal(response.total);
          setNotice(null);
        })
        .catch((error) => {
          if (cancelled) return;
          setSearchHits([]);
          setSearchTotal(0);
          setNotice(messageFromError(error));
        })
        .finally(() => {
          if (!cancelled) setSearchLoading(false);
        });
    }, 250);
    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
  }, [apiOptions, query, searchMode, typeFilter]);

  useEffect(() => {
    const timer = window.setInterval(() => {
      const hasActive = items.some((item) => item.status === "queued" || item.status === "running");
      if (hasActive) {
        refreshLibrary(selectedId).catch(() => undefined);
      }
    }, 3000);
    return () => window.clearInterval(timer);
  }, [items, refreshLibrary, selectedId]);

  useEffect(() => {
    if (!selected?.events_url || apiKey.trim()) {
      return;
    }
    const source = new EventSource(selected.events_url);
    source.onmessage = () => refreshLibrary(selected.library_item_id).catch(() => undefined);
    source.addEventListener("progress", () =>
      refreshLibrary(selected.library_item_id).catch(() => undefined)
    );
    source.addEventListener("succeeded", () =>
      refreshLibrary(selected.library_item_id).catch(() => undefined)
    );
    source.addEventListener("failed", () =>
      refreshLibrary(selected.library_item_id).catch(() => undefined)
    );
    return () => source.close();
  }, [apiKey, refreshLibrary, selected?.events_url, selected?.library_item_id]);

  useEffect(() => {
    clearPreviewObjectUrl();
    setPreview(selected ? { kind: "loading" } : { kind: "idle" });
    if (!selected) {
      return;
    }
    let cancelled = false;
    apiBlob(`/v1/library/${selected.library_item_id}/preview`, apiOptions)
      .then(async (blob) => {
        if (cancelled) return;
        if (isTextPreview(selected, blob)) {
          setPreview({ kind: "text", text: await blob.text() });
          return;
        }
        const url = URL.createObjectURL(blob);
        previewObjectUrl.current = url;
        if (isPdfPreview(selected, blob)) {
          setPreview({ kind: "pdf", url });
        } else if (blob.type.startsWith("image/")) {
          setPreview({ kind: "image", url });
        } else {
          setPreview({ kind: "download", url, label: selected.filename });
        }
      })
      .catch((error) => {
        if (cancelled) return;
        if (error instanceof ApiError && error.status === 409) {
          setPreview({ kind: "pending", message: error.detail });
        } else {
          setPreview({ kind: "error", message: messageFromError(error) });
        }
      });
    return () => {
      cancelled = true;
      clearPreviewObjectUrl();
    };
  }, [apiOptions, clearPreviewObjectUrl, selected?.library_item_id]);

  useEffect(() => {
    setMarkdown(selected ? { kind: "loading" } : { kind: "idle" });
    if (!selected) {
      return;
    }
    let cancelled = false;
    getMarkdown(selected.library_item_id, apiOptions)
      .then((response) => {
        if (cancelled) return;
        setMarkdown({ kind: "ready", markdown: response.markdown || "" });
      })
      .catch((error) => {
        if (cancelled) return;
        if (error instanceof ApiError && error.status === 409) {
          setMarkdown({ kind: "pending", message: "Markdown is not ready." });
        } else {
          setMarkdown({ kind: "error", message: messageFromError(error) });
        }
      });
    return () => {
      cancelled = true;
    };
  }, [apiOptions, selected?.library_item_id, selected?.status, selected?.updated_at]);

  const handleFiles = useCallback(
    async (fileList: FileList | File[]) => {
      const files = Array.from(fileList);
      if (!files.length) return;
      setUploadProgress(0);
      setNotice(null);
      try {
        const response = await uploadFiles(files, apiOptions, setUploadProgress);
        const preferredId =
          "child_jobs" in response
            ? response.child_jobs.find((job) => job.library_item_id)?.library_item_id || null
            : response.library_item_id || null;
        await refreshLibrary(preferredId);
      } catch (error) {
        setNotice(messageFromError(error));
      } finally {
        window.setTimeout(() => setUploadProgress(null), 700);
      }
    },
    [apiOptions, refreshLibrary]
  );

  const handleDelete = async () => {
    if (!selected) return;
    const confirmed = window.confirm(`Delete ${selected.filename}?`);
    if (!confirmed) return;
    try {
      await deleteLibraryItem(selected.library_item_id, apiOptions);
      await refreshLibrary(null);
    } catch (error) {
      setNotice(messageFromError(error));
    }
  };

  const handleReprocess = async () => {
    if (!selected) return;
    try {
      await reprocessLibraryItem(selected.library_item_id, apiOptions);
      await refreshLibrary(selected.library_item_id);
    } catch (error) {
      setNotice(messageFromError(error));
    }
  };

  const handleReindexSearch = async () => {
    try {
      const response = await reindexSearch(apiOptions);
      setNotice(`Search index updated: ${response.indexed} indexed, ${response.skipped} skipped.`);
      if (query.trim().length >= 2) {
        const data = await searchLibraryContent({ q: query.trim(), detectedType: typeFilter, limit: 10, mode: searchMode }, apiOptions);
        setSearchHits(data.hits);
        setSearchTotal(data.total);
      }
    } catch (error) {
      setNotice(messageFromError(error));
    }
  };

  const copyMarkdown = async () => {
    if (markdown.kind !== "ready") return;
    await navigator.clipboard.writeText(markdown.markdown);
  };

  const downloadMarkdown = () => {
    if (!selected || markdown.kind !== "ready") return;
    const blob = new Blob([markdown.markdown], { type: "text/markdown;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = markdownFilename(selected.filename);
    anchor.click();
    URL.revokeObjectURL(url);
  };

  return (
    <main
      className={`app-shell ${dragActive ? "drag-active" : ""}`}
      onDragOver={(event) => {
        event.preventDefault();
        setDragActive(true);
      }}
      onDragLeave={() => setDragActive(false)}
      onDrop={(event) => {
        event.preventDefault();
        setDragActive(false);
        handleFiles(event.dataTransfer.files).catch(() => undefined);
      }}
    >
      <aside className="library-pane">
        <div className="brand-row">
          <div className="brand-mark">
            <SplitSquareHorizontal size={18} />
          </div>
          <div>
            <h1>Document Agent</h1>
            <p>{items.length} files</p>
          </div>
          <button className="icon-button" onClick={() => setShowSettings(true)} title="Settings">
            <Settings size={18} />
          </button>
        </div>

        <label className="upload-zone">
          <UploadCloud size={22} />
          <span>Upload</span>
          <input
            type="file"
            multiple
            accept=".pdf,.jpg,.jpeg,.png,.heic,.txt,.doc,.docx"
            onChange={(event) => {
              if (event.target.files) {
                handleFiles(event.target.files).catch(() => undefined);
                event.currentTarget.value = "";
              }
            }}
          />
        </label>

        {uploadProgress !== null && (
          <div className="upload-progress">
            <div style={{ width: `${uploadProgress}%` }} />
          </div>
        )}

        <div className="search-box">
          <Search size={16} />
          <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search files and text" />
        </div>

        <div className="filter-row">
          <select value={statusFilter} onChange={(event) => setStatusFilter(event.target.value)}>
            {STATUS_FILTERS.map((item) => (
              <option key={item} value={item}>
                {label(item)}
              </option>
            ))}
          </select>
          <select value={typeFilter} onChange={(event) => setTypeFilter(event.target.value)}>
            {TYPE_FILTERS.map((item) => (
              <option key={item} value={item}>
                {item === "all" ? "All types" : item.toUpperCase()}
              </option>
            ))}
          </select>
        </div>

        {notice && (
          <div className="notice">
            <AlertCircle size={16} />
            <span>{notice}</span>
            <button className="plain-icon" onClick={() => setNotice(null)}>
              <X size={14} />
            </button>
          </div>
        )}

        <ContentSearchPanel
          query={query}
          hits={searchHits}
          total={searchTotal}
          loading={searchLoading}
          mode={searchMode}
          onModeChange={setSearchMode}
          onSelect={(hit) => selectLibraryItem(hit.library_item_id).catch(() => undefined)}
          onReindex={handleReindexSearch}
        />

        <div className="file-list">
          {items.map((item) => (
            <button
              key={item.library_item_id}
              className={`file-row ${selectedId === item.library_item_id ? "selected" : ""}`}
              onClick={() => selectLibraryItem(item.library_item_id).catch(() => undefined)}
            >
              <FileGlyph type={item.detected_type} />
              <span className="file-main">
                <span className="file-name">{item.filename}</span>
                <span className="file-meta">
                  {typeLabel(item.detected_type)} · {formatBytes(item.size_bytes)}
                </span>
              </span>
              <StatusPill item={item} />
            </button>
          ))}
          {!items.length && <div className="empty-list">No files</div>}
        </div>
      </aside>

      <section className="workspace">
        <header className="workspace-header">
          <div className="selected-title">
            <FileGlyph type={selected?.detected_type || null} />
            <div>
              <h2>{selected?.filename || "No file selected"}</h2>
              {selected && (
                <p>
                  {typeLabel(selected.detected_type)} · {formatBytes(selected.size_bytes)} ·{" "}
                  {label(selected.status)}
                </p>
              )}
            </div>
          </div>
          <div className="toolbar">
            <button className="tool-button" onClick={() => refreshLibrary(selectedId)}>
              <RefreshCcw size={16} />
              <span>Refresh</span>
            </button>
            <button className="tool-button" disabled={!selected} onClick={handleReprocess}>
              <Play size={16} />
              <span>Reprocess</span>
            </button>
            <button
              className="tool-button"
              onClick={() => window.open("/app/observability", "_blank")}
              title="Open Observability"
            >
              <Activity size={16} />
              <span>Observability</span>
            </button>
            <button className="icon-button danger" disabled={!selected} onClick={handleDelete} title="Delete">
              <Trash2 size={17} />
            </button>
          </div>
        </header>

        <div className="detail-grid">
          <section className="detail-pane">
            <PaneHeader title="Preview" item={selected} />
            <div className="pane-body">
              <PreviewView state={preview} item={selected} />
            </div>
          </section>

          <section className="detail-pane markdown-pane">
            <div className="pane-header">
              <div>
                <h3>Markdown</h3>
                <p>{markdown.kind === "ready" ? `${markdown.markdown.length.toLocaleString()} chars` : ""}</p>
              </div>
              <div className="segmented">
                <button
                  className={markdownMode === "preview" ? "active" : ""}
                  onClick={() => setMarkdownMode("preview")}
                >
                  Preview
                </button>
                <button
                  className={markdownMode === "raw" ? "active" : ""}
                  onClick={() => setMarkdownMode("raw")}
                >
                  Raw
                </button>
              </div>
              <div className="pane-actions">
                <button className="icon-button" disabled={markdown.kind !== "ready"} onClick={copyMarkdown} title="Copy">
                  <Copy size={16} />
                </button>
                <button
                  className="icon-button"
                  disabled={markdown.kind !== "ready"}
                  onClick={downloadMarkdown}
                  title="Download"
                >
                  <Download size={16} />
                </button>
              </div>
            </div>
            <div className="pane-body">
              <MarkdownView state={markdown} mode={markdownMode} />
            </div>
          </section>
        </div>
      </section>

      {showSettings && (
        <div className="modal-backdrop" onMouseDown={() => setShowSettings(false)}>
          <div className="modal" onMouseDown={(event) => event.stopPropagation()}>
            <div className="modal-header">
              <h3>Settings</h3>
              <button className="icon-button" onClick={() => setShowSettings(false)}>
                <X size={18} />
              </button>
            </div>
            <label className="field">
              <span>API key</span>
              <input
                value={apiKey}
                onChange={(event) => setApiKey(event.target.value)}
                type="password"
                autoComplete="off"
              />
            </label>
          </div>
        </div>
      )}
    </main>
  );
}

function ContentSearchPanel({
  query,
  hits,
  total,
  loading,
  mode,
  onModeChange,
  onSelect,
  onReindex,
}: {
  query: string;
  hits: SearchHit[];
  total: number;
  loading: boolean;
  mode: "hybrid" | "keyword" | "semantic";
  onModeChange: (mode: "hybrid" | "keyword" | "semantic") => void;
  onSelect: (hit: SearchHit) => void;
  onReindex: () => void;
}) {
  const active = query.trim().length >= 2;
  if (!active) {
    return (
      <div className="content-search compact">
        <div>
          <strong>Content search</strong>
          <span>Search converted Markdown across the library.</span>
        </div>
        <button className="mini-button" onClick={onReindex}>Index existing</button>
      </div>
    );
  }
  return (
    <div className="content-search">
      <div className="content-search-head">
        <div>
          <strong>Content matches</strong>
          <span>{loading ? "Searching..." : `${total} result${total === 1 ? "" : "s"}`}</span>
        </div>
        <select value={mode} onChange={(event) => onModeChange(event.target.value as typeof mode)}>
          <option value="hybrid">Hybrid</option>
          <option value="keyword">Keyword</option>
          <option value="semantic">Semantic</option>
        </select>
        <button className="mini-button" onClick={onReindex}>Reindex</button>
      </div>
      <div className="content-search-results">
        {hits.map((hit) => (
          <button key={`${hit.library_item_id}-${hit.asset_id}`} className="search-hit" onClick={() => onSelect(hit)}>
            <span className="search-hit-title">{hit.filename}</span>
            <Snippet text={hit.snippet} />
          </button>
        ))}
        {!loading && hits.length === 0 && <div className="search-empty">No content matches</div>}
      </div>
    </div>
  );
}

function Snippet({ text }: { text: string }) {
  const parts = text.split(/(<mark>|<\/mark>)/g);
  let marked = false;
  return (
    <span className="search-snippet">
      {parts.map((part, index) => {
        if (part === "<mark>") {
          marked = true;
          return null;
        }
        if (part === "</mark>") {
          marked = false;
          return null;
        }
        return marked ? <mark key={index}>{part}</mark> : <React.Fragment key={index}>{part}</React.Fragment>;
      })}
    </span>
  );
}

function PaneHeader({ title, item }: { title: string; item: LibraryItem | null }) {
  return (
    <div className="pane-header">
      <div>
        <h3>{title}</h3>
        <p>{item ? item.stage.replace(/_/g, " ") : ""}</p>
      </div>
      {item && <ProgressRing value={item.percent} status={item.status} />}
    </div>
  );
}

function PreviewView({ state, item }: { state: PreviewState; item: LibraryItem | null }) {
  if (!item) return <EmptyState title="Select a file" />;
  if (state.kind === "loading") return <BusyState label="Loading preview" />;
  if (state.kind === "pending") return <EmptyState title={state.message} active />;
  if (state.kind === "error") return <ErrorState message={state.message} />;
  if (state.kind === "text") return <pre className="text-preview">{state.text}</pre>;
  if (state.kind === "image") return <img className="image-preview" src={state.url} alt={item.filename} />;
  if (state.kind === "pdf") return <iframe className="pdf-preview" src={state.url} title={item.filename} />;
  if (state.kind === "download") {
    return (
      <a className="download-preview" href={state.url} download={state.label}>
        <Download size={20} />
        <span>{state.label}</span>
      </a>
    );
  }
  return <EmptyState title="No preview" />;
}

function MarkdownView({ state, mode }: { state: MarkdownState; mode: "preview" | "raw" }) {
  if (state.kind === "idle") return <EmptyState title="Select a file" />;
  if (state.kind === "loading") return <BusyState label="Loading Markdown" />;
  if (state.kind === "pending") return <EmptyState title={state.message} active />;
  if (state.kind === "error") return <ErrorState message={state.message} />;
  if (mode === "raw") return <pre className="raw-markdown">{state.markdown}</pre>;
  return (
    <article className="markdown-render">
      <ReactMarkdown remarkPlugins={[remarkGfm]}>{state.markdown}</ReactMarkdown>
    </article>
  );
}

function StatusPill({ item }: { item: LibraryItem }) {
  const active = item.status === "queued" || item.status === "running";
  return (
    <span className={`status-pill ${item.status}`}>
      {active ? <Loader2 size={12} /> : item.status === "succeeded" ? <CheckCircle2 size={12} /> : null}
      {active ? `${item.percent}%` : label(item.status)}
    </span>
  );
}

function ProgressRing({ value, status }: { value: number; status: string }) {
  const clamped = Math.max(0, Math.min(100, value));
  return (
    <div className={`progress-ring ${status}`} style={{ "--progress": `${clamped}%` } as React.CSSProperties}>
      <span>{clamped}</span>
    </div>
  );
}

function FileGlyph({ type }: { type: string | null }) {
  const normalized = (type || "").toLowerCase();
  if (["jpg", "jpeg", "png", "heic"].includes(normalized)) {
    return (
      <span className="file-glyph image">
        <ImageIcon size={18} />
      </span>
    );
  }
  if (["txt", "doc", "docx", "pdf"].includes(normalized)) {
    return (
      <span className="file-glyph doc">
        <FileText size={18} />
      </span>
    );
  }
  return (
    <span className="file-glyph">
      <File size={18} />
    </span>
  );
}

function BusyState({ label: busyLabel }: { label: string }) {
  return (
    <div className="state-view">
      <Loader2 size={22} className="spin" />
      <span>{busyLabel}</span>
    </div>
  );
}

function EmptyState({ title, active = false }: { title: string; active?: boolean }) {
  return (
    <div className={`state-view ${active ? "active" : ""}`}>
      <MoreVertical size={22} />
      <span>{title}</span>
    </div>
  );
}

function ErrorState({ message }: { message: string }) {
  return (
    <div className="state-view error">
      <AlertCircle size={22} />
      <span>{message}</span>
    </div>
  );
}

function useLocalStorage(key: string, initialValue: string) {
  const [value, setValue] = useState(() => window.localStorage.getItem(key) || initialValue);
  const update = (next: string) => {
    setValue(next);
    window.localStorage.setItem(key, next);
  };
  return [value, update] as const;
}

function isTextPreview(item: LibraryItem, blob: Blob) {
  return item.detected_type === "txt" || blob.type.startsWith("text/");
}

function isPdfPreview(item: LibraryItem, blob: Blob) {
  return item.detected_type === "pdf" || blob.type === "application/pdf";
}

function markdownFilename(filename: string) {
  return `${filename.replace(/\.[^.]+$/, "") || "document"}.md`;
}

function typeLabel(type: string | null) {
  return type ? type.toUpperCase() : "UNKNOWN";
}

function label(value: string) {
  if (value === "all") return "All";
  return value
    .split("_")
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

function formatBytes(bytes: number) {
  if (!Number.isFinite(bytes)) return "";
  const units = ["B", "KB", "MB", "GB"];
  let value = bytes;
  let index = 0;
  while (value >= 1024 && index < units.length - 1) {
    value /= 1024;
    index += 1;
  }
  return `${value.toFixed(value >= 10 || index === 0 ? 0 : 1)} ${units[index]}`;
}

function messageFromError(error: unknown) {
  if (error instanceof ApiError) return error.detail;
  if (error instanceof Error) return error.message;
  return "Request failed.";
}

const isObsPage = window.location.pathname.startsWith("/app/observability");

createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    {isObsPage ? <ObservabilityApp /> : <App />}
  </React.StrictMode>
);
