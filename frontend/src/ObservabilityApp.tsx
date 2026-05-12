import React, {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import {
  Activity,
  AlertCircle,
  ArrowLeft,
  CheckCircle2,
  ChevronDown,
  Clock,
  Database,
  FileText,
  RefreshCcw,
  Search,
  Server,
  Wifi,
  X,
  Zap,
} from "lucide-react";
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import {
  ApiOptions,
  ObsErrorsResponse,
  ObsEventRow,
  ObsEventsResponse,
  ObsLogRecord,
  ObsLogsResponse,
  ObsStatsResponse,
  ObsTimeRange,
  fetchObsErrors,
  fetchObsEvents,
  fetchObsLogs,
  fetchObsStats,
} from "./api";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

type Tab = "overview" | "events" | "logs" | "metrics";
type RefreshInterval = 0 | 5 | 15 | 30;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const REFRESH_INTERVALS: RefreshInterval[] = [0, 5, 15, 30];
const REFRESH_LABELS: Record<RefreshInterval, string> = {
  0: "Off",
  5: "5 s",
  15: "15 s",
  30: "30 s",
};

const TIME_RANGES: ObsTimeRange[] = ["1h", "6h", "24h", "7d", "30d", "all"];
const TIME_RANGE_LABELS: Record<ObsTimeRange, string> = {
  "1h": "Last 1 hour",
  "6h": "Last 6 hours",
  "24h": "Last 24 hours",
  "7d": "Last 7 days",
  "30d": "Last 30 days",
  all: "All time",
};

const PIE_COLORS = ["#2563eb", "#16a34a", "#dc2626", "#d97706", "#7c3aed", "#0891b2", "#db2777"];

const EVENT_TYPE_COLORS: Record<string, string> = {
  queued: "ev-blue",
  started: "ev-blue",
  progress: "ev-sky",
  succeeded: "ev-green",
  failed: "ev-red",
  cancelled: "ev-red",
  asset_uploaded: "ev-purple",
};

const LOG_LEVEL_COLORS: Record<string, string> = {
  DEBUG: "lvl-slate",
  INFO: "lvl-sky",
  WARNING: "lvl-orange",
  ERROR: "lvl-red",
  CRITICAL: "lvl-red",
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function useLocalStorage(key: string, initial: string) {
  const [value, setValue] = useState(() => window.localStorage.getItem(key) || initial);
  const update = (next: string) => {
    setValue(next);
    window.localStorage.setItem(key, next);
  };
  return [value, update] as const;
}

function fmtTimeBucket(iso: string, range: ObsTimeRange) {
  try {
    const d = new Date(iso);
    if (range === "1h" || range === "6h" || range === "24h") {
      return `${d.getHours().toString().padStart(2, "0")}:00`;
    }
    return d.toLocaleString(undefined, {
      month: "short",
      day: "numeric",
      hour: "2-digit",
    });
  } catch {
    return iso;
  }
}


function fmtTs(iso: string) {
  try {
    return new Date(iso).toLocaleString(undefined, {
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    });
  } catch {
    return iso;
  }
}

function fmtDuration(sec: number | null) {
  if (sec == null) return "—";
  if (sec < 1) return `${Math.round(sec * 1000)} ms`;
  if (sec < 60) return `${sec.toFixed(1)} s`;
  return `${(sec / 60).toFixed(1)} min`;
}

function normalizeTimeRange(value: string): ObsTimeRange {
  return TIME_RANGES.includes(value as ObsTimeRange) ? (value as ObsTimeRange) : "24h";
}

// ---------------------------------------------------------------------------
// Top-level component
// ---------------------------------------------------------------------------

export function ObservabilityApp() {
  const [tab, setTab] = useState<Tab>("overview");
  const [refreshInterval, setRefreshInterval] = useState<RefreshInterval>(15);
  const [apiKey] = useLocalStorage("document-agent-api-key", "");
  const [storedTimeRange, setStoredTimeRange] = useLocalStorage(
    "document-agent-observability-time-range",
    "24h"
  );
  const timeRange = normalizeTimeRange(storedTimeRange);

  const apiOptions = useMemo<ApiOptions>(() => ({ apiKey: apiKey.trim() || undefined }), [apiKey]);

  const cycleRefresh = () => {
    const idx = REFRESH_INTERVALS.indexOf(refreshInterval);
    setRefreshInterval(REFRESH_INTERVALS[(idx + 1) % REFRESH_INTERVALS.length]);
  };

  return (
    <div className="obs-shell">
      <header className="obs-header">
        <a href="/app" className="obs-back">
          <ArrowLeft size={16} />
          <span>Library</span>
        </a>
        <div className="obs-brand">
          <Activity size={18} />
          <span>Observability</span>
        </div>
        <nav className="obs-tabs">
          {(["overview", "events", "logs", "metrics"] as Tab[]).map((t) => (
            <button
              key={t}
              className={`obs-tab ${tab === t ? "active" : ""}`}
              onClick={() => setTab(t)}
            >
              {t.charAt(0).toUpperCase() + t.slice(1)}
            </button>
          ))}
        </nav>
        <button
          className={`obs-refresh-btn ${refreshInterval > 0 ? "active" : ""}`}
          onClick={cycleRefresh}
          title="Auto-refresh interval"
        >
          <RefreshCcw size={14} />
          <span>{REFRESH_LABELS[refreshInterval]}</span>
          <ChevronDown size={12} />
        </button>
        <label className="obs-time-range">
          <Clock size={14} />
          <select
            value={timeRange}
            onChange={(e) => setStoredTimeRange(normalizeTimeRange(e.target.value))}
            title="Observability time range"
          >
            {TIME_RANGES.map((range) => (
              <option key={range} value={range}>
                {TIME_RANGE_LABELS[range]}
              </option>
            ))}
          </select>
        </label>
      </header>
      <div className="obs-body">
        {tab === "overview" && (
          <OverviewSection
            apiOptions={apiOptions}
            refreshInterval={refreshInterval}
            timeRange={timeRange}
          />
        )}
        {tab === "events" && (
          <EventsSection
            apiOptions={apiOptions}
            refreshInterval={refreshInterval}
            timeRange={timeRange}
          />
        )}
        {tab === "logs" && (
          <LogsSection
            apiOptions={apiOptions}
            refreshInterval={refreshInterval}
            timeRange={timeRange}
          />
        )}
        {tab === "metrics" && (
          <MetricsSection
            apiOptions={apiOptions}
            refreshInterval={refreshInterval}
            timeRange={timeRange}
          />
        )}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Overview
// ---------------------------------------------------------------------------

function OverviewSection({
  apiOptions,
  refreshInterval,
  timeRange,
}: {
  apiOptions: ApiOptions;
  refreshInterval: RefreshInterval;
  timeRange: ObsTimeRange;
}) {
  const [stats, setStats] = useState<ObsStatsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    try {
      const data = await fetchObsStats(apiOptions, timeRange);
      setStats(data);
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load stats");
    } finally {
      setLoading(false);
    }
  }, [apiOptions, timeRange]);

  useEffect(() => {
    load();
  }, [load]);

  useEffect(() => {
    if (refreshInterval === 0) return;
    const timer = window.setInterval(load, refreshInterval * 1000);
    return () => window.clearInterval(timer);
  }, [load, refreshInterval]);

  if (loading && !stats) return <LoadingPane />;
  if (error) return <ErrorPane message={error} onRetry={load} />;
  if (!stats) return null;

  const failed = stats.jobs_by_status["failed"] ?? 0;
  const throughputData = stats.throughput_by_hour.map((r) => ({
    ...r,
    hour: fmtTimeBucket(r.hour, timeRange),
  }));
  const typeData = stats.jobs_by_type.map((r) => ({
    name: r.detected_type ? r.detected_type.toUpperCase() : "UNKNOWN",
    value: r.count,
  }));
  const rangeLabel = TIME_RANGE_LABELS[timeRange].toLowerCase();

  return (
    <div className="obs-overview">
      <div className="obs-cards">
        <StatCard
          label="Total Jobs"
          value={stats.total_jobs.toLocaleString()}
          icon={<FileText size={18} />}
        />
        <StatCard
          label="Success Rate"
          value={stats.success_rate_pct != null ? `${stats.success_rate_pct}%` : "—"}
          icon={<CheckCircle2 size={18} />}
          variant="success"
        />
        <StatCard
          label="Active Now"
          value={stats.active_jobs.toLocaleString()}
          icon={<Zap size={18} />}
          variant="active"
        />
        <StatCard
          label="Failed"
          value={failed.toLocaleString()}
          icon={<AlertCircle size={18} />}
          variant={failed > 0 ? "error" : undefined}
        />
        <StatCard
          label="Avg Duration"
          value={fmtDuration(stats.avg_duration_seconds)}
          icon={<Clock size={18} />}
        />
      </div>

      <div className="obs-chart-row">
        <div className="obs-chart-card">
          <div className="obs-chart-title">Job throughput — {rangeLabel}</div>
          {throughputData.length === 0 ? (
            <EmptyChartNote>No jobs in {rangeLabel}</EmptyChartNote>
          ) : (
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={throughputData} margin={{ top: 8, right: 16, bottom: 0, left: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                <XAxis dataKey="hour" tick={{ fontSize: 11, fill: "#64748b" }} />
                <YAxis tick={{ fontSize: 11, fill: "#64748b" }} allowDecimals={false} />
                <Tooltip
                  contentStyle={{ fontSize: 12, borderRadius: 8, border: "1px solid #e5e7eb" }}
                />
                <Legend wrapperStyle={{ fontSize: 12 }} />
                <Bar dataKey="succeeded" name="Succeeded" stackId="a" fill="#16a34a" radius={[0, 0, 0, 0]} />
                <Bar dataKey="failed" name="Failed" stackId="a" fill="#dc2626" radius={[3, 3, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          )}
        </div>

        <div className="obs-chart-card">
          <div className="obs-chart-title">Jobs by file type — {rangeLabel}</div>
          {typeData.length === 0 ? (
            <EmptyChartNote>No jobs yet</EmptyChartNote>
          ) : (
            <ResponsiveContainer width="100%" height={220}>
              <PieChart>
                <Pie
                  data={typeData}
                  cx="50%"
                  cy="50%"
                  innerRadius={55}
                  outerRadius={85}
                  paddingAngle={2}
                  dataKey="value"
                >
                  {typeData.map((_, i) => (
                    <Cell key={i} fill={PIE_COLORS[i % PIE_COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip
                  contentStyle={{ fontSize: 12, borderRadius: 8, border: "1px solid #e5e7eb" }}
                />
                <Legend wrapperStyle={{ fontSize: 12 }} />
              </PieChart>
            </ResponsiveContainer>
          )}
        </div>
      </div>

      <div className="obs-health-row">
        <span className="obs-health-label">Health</span>
        <HealthBadge label="API" status={stats.health["api"]} icon={<Server size={13} />} />
        <HealthBadge label="Database" status={stats.health["db"]} icon={<Database size={13} />} />
        <HealthBadge label="Worker" status={stats.health["worker"]} icon={<Wifi size={13} />} />
      </div>
    </div>
  );
}

function StatCard({
  label,
  value,
  icon,
  variant,
}: {
  label: string;
  value: string;
  icon: React.ReactNode;
  variant?: "success" | "error" | "active";
}) {
  return (
    <div className={`obs-card ${variant ?? ""}`}>
      <div className="obs-card-icon">{icon}</div>
      <div className="obs-card-body">
        <div className="obs-card-label">{label}</div>
        <div className="obs-card-value">{value}</div>
      </div>
    </div>
  );
}

function HealthBadge({
  label,
  status,
  icon,
}: {
  label: string;
  status: string;
  icon: React.ReactNode;
}) {
  const cls = status === "ok" ? "health-ok" : status === "idle" ? "health-idle" : "health-err";
  return (
    <span className={`obs-health-badge ${cls}`}>
      {icon}
      <span>{label}</span>
      <span className="health-status">{status}</span>
    </span>
  );
}

function EmptyChartNote({ children }: { children: React.ReactNode }) {
  return (
    <div className="obs-chart-empty">{children}</div>
  );
}

// ---------------------------------------------------------------------------
// Events
// ---------------------------------------------------------------------------

function EventsSection({
  apiOptions,
  refreshInterval,
  timeRange,
}: {
  apiOptions: ApiOptions;
  refreshInterval: RefreshInterval;
  timeRange: ObsTimeRange;
}) {
  const [events, setEvents] = useState<ObsEventRow[]>([]);
  const [hasMore, setHasMore] = useState(false);
  const [nextBeforeId, setNextBeforeId] = useState<number | null>(null);
  const [maxId, setMaxId] = useState<number>(0);
  const [loading, setLoading] = useState(true);
  const [loadingMore, setLoadingMore] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [eventType, setEventType] = useState("");
  const [q, setQ] = useState("");
  const [newCount, setNewCount] = useState(0);
  const tableRef = useRef<HTMLDivElement>(null);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const resp = await fetchObsEvents(
        { limit: 50, event_type: eventType || undefined, q: q || undefined, timeRange },
        apiOptions
      );
      setEvents(resp.events);
      setHasMore(resp.has_more);
      setNextBeforeId(resp.next_before_id ?? null);
      setMaxId(resp.events[0]?.id ?? 0);
      setNewCount(0);
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load events");
    } finally {
      setLoading(false);
    }
  }, [apiOptions, eventType, q, timeRange]);

  useEffect(() => {
    load();
  }, [load]);

  // Poll for new events
  useEffect(() => {
    if (refreshInterval === 0) return;
    const timer = window.setInterval(async () => {
      if (maxId === 0) return;
      try {
        const resp = await fetchObsEvents(
          { since_id: maxId, event_type: eventType || undefined, q: q || undefined, timeRange },
          apiOptions
        );
        if (resp.events.length > 0) {
          setEvents((prev) => {
            const newEvents = resp.events.filter((e) => !prev.some((p) => p.id === e.id));
            if (newEvents.length === 0) return prev;
            return [...newEvents, ...prev];
          });
          setMaxId(resp.events[resp.events.length - 1].id);
          setNewCount((c) => c + resp.events.length);
        }
      } catch {
        // silent poll failure
      }
    }, refreshInterval * 1000);
    return () => window.clearInterval(timer);
  }, [apiOptions, eventType, maxId, q, refreshInterval, timeRange]);

  const loadMore = async () => {
    if (!nextBeforeId) return;
    setLoadingMore(true);
    try {
      const resp = await fetchObsEvents(
        {
          limit: 50,
          before_id: nextBeforeId,
          event_type: eventType || undefined,
          q: q || undefined,
          timeRange,
        },
        apiOptions
      );
      setEvents((prev) => [...prev, ...resp.events]);
      setHasMore(resp.has_more);
      setNextBeforeId(resp.next_before_id ?? null);
    } catch {
      // ignore
    } finally {
      setLoadingMore(false);
    }
  };

  const scrollToTop = () => {
    tableRef.current?.scrollTo({ top: 0, behavior: "smooth" });
    setNewCount(0);
  };

  return (
    <div className="obs-section">
      <div className="obs-filter-row">
        <div className="obs-search-box">
          <Search size={14} />
          <input
            value={q}
            onChange={(e) => setQ(e.target.value)}
            placeholder="Search messages…"
          />
          {q && (
            <button className="obs-clear-btn" onClick={() => setQ("")}>
              <X size={12} />
            </button>
          )}
        </div>
        <select
          className="obs-select"
          value={eventType}
          onChange={(e) => setEventType(e.target.value)}
        >
          <option value="">All event types</option>
          {["queued", "started", "progress", "succeeded", "failed", "cancelled", "asset_uploaded"].map(
            (t) => (
              <option key={t} value={t}>
                {t}
              </option>
            )
          )}
        </select>
        <span className="obs-range-note">{TIME_RANGE_LABELS[timeRange]}</span>
      </div>

      {newCount > 0 && (
        <button className="obs-new-banner" onClick={scrollToTop}>
          {newCount} new event{newCount !== 1 ? "s" : ""} — click to scroll to top
        </button>
      )}

      {error && <InlineError message={error} onRetry={load} />}
      {loading && !events.length ? (
        <LoadingPane />
      ) : (
        <>
          <div className="obs-table-wrap" ref={tableRef}>
            <table className="obs-table">
              <thead>
                <tr>
                  <th>ID</th>
                  <th>Type</th>
                  <th>Stage</th>
                  <th>Message</th>
                  <th>Time</th>
                </tr>
              </thead>
              <tbody>
                {events.length === 0 && (
                  <tr>
                    <td colSpan={5} className="obs-table-empty">
                      No events found
                    </td>
                  </tr>
                )}
                {events.map((ev) => (
                  <EventRow key={ev.id} event={ev} />
                ))}
              </tbody>
            </table>
          </div>
          {hasMore && (
            <div className="obs-load-more">
              <button className="obs-load-more-btn" onClick={loadMore} disabled={loadingMore}>
                {loadingMore ? "Loading…" : "Load older events"}
              </button>
            </div>
          )}
          <div className="obs-table-footer">
            Showing {events.length} event{events.length !== 1 ? "s" : ""}
          </div>
        </>
      )}
    </div>
  );
}

function EventRow({ event }: { event: ObsEventRow }) {
  const colorClass = EVENT_TYPE_COLORS[event.event_type] ?? "ev-default";
  return (
    <tr className="obs-table-row">
      <td className="obs-cell-id">{event.id}</td>
      <td>
        <span className={`obs-ev-pill ${colorClass}`}>{event.event_type}</span>
      </td>
      <td className="obs-cell-dim">{event.stage ?? "—"}</td>
      <td className="obs-cell-msg">{event.message ?? "—"}</td>
      <td className="obs-cell-time">{fmtTs(event.created_at)}</td>
    </tr>
  );
}

// ---------------------------------------------------------------------------
// Logs
// ---------------------------------------------------------------------------

function LogsSection({
  apiOptions,
  refreshInterval,
  timeRange,
}: {
  apiOptions: ApiOptions;
  refreshInterval: RefreshInterval;
  timeRange: ObsTimeRange;
}) {
  const [logs, setLogs] = useState<ObsLogRecord[]>([]);
  const [maxSeq, setMaxSeq] = useState(0);
  const [bufStats, setBufStats] = useState({ capacity: 0, used: 0 });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [level, setLevel] = useState("");
  const [q, setQ] = useState("");
  const viewerRef = useRef<HTMLDivElement>(null);
  const atBottomRef = useRef(true);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const resp = await fetchObsLogs(
        { limit: 200, level: level || undefined, q: q || undefined, timeRange },
        apiOptions
      );
      setLogs(resp.logs);
      setMaxSeq(resp.max_seq);
      setBufStats({ capacity: resp.buffer_capacity, used: resp.buffer_used });
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load logs");
    } finally {
      setLoading(false);
    }
  }, [apiOptions, level, q, timeRange]);

  useEffect(() => {
    load();
  }, [load]);

  // Incremental poll
  useEffect(() => {
    if (refreshInterval === 0) return;
    const timer = window.setInterval(async () => {
      try {
        const resp = await fetchObsLogs(
          {
            limit: 200,
            level: level || undefined,
            q: q || undefined,
            since_seq: maxSeq,
            timeRange,
          },
          apiOptions
        );
        if (resp.logs.length > 0) {
          setLogs((prev) => {
            const merged = [...resp.logs, ...prev];
            return merged.slice(0, 500);
          });
          setMaxSeq(resp.max_seq);
          setBufStats({ capacity: resp.buffer_capacity, used: resp.buffer_used });
          if (atBottomRef.current && viewerRef.current) {
            requestAnimationFrame(() => {
              viewerRef.current?.scrollTo({ top: viewerRef.current.scrollHeight });
            });
          }
        }
      } catch {
        // silent
      }
    }, refreshInterval * 1000);
    return () => window.clearInterval(timer);
  }, [apiOptions, level, maxSeq, q, refreshInterval, timeRange]);

  const handleScroll = () => {
    const el = viewerRef.current;
    if (!el) return;
    atBottomRef.current = el.scrollHeight - el.scrollTop - el.clientHeight < 120;
  };

  return (
    <div className="obs-section">
      <div className="obs-filter-row">
        <select
          className="obs-select"
          value={level}
          onChange={(e) => setLevel(e.target.value)}
        >
          <option value="">All levels</option>
          {["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"].map((l) => (
            <option key={l} value={l}>
              {l}
            </option>
          ))}
        </select>
        <div className="obs-search-box">
          <Search size={14} />
          <input
            value={q}
            onChange={(e) => setQ(e.target.value)}
            placeholder="Search logs…"
          />
          {q && (
            <button className="obs-clear-btn" onClick={() => setQ("")}>
              <X size={12} />
            </button>
          )}
        </div>
        <span className="obs-range-note">{TIME_RANGE_LABELS[timeRange]}</span>
      </div>

      <div className="obs-buf-stats">
        Buffer: {bufStats.used} / {bufStats.capacity} records &nbsp;·&nbsp; Showing {logs.length} in{" "}
        {TIME_RANGE_LABELS[timeRange].toLowerCase()}
        {refreshInterval > 0 && <span className="obs-live-dot" title="Live polling active" />}
      </div>

      {error && <InlineError message={error} onRetry={load} />}
      {loading && !logs.length ? (
        <LoadingPane />
      ) : (
        <div className="obs-log-viewer" ref={viewerRef} onScroll={handleScroll}>
          {logs.length === 0 && <div className="obs-log-empty">No log records</div>}
          {logs.map((rec) => (
            <LogLine key={rec.seq} record={rec} />
          ))}
        </div>
      )}
    </div>
  );
}

function LogLine({ record }: { record: ObsLogRecord }) {
  const lvlClass = LOG_LEVEL_COLORS[record.level] ?? "lvl-slate";
  return (
    <div className="obs-log-line">
      <span className="obs-log-ts">{record.ts}</span>
      <span className={`obs-log-level ${lvlClass}`}>{record.level}</span>
      <span className="obs-log-logger">{record.logger}</span>
      <span className="obs-log-msg">{record.message}</span>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Metrics
// ---------------------------------------------------------------------------

function MetricsSection({
  apiOptions,
  refreshInterval,
  timeRange,
}: {
  apiOptions: ApiOptions;
  refreshInterval: RefreshInterval;
  timeRange: ObsTimeRange;
}) {
  const [stats, setStats] = useState<ObsStatsResponse | null>(null);
  const [errors, setErrors] = useState<ObsErrorsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    try {
      const [s, e] = await Promise.all([
        fetchObsStats(apiOptions, timeRange),
        fetchObsErrors({ limit: 100, timeRange }, apiOptions),
      ]);
      setStats(s);
      setErrors(e);
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load metrics");
    } finally {
      setLoading(false);
    }
  }, [apiOptions, timeRange]);

  useEffect(() => {
    load();
  }, [load]);

  useEffect(() => {
    if (refreshInterval === 0) return;
    const timer = window.setInterval(load, refreshInterval * 1000);
    return () => window.clearInterval(timer);
  }, [load, refreshInterval]);

  if (loading && !stats) return <LoadingPane />;
  if (error) return <ErrorPane message={error} onRetry={load} />;
  if (!stats || !errors) return null;

  const throughputData = stats.throughput_by_hour.map((r) => ({
    ...r,
    hour: fmtTimeBucket(r.hour, timeRange),
  }));

  const statusData = Object.entries(stats.jobs_by_status).map(([status, count]) => ({
    name: status,
    value: count,
  }));

  const errorRateData = stats.throughput_by_hour.map((r) => ({
    hour: fmtTimeBucket(r.hour, timeRange),
    total: (r.succeeded ?? 0) + (r.failed ?? 0),
    failed: r.failed ?? 0,
    rate:
      (r.succeeded ?? 0) + (r.failed ?? 0) > 0
        ? Math.round(((r.failed ?? 0) / ((r.succeeded ?? 0) + (r.failed ?? 0))) * 100)
        : 0,
  }));

  const typeData = stats.jobs_by_type.map((r) => ({
    name: r.detected_type ? r.detected_type.toUpperCase() : "UNKNOWN",
    count: r.count,
    avg: stats.avg_duration_seconds ?? 0,
  }));
  const rangeLabel = TIME_RANGE_LABELS[timeRange].toLowerCase();

  return (
    <div className="obs-chart-grid">
        <div className="obs-chart-card">
          <div className="obs-chart-title">Throughput — {rangeLabel}</div>
          {throughputData.length === 0 ? (
            <EmptyChartNote>No data</EmptyChartNote>
          ) : (
            <ResponsiveContainer width="100%" height={220}>
              <AreaChart data={throughputData} margin={{ top: 8, right: 16, bottom: 0, left: 0 }}>
                <defs>
                  <linearGradient id="gradSucceeded" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#16a34a" stopOpacity={0.3} />
                    <stop offset="95%" stopColor="#16a34a" stopOpacity={0} />
                  </linearGradient>
                  <linearGradient id="gradFailed" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#dc2626" stopOpacity={0.3} />
                    <stop offset="95%" stopColor="#dc2626" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                <XAxis dataKey="hour" tick={{ fontSize: 11, fill: "#64748b" }} />
                <YAxis tick={{ fontSize: 11, fill: "#64748b" }} allowDecimals={false} />
                <Tooltip
                  contentStyle={{ fontSize: 12, borderRadius: 8, border: "1px solid #e5e7eb" }}
                />
                <Legend wrapperStyle={{ fontSize: 12 }} />
                <Area
                  type="monotone"
                  dataKey="succeeded"
                  name="Succeeded"
                  stroke="#16a34a"
                  fill="url(#gradSucceeded)"
                  strokeWidth={2}
                />
                <Area
                  type="monotone"
                  dataKey="failed"
                  name="Failed"
                  stroke="#dc2626"
                  fill="url(#gradFailed)"
                  strokeWidth={2}
                />
              </AreaChart>
            </ResponsiveContainer>
          )}
        </div>

        <div className="obs-chart-card">
          <div className="obs-chart-title">Jobs by status — {rangeLabel}</div>
          {statusData.length === 0 ? (
            <EmptyChartNote>No data</EmptyChartNote>
          ) : (
            <ResponsiveContainer width="100%" height={220}>
              <PieChart>
                <Pie
                  data={statusData}
                  cx="50%"
                  cy="50%"
                  innerRadius={55}
                  outerRadius={85}
                  paddingAngle={2}
                  dataKey="value"
                >
                  {statusData.map((entry, i) => {
                    const fill =
                      entry.name === "succeeded"
                        ? "#16a34a"
                        : entry.name === "failed" || entry.name === "cancelled"
                        ? "#dc2626"
                        : entry.name === "running" || entry.name === "queued"
                        ? "#2563eb"
                        : PIE_COLORS[i % PIE_COLORS.length];
                    return <Cell key={entry.name} fill={fill} />;
                  })}
                </Pie>
                <Tooltip
                  contentStyle={{ fontSize: 12, borderRadius: 8, border: "1px solid #e5e7eb" }}
                />
                <Legend wrapperStyle={{ fontSize: 12 }} />
              </PieChart>
            </ResponsiveContainer>
          )}
        </div>

        <div className="obs-chart-card">
          <div className="obs-chart-title">Error rate (%) — {rangeLabel}</div>
          {errorRateData.length === 0 ? (
            <EmptyChartNote>No data</EmptyChartNote>
          ) : (
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={errorRateData} margin={{ top: 8, right: 16, bottom: 0, left: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                <XAxis dataKey="hour" tick={{ fontSize: 11, fill: "#64748b" }} />
                <YAxis
                  tick={{ fontSize: 11, fill: "#64748b" }}
                  allowDecimals={false}
                  domain={[0, 100]}
                  unit="%"
                />
                <Tooltip
                  contentStyle={{ fontSize: 12, borderRadius: 8, border: "1px solid #e5e7eb" }}
                  formatter={(v: number) => [`${v}%`, "Error rate"]}
                />
                <Bar dataKey="rate" name="Error rate %" fill="#dc2626" radius={[3, 3, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          )}
        </div>

        <div className="obs-chart-card">
          <div className="obs-chart-title">Top error codes — {rangeLabel}</div>
          {errors.error_code_counts.length === 0 ? (
            <EmptyChartNote>No errors recorded</EmptyChartNote>
          ) : (
            <ResponsiveContainer width="100%" height={220}>
              <BarChart
                data={errors.error_code_counts.slice(0, 8)}
                layout="vertical"
                margin={{ top: 8, right: 24, bottom: 0, left: 8 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                <XAxis type="number" tick={{ fontSize: 11, fill: "#64748b" }} allowDecimals={false} />
                <YAxis
                  type="category"
                  dataKey="error_code"
                  tick={{ fontSize: 10, fill: "#64748b" }}
                  width={120}
                />
                <Tooltip
                  contentStyle={{ fontSize: 12, borderRadius: 8, border: "1px solid #e5e7eb" }}
                />
                <Bar dataKey="count" name="Count" fill="#7c3aed" radius={[0, 3, 3, 0]} />
              </BarChart>
            </ResponsiveContainer>
          )}
        </div>

        <div className="obs-chart-card obs-duration-card">
          <div className="obs-chart-title">Duration stats — {rangeLabel}</div>
          <div className="obs-duration-stats">
            <div className="obs-dur-row">
              <span className="obs-dur-label">Avg duration</span>
              <span className="obs-dur-value">{fmtDuration(stats.avg_duration_seconds)}</span>
            </div>
            <div className="obs-dur-row">
              <span className="obs-dur-label">P95 duration</span>
              <span className="obs-dur-value">{fmtDuration(stats.p95_duration_seconds)}</span>
            </div>
            <div className="obs-dur-row">
              <span className="obs-dur-label">Total batches</span>
              <span className="obs-dur-value">{stats.total_batches.toLocaleString()}</span>
            </div>
            <div className="obs-dur-row">
              <span className="obs-dur-label">Active leases</span>
              <span className="obs-dur-value">{stats.active_jobs}</span>
            </div>
          </div>
          {typeData.length > 0 && (
            <>
              <div className="obs-chart-title" style={{ marginTop: 16 }}>
                Jobs by file type
              </div>
              <ResponsiveContainer width="100%" height={160}>
                <BarChart
                  data={typeData}
                  layout="vertical"
                  margin={{ top: 4, right: 24, bottom: 0, left: 8 }}
                >
                  <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                  <XAxis type="number" tick={{ fontSize: 11, fill: "#64748b" }} allowDecimals={false} />
                  <YAxis
                    type="category"
                    dataKey="name"
                    tick={{ fontSize: 10, fill: "#64748b" }}
                    width={60}
                  />
                  <Tooltip
                    contentStyle={{ fontSize: 12, borderRadius: 8, border: "1px solid #e5e7eb" }}
                  />
                  <Bar dataKey="count" name="Jobs" fill="#2563eb" radius={[0, 3, 3, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </>
          )}
        </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Shared UI helpers
// ---------------------------------------------------------------------------

function LoadingPane() {
  return (
    <div className="obs-loading">
      <div className="obs-skeleton" />
      <div className="obs-skeleton" style={{ width: "70%" }} />
      <div className="obs-skeleton" style={{ width: "85%" }} />
    </div>
  );
}

function ErrorPane({ message, onRetry }: { message: string; onRetry: () => void }) {
  return (
    <div className="obs-error-pane">
      <AlertCircle size={22} />
      <span>{message}</span>
      <button className="obs-retry-btn" onClick={onRetry}>
        Retry
      </button>
    </div>
  );
}

function InlineError({ message, onRetry }: { message: string; onRetry: () => void }) {
  return (
    <div className="obs-inline-error">
      <AlertCircle size={14} />
      <span>{message}</span>
      <button onClick={onRetry}>Retry</button>
    </div>
  );
}
