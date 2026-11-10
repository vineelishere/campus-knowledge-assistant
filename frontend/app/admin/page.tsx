"use client";

import { useEffect, useState } from "react";

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

type Summary = {
  total: number;
  answered: number;
  abstained: number;
  stale_only: number;
  unsupported: number;
  errors: number;
  p50_ms: number;
  p95_ms: number;
  total_cost_usd: number;
};
type Failed = { ts: string; question: string; status: string; reason: string | null; top_score: number | null };
type Daily = { day: string; queries: number; not_answered: number; avg_ms: number | null; cost_usd: number | null };
type Metrics = { summary: Summary; failed_searches: Failed[]; daily: Daily[] };

const card: React.CSSProperties = {
  background: "#fff",
  border: "1px solid #e3e6ea",
  borderRadius: 12,
  padding: 16,
};
const th: React.CSSProperties = { textAlign: "left", padding: "6px 8px", borderBottom: "1px solid #e3e6ea", fontSize: 13, color: "#5b6470" };
const td: React.CSSProperties = { padding: "6px 8px", borderBottom: "1px solid #f0f2f4", fontSize: 14, verticalAlign: "top" };

function Stat({ label, value, hint }: { label: string; value: string; hint?: string }) {
  return (
    <div style={card}>
      <div style={{ fontSize: 13, color: "#5b6470" }}>{label}</div>
      <div style={{ fontSize: 26, fontWeight: 600, marginTop: 4 }}>{value}</div>
      {hint && <div style={{ fontSize: 12, color: "#8a93a0", marginTop: 2 }}>{hint}</div>}
    </div>
  );
}

export default function Admin() {
  const [data, setData] = useState<Metrics | null>(null);
  const [error, setError] = useState("");
  const [days, setDays] = useState(7);

  useEffect(() => {
    const token = sessionStorage.getItem("token");
    if (!token) {
      setError("Please log in as an administrator first.");
      return;
    }
    setError("");
    fetch(`${API}/admin/metrics?days=${days}`, { headers: { Authorization: `Bearer ${token}` } })
      .then(async (r) => {
        if (r.status === 403) throw new Error("This page is for administrators only.");
        if (r.status === 401) throw new Error("Session expired. Please log in again.");
        if (!r.ok) throw new Error("Could not load metrics.");
        return r.json();
      })
      .then(setData)
      .catch((e) => setError(String(e.message ?? e)));
  }, [days]);

  const s = data?.summary;
  const total = Number(s?.total ?? 0);
  const pct = (n: number) => (total ? `${Math.round((100 * n) / total)}%` : "-");
  const maxDay = Math.max(1, ...(data?.daily ?? []).map((d) => Number(d.queries)));

  return (
    <main style={{ maxWidth: 980, margin: "0 auto", padding: 16 }}>
      <header style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "8px 0 16px" }}>
        <h1 style={{ fontSize: 20, margin: 0 }}>Assistant dashboard</h1>
        <span style={{ fontSize: 14 }}>
          <a href="/" style={{ color: "#1f4fd8" }}>back to chat</a>
          {" · "}
          <select value={days} onChange={(e) => setDays(Number(e.target.value))} style={{ fontSize: 14 }}>
            <option value={1}>last 24 hours</option>
            <option value={7}>last 7 days</option>
            <option value={30}>last 30 days</option>
          </select>
        </span>
      </header>

      {error && <p style={{ color: "#b3261e" }}>{error}</p>}
      {!error && !data && <p>Loading…</p>}

      {s && (
        <>
          <section style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(190px, 1fr))", gap: 12 }}>
            <Stat label="Questions" value={String(total)} />
            <Stat label="Answered" value={pct(Number(s.answered))} hint={`${s.answered} of ${total}`} />
            <Stat label="Failed searches (I do not know)" value={String(Number(s.abstained))} hint={pct(Number(s.abstained))} />
            <Stat label="Only outdated document found" value={String(Number(s.stale_only))} />
            <Stat label="Unsupported answers (refused)" value={String(Number(s.unsupported))} />
            <Stat label="Errors (should be 0)" value={String(Number(s.errors))} />
            <Stat label="Median response time" value={`${(Number(s.p50_ms) / 1000).toFixed(1)} s`} />
            <Stat label="95th percentile time" value={`${(Number(s.p95_ms) / 1000).toFixed(1)} s`} />
            <Stat label="Total cost" value={`$${Number(s.total_cost_usd).toFixed(4)}`} hint="0 on free tiers unless prices are set in .env" />
          </section>

          <h2 style={{ fontSize: 16, marginTop: 24 }}>Per day</h2>
          <div style={card}>
            <table style={{ width: "100%", borderCollapse: "collapse" }}>
              <thead>
                <tr>
                  <th style={th}>Day</th>
                  <th style={th}>Questions</th>
                  <th style={th}>Not answered</th>
                  <th style={th}>Avg time</th>
                  <th style={th}>Cost</th>
                </tr>
              </thead>
              <tbody>
                {data!.daily.map((d) => (
                  <tr key={d.day}>
                    <td style={td}>{String(d.day).slice(0, 10)}</td>
                    <td style={td}>
                      <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                        <div style={{ height: 8, borderRadius: 4, background: "#1f4fd8", width: `${(100 * Number(d.queries)) / maxDay}%`, minWidth: 4, maxWidth: 160 }} />
                        {d.queries}
                      </div>
                    </td>
                    <td style={td}>{d.not_answered}</td>
                    <td style={td}>{d.avg_ms == null ? "-" : `${(Number(d.avg_ms) / 1000).toFixed(1)} s`}</td>
                    <td style={td}>{d.cost_usd == null ? "-" : `$${Number(d.cost_usd).toFixed(4)}`}</td>
                  </tr>
                ))}
                {data!.daily.length === 0 && (
                  <tr>
                    <td style={td} colSpan={5}>No questions in this period yet.</td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>

          <h2 style={{ fontSize: 16, marginTop: 24 }}>Questions that were not answered</h2>
          <div style={{ ...card, overflowX: "auto" }}>
            <table style={{ width: "100%", borderCollapse: "collapse" }}>
              <thead>
                <tr>
                  <th style={th}>When</th>
                  <th style={th}>Question</th>
                  <th style={th}>Status</th>
                  <th style={th}>Reason</th>
                  <th style={th}>Top score</th>
                </tr>
              </thead>
              <tbody>
                {data!.failed_searches.map((f, i) => (
                  <tr key={i}>
                    <td style={td}>{new Date(f.ts).toLocaleString()}</td>
                    <td style={td}>{f.question}</td>
                    <td style={td}>{f.status}</td>
                    <td style={td}>{f.reason ?? "-"}</td>
                    <td style={td}>{f.top_score == null ? "-" : Number(f.top_score).toFixed(1)}</td>
                  </tr>
                ))}
                {data!.failed_searches.length === 0 && (
                  <tr>
                    <td style={td} colSpan={5}>None in this period.</td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </>
      )}
    </main>
  );
}
