"use client";

import { FormEvent, useEffect, useRef, useState } from "react";

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

type Citation = {
  n: number;
  title: string;
  year: number;
  page: number | null;
  department: string | null;
  source_url: string | null;
  excerpt: string;
};
type Reply = { answer: string; status: string; citations: Citation[]; warnings: string[] };
type Turn = { question: string; reply?: Reply; error?: string };

const card: React.CSSProperties = {
  background: "#fff",
  border: "1px solid #e3e6ea",
  borderRadius: 12,
  padding: 16,
};

export default function Home() {
  const [token, setToken] = useState<string | null>(null);
  const [role, setRole] = useState("");
  const [loginErr, setLoginErr] = useState("");
  const [turns, setTurns] = useState<Turn[]>([]);
  const [busy, setBusy] = useState(false);
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const t = sessionStorage.getItem("token");
    if (t) {
      setToken(t);
      setRole(sessionStorage.getItem("role") ?? "");
    }
  }, []);
  useEffect(() => endRef.current?.scrollIntoView({ behavior: "smooth" }), [turns]);

  async function login(e: FormEvent<HTMLFormElement>) {
    e.preventDefault();
    const f = new FormData(e.currentTarget);
    setLoginErr("");
    try {
      const r = await fetch(`${API}/auth/login`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email: f.get("email"), password: f.get("password") }),
      });
      if (r.status === 401) return setLoginErr("Wrong email or password.");
      if (r.status === 429) return setLoginErr("Too many failed attempts. Wait a few minutes and try again.");
      if (!r.ok) return setLoginErr("The server could not log you in. Try again.");
      const d = await r.json();
      sessionStorage.setItem("token", d.token);
      sessionStorage.setItem("role", d.role);
      setToken(d.token);
      setRole(d.role);
    } catch {
      setLoginErr("Cannot reach the server. Check that the API is running.");
    }
  }

  function logout() {
    sessionStorage.clear();
    setToken(null);
    setTurns([]);
  }

  async function ask(e: FormEvent<HTMLFormElement>) {
    e.preventDefault();
    const form = e.currentTarget;
    const q = (new FormData(form).get("q") as string).trim();
    if (!q || busy) return;
    form.reset();
    setBusy(true);
    setTurns((t) => [...t, { question: q }]);
    try {
      const r = await fetch(`${API}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json", Authorization: `Bearer ${token}` },
        body: JSON.stringify({ question: q }),
      });
      if (r.status === 401) return logout();
      if (!r.ok) throw new Error("The server could not answer. Try again.");
      const reply: Reply = await r.json();
      setTurns((t) => t.map((x, i) => (i === t.length - 1 ? { ...x, reply } : x)));
    } catch (err) {
      setTurns((t) => t.map((x, i) => (i === t.length - 1 ? { ...x, error: String(err) } : x)));
    } finally {
      setBusy(false);
    }
  }

  if (!token) {
    return (
      <main style={{ maxWidth: 360, margin: "12vh auto", padding: 16 }}>
        <h1 style={{ fontSize: 22 }}>Campus Knowledge Assistant</h1>
        <form onSubmit={login} style={{ ...card, display: "grid", gap: 10 }}>
          <input name="email" type="email" placeholder="University email" required style={inputStyle} />
          <input name="password" type="password" placeholder="Password" required style={inputStyle} />
          {loginErr && <span style={{ color: "#b3261e", fontSize: 14 }}>{loginErr}</span>}
          <button style={btnStyle}>Log in</button>
        </form>
      </main>
    );
  }

  return (
    <main style={{ maxWidth: 780, margin: "0 auto", padding: 16, minHeight: "100vh", display: "flex", flexDirection: "column" }}>
      <header style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "8px 0 16px" }}>
        <h1 style={{ fontSize: 20, margin: 0 }}>Campus Knowledge Assistant</h1>
        <span style={{ fontSize: 14 }}>
          {role} ·{" "}
          {role === "admin" && (
            <>
              <a href="/admin" style={{ color: "#1f4fd8" }}>dashboard</a> ·{" "}
            </>
          )}
          <button onClick={logout} style={{ ...linkBtn }}>log out</button>
        </span>
      </header>

      <section style={{ flex: 1, display: "grid", gap: 16, alignContent: "start" }}>
        {turns.length === 0 && (
          <p style={{ color: "#5b6470" }}>
            Try: “Can I take the machine learning course before statistics?” Answers come only from official
            documents you have access to, with the source cited.
          </p>
        )}
        {turns.map((t, i) => (
          <div key={i} style={{ display: "grid", gap: 8 }}>
            <div style={{ justifySelf: "end", background: "#1f4fd8", color: "#fff", padding: "8px 14px", borderRadius: 12, maxWidth: "85%" }}>
              {t.question}
            </div>
            <div style={card}>
              {t.error ? (
                <span style={{ color: "#b3261e" }}>{t.error}</span>
              ) : !t.reply ? (
                "Searching documents…"
              ) : (
                <>
                  {t.reply.warnings.map((w, j) => (
                    <p key={j} style={{ background: "#fff4d6", border: "1px solid #f0d78c", borderRadius: 8, padding: "6px 10px", fontSize: 14, margin: "0 0 10px" }}>
                      ⚠ {w}
                    </p>
                  ))}
                  <p style={{ margin: 0, whiteSpace: "pre-wrap", lineHeight: 1.5 }}>{t.reply.answer}</p>
                  {t.reply.citations.length > 0 && (
                    <div style={{ marginTop: 12, display: "grid", gap: 6 }}>
                      <strong style={{ fontSize: 13, color: "#5b6470" }}>Sources</strong>
                      {t.reply.citations.map((c) => (
                        <details key={c.n} style={{ fontSize: 14 }}>
                          <summary style={{ cursor: "pointer" }}>
                            [{c.n}] {c.title} ({c.year}
                            {c.page ? `, p. ${c.page}` : ""})
                          </summary>
                          <blockquote style={{ margin: "6px 0 0", paddingLeft: 10, borderLeft: "3px solid #d5d9df", color: "#444" }}>
                            {c.excerpt}…
                            {c.source_url && (
                              <>
                                {" "}
                                <a href={c.source_url} target="_blank" rel="noreferrer">
                                  open document
                                </a>
                              </>
                            )}
                          </blockquote>
                        </details>
                      ))}
                    </div>
                  )}
                </>
              )}
            </div>
          </div>
        ))}
        <div ref={endRef} />
      </section>

      <form onSubmit={ask} style={{ display: "flex", gap: 8, position: "sticky", bottom: 0, padding: "12px 0", background: "#f6f7f9" }}>
        <input name="q" placeholder="Ask about a policy, course, or rule…" autoComplete="off" style={{ ...inputStyle, flex: 1 }} />
        <button disabled={busy} style={btnStyle}>
          {busy ? "…" : "Ask"}
        </button>
      </form>
    </main>
  );
}

const inputStyle: React.CSSProperties = {
  padding: "10px 12px",
  border: "1px solid #cfd4da",
  borderRadius: 8,
  fontSize: 16,
};
const btnStyle: React.CSSProperties = {
  padding: "10px 18px",
  border: 0,
  borderRadius: 8,
  background: "#1f4fd8",
  color: "#fff",
  fontSize: 16,
  cursor: "pointer",
};
const linkBtn: React.CSSProperties = { background: "none", border: 0, color: "#1f4fd8", cursor: "pointer", padding: 0, fontSize: 14 };
