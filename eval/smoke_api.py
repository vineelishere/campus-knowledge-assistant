"""End-to-end smoke test of the running API, as the three roles a user would log in as.

    python eval/smoke_api.py [BASE_URL] [--delay SECONDS]

Needs the API running and the sample data ingested, with these users:
    student@mru.edu / professor: prof@mru.edu / admin@mru.edu  (password pass123)
Uses only the standard library. Each check prints PASS or FAIL; exit code 1 if any fail.
"""
import argparse
import json
import sys
import time
import urllib.error
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("base", nargs="?", default="http://localhost:8000")
ap.add_argument("--delay", type=float, default=3.0, help="seconds between /ask calls (free-tier rate limits)")
args = ap.parse_args()
BASE = args.base.rstrip("/")

USERS = {
    "student": ("student@mru.edu", "pass123"),
    "professor": ("prof@mru.edu", "pass123"),
    "admin": ("admin@mru.edu", "pass123"),
}
failures = 0


def call(method: str, path: str, body=None, token: str | None = None):
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(BASE + path, method=method, data=data, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read() or b"{}")
        except Exception:
            return e.code, {}


def check(label: str, ok: bool, detail: str = "") -> None:
    global failures
    failures += 0 if ok else 1
    print(f"[{'PASS' if ok else 'FAIL'}] {label}" + (f"\n       {detail}" if detail else ""))


def ask(role: str, question: str) -> dict:
    status, d = call("POST", "/ask", {"question": question}, TOKENS[role])
    time.sleep(args.delay)
    d["_http"] = status
    return d


def titles(d: dict) -> set[str]:
    return {c["title"] for c in d.get("citations", [])}


def show(d: dict) -> str:
    cites = ", ".join(f"{c['title']} p.{c.get('page')}" for c in d.get("citations", [])) or "none"
    return f"[{d.get('status')}, {d.get('latency_ms')} ms] {str(d.get('answer'))[:160]!r} | cites: {cites}"


# ---- login ----
TOKENS: dict[str, str] = {}
for role, (email, pw) in USERS.items():
    st, d = call("POST", "/auth/login", {"email": email, "password": pw})
    check(f"login as {role}", st == 200 and d.get("role") == role, f"http {st}")
    if st == 200:
        TOKENS[role] = d["token"]
if len(TOKENS) < 3:
    sys.exit("Cannot continue without all three users. Create them with: python -m app.auth EMAIL PASSWORD ROLE")

st, _ = call("POST", "/auth/login", {"email": "student@mru.edu", "password": "wrong"})
check("wrong password is rejected", st == 401, f"http {st}")
st, _ = call("POST", "/ask", {"question": "anything"})
check("asking without logging in is rejected", st == 401, f"http {st}")

# ---- admin-only endpoint ----
st, _ = call("GET", "/admin/metrics", token=TOKENS["student"])
check("student cannot open the admin dashboard data", st == 403, f"http {st}")
st, d = call("GET", "/admin/metrics", token=TOKENS["admin"])
check("admin can open the dashboard data", st == 200 and "summary" in d, f"http {st}")

# ---- questions ----
d = ask("student", "Can I take the machine learning course before taking statistics?")
check("student: prerequisite question is answered and cites the syllabus",
      d.get("status") == "answered" and "B.Tech CSE Syllabus 2026" in titles(d), show(d))

d = ask("student", "What is the minimum attendance required to sit for end semester exams?")
check("student: attendance answer uses the 2026 rule (80%) and never cites 2024",
      d.get("status") == "answered" and "80" in d.get("answer", "") and "Academic Regulations 2024" not in titles(d), show(d))

d = ask("student", "What is the cafeteria's Sunday menu?")
check("student: unrelated question gets 'I do not know'",
      d.get("status") != "answered" and "do not know" in d.get("answer", "").lower(), show(d))

d = ask("student", "What are the promotion criteria for associate professors?")
check("student: faculty-only question is refused and no faculty document is cited",
      d.get("status") != "answered" and "Faculty Handbook 2026" not in titles(d), show(d))

d = ask("professor", "What are the promotion criteria for associate professors?")
check("professor: same question is answered from the Faculty Handbook",
      d.get("status") == "answered" and "Faculty Handbook 2026" in titles(d), show(d))

d = ask("professor", "Who can award grace marks and how many?")
check("professor: admin-only question is refused", d.get("status") != "answered" and "Exam Cell Procedures 2026" not in titles(d), show(d))

d = ask("admin", "Who can award grace marks and how many?")
check("admin: same question is answered from the Exam Cell document",
      d.get("status") == "answered" and "Exam Cell Procedures 2026" in titles(d), show(d))

print(f"\n{'ALL PASSED' if not failures else str(failures) + ' CHECK(S) FAILED'}")
sys.exit(1 if failures else 0)
