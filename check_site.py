"""
check_site.py — task2vec.com site health check
Usage: python check_site.py
"""
import json, sys, urllib.request, urllib.error, time

BASE = "https://task2vec.com"

PAGES = [
    "/",
    "/score.html",
    "/ai_readiness_chart.html",
    "/cockpit.html",
    "/workintelligence.html",
    "/stories_spring/spring_explorer.html",
]

API_CHECKS = [
    # (description, url, method, body, expected_key)
    ("health",         "/api/health.php",  "GET",  None,
        lambda r: r.get("status") == "ok"),
    ("score (simple)", "/api/score.php",   "POST", {"text": "Fix typo in documentation"},
        lambda r: r.get("tier") in ("Automate","Assist","Escalate") and r.get("coverage",0) > 0),
    ("score (hard)",   "/api/score.php",   "POST", {"text": "Implement reactive WebSocket failover with backpressure in Spring WebFlux"},
        lambda r: r.get("tier") == "Escalate"),
    ("analyze",        "/api/analyze.php", "POST", {"text": "Fix NPE in RedisTemplate"},
        lambda r: "similar" in r and len(r["similar"]) > 0),
]

GREEN = "\033[92m"
RED   = "\033[91m"
RESET = "\033[0m"
BOLD  = "\033[1m"

def check(symbol, label, detail=""):
    color = GREEN if symbol == "OK" else RED
    mark  = "+" if symbol == "OK" else "x"
    print(f"  [{color}{mark}{RESET}] {label}" + (f"  {detail}" if detail else ""))
    return symbol == "OK"

def fetch(url, method="GET", body=None, timeout=30):
    data = json.dumps(body).encode() if body else None
    req  = urllib.request.Request(BASE + url, data=data, method=method)
    if body:
        req.add_header("Content-Type", "application/json")
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read().decode(), round(time.time()-t0, 2)
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode(), round(time.time()-t0, 2)
    except Exception as e:
        return 0, str(e), round(time.time()-t0, 2)

def main():
    print(f"\n{BOLD}task2vec.com site check{RESET}  ({BASE})\n")
    all_ok = True

    # Pages
    print(f"{BOLD}Pages{RESET}")
    for path in PAGES:
        code, body, secs = fetch(path)
        ok = code == 200
        all_ok &= check("OK" if ok else "FAIL",
                         path,
                         f"{code}  {secs}s")

    # API
    print(f"\n{BOLD}API{RESET}")
    for desc, path, method, body, validator in API_CHECKS:
        code, raw, secs = fetch(path, method, body, timeout=90)
        try:
            resp = json.loads(raw)
            ok   = code == 200 and validator(resp)
            detail = ""
            if not ok and "error" in resp:
                detail = resp["error"]
            elif ok and "tier" in resp:
                detail = f"tier={resp['tier']} conf={resp.get('confidence','')} cov={resp.get('coverage','')}"
            elif ok and "status" in resp:
                detail = f"indexed={resp.get('indexed','')}"
        except Exception:
            ok     = False
            detail = raw[:80]
        all_ok &= check("OK" if ok else "FAIL",
                         f"{method} {path}  ({desc})",
                         f"{code}  {secs}s  {detail}")

    print()
    if all_ok:
        print(f"{GREEN}{BOLD}All checks passed.{RESET}")
    else:
        print(f"{RED}{BOLD}Some checks FAILED.{RESET}")
        sys.exit(1)

if __name__ == "__main__":
    main()
