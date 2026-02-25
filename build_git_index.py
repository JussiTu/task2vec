"""
build_git_index.py
==================
Clones the Spring Framework git repository (one-time, ~244 MB) and builds a
mapping from SPR-#### Jira keys to the commit that fixed them.

Output: .cache/git_index.json
  {
    "SPR-12345": {
      "sha":     "abc123def...",
      "parent":  "000fff...",
      "message": "SPR-12345 Fix NPE in RedisTemplate",
      "files":   ["spring-data/src/main/java/org/.../Foo.java"],
      "label":   "Automate"
    },
    ...
  }

Only includes tickets that:
  - appear in outcome_signals.json  (have an Automate/Assist/Escalate label)
  - changed at least one non-test .java file

Run once; subsequent runs skip the clone if the repo already exists.

Usage:
    python build_git_index.py
    python build_git_index.py --skip-fetch   # offline / speed up reruns
"""
import argparse, json, re, subprocess, sys
from collections import Counter
from pathlib import Path

CACHE    = Path(__file__).parent / ".cache"
REPO_DIR = CACHE / "spring-framework"
REPO_URL = "https://github.com/spring-projects/spring-framework.git"

SIGNALS_FILE = CACHE / "outcome_signals.json"
OUTPUT_FILE  = CACHE / "git_index.json"


# ── Git helpers ───────────────────────────────────────────────────────────────

def run(cmd, cwd=None, check=False):
    r = subprocess.run(cmd, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", cwd=cwd)
    if check and r.returncode != 0:
        print(f"ERROR: {' '.join(cmd)}\n{r.stderr[:400]}")
        sys.exit(1)
    return r.stdout


def clone_or_fetch(skip_fetch: bool):
    if REPO_DIR.exists():
        if skip_fetch:
            print("Repo exists — skipping fetch (--skip-fetch).")
        else:
            print("Repo exists. Fetching latest …")
            run(["git", "fetch", "--quiet", "--all"], cwd=REPO_DIR)
            print("  done.")
    else:
        print(f"Cloning spring-framework (~244 MB, this takes a few minutes) …")
        r = subprocess.run(
            ["git", "clone", "--quiet", REPO_URL, str(REPO_DIR)],
            capture_output=True, text=True,
        )
        if r.returncode != 0:
            print(f"Clone failed:\n{r.stderr}")
            sys.exit(1)
        print("  Clone complete.")


def iter_spr_commits():
    """
    Stream all commits that mention SPR- in the subject line.
    Yields (sha, parent_sha, subject).
    Parent is the first parent only (ignores merge commits' second parent).
    """
    out = run(
        ["git", "log", "--all", "--format=%H|%P|%s", "--grep=SPR-"],
        cwd=REPO_DIR,
    )
    for line in out.splitlines():
        parts = line.split("|", 2)
        if len(parts) < 3:
            continue
        sha, parents, subject = parts
        sha     = sha.strip()
        subject = subject.strip()
        # parents may be "sha1 sha2" for merge commits — take first
        parent  = parents.strip().split()[0] if parents.strip() else ""
        if sha and parent:
            yield sha, parent, subject


def get_java_source_files(sha):
    """Return changed .java files that are NOT in a test directory."""
    out = run(
        ["git", "diff-tree", "--no-commit-id", "-r", "--name-only", sha],
        cwd=REPO_DIR,
    )
    return [
        f.strip() for f in out.splitlines()
        if f.endswith(".java")
        and "/test/" not in f
        and "/test-" not in f
        and "Test" not in Path(f).name  # e.g. FooTests.java
    ]


# ── Key extraction ────────────────────────────────────────────────────────────

SPR_RE = re.compile(r"\bSPR-(\d+)\b", re.IGNORECASE)

def extract_key(subject):
    m = SPR_RE.search(subject)
    return f"SPR-{m.group(1)}" if m else None


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-fetch", action="store_true",
                        help="Don't run git fetch (useful offline)")
    args = parser.parse_args()

    CACHE.mkdir(exist_ok=True)

    # Load labeled tickets
    print("Loading outcome signals …")
    with open(SIGNALS_FILE, encoding="utf-8") as f:
        data = json.load(f)
    signals = data["signals"]
    labeled = set(signals.keys())
    print(f"  {len(labeled):,} labeled tickets")

    clone_or_fetch(args.skip_fetch)

    # Pass 1: scan all SPR commits, keep only labeled ones
    print("\nScanning git log for SPR-#### commits …")
    candidates = {}   # key → (sha, parent, subject)
    total_spr  = 0
    skip_multi = 0

    for sha, parent, subject in iter_spr_commits():
        total_spr += 1
        key = extract_key(subject)
        if key is None or key not in labeled:
            continue
        if key in candidates:
            skip_multi += 1
            continue   # keep first (most recent in log order)
        candidates[key] = (sha, parent, subject)

    print(f"  {total_spr:,} total SPR commits found")
    print(f"  {len(candidates):,} matched labeled tickets "
          f"({skip_multi} keys had multiple commits — kept most recent)")

    # Pass 2: get changed java files for each candidate
    print("\nExtracting changed Java source files …")
    index   = {}
    no_java = 0

    for i, (key, (sha, parent, subject)) in enumerate(candidates.items(), 1):
        if i % 100 == 0:
            print(f"  {i}/{len(candidates)} …", end="\r", flush=True)
        files = get_java_source_files(sha)
        if not files:
            no_java += 1
            continue
        index[key] = {
            "sha":     sha,
            "parent":  parent,
            "message": subject,
            "files":   files,
            "label":   signals[key]["label"],
        }

    print(f"\n  Skipped {no_java} commits with no Java source changes")

    # Summary
    counts = Counter(v["label"] for v in index.values())
    print(f"\nIndex built: {len(index):,} tickets with git commits + java changes")
    for tier in ["Automate", "Assist", "Escalate"]:
        n = counts[tier]
        pct = n / len(index) * 100 if index else 0
        print(f"  {tier:<12} {n:>4}  ({pct:.0f}%)")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)
    size_kb = OUTPUT_FILE.stat().st_size / 1024
    print(f"\nSaved: {OUTPUT_FILE}  ({size_kb:.0f} KB)")
    print("\nNext: python run_eval.py")


if __name__ == "__main__":
    main()
