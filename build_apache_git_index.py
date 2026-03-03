"""
build_apache_git_index.py
=========================
Clones three Apache repos as bare repos (no file content — much smaller)
and builds a mapping from Jira ticket keys to the commits that reference them.

Repos and key patterns:
  Apache Camel    -> CAMEL-####
  Apache Spark    -> SPARK-####
  Apache Hadoop   -> HADOOP-####

Uses --filter=blob:none so only commit + tree objects are downloaded.
Camel ~150 MB, Spark ~400 MB, Hadoop ~200 MB.

Output: .cache/apache_git_index.json
  {
    "CAMEL-12345": {
      "sha":     "abc123...",
      "year":    2018,
      "message": "CAMEL-12345: Fix race condition in RouteBuilder",
      "files":   ["components/camel-core/src/.../RouteBuilder.java"]
    },
    ...
  }

Run once; subsequent runs skip cloning if repo exists.

Usage:
    python build_apache_git_index.py
    python build_apache_git_index.py --skip-fetch
    python build_apache_git_index.py --projects camel spark   # subset
"""

import argparse, json, re, subprocess, sys
from pathlib import Path

CACHE     = Path(__file__).parent / ".cache"
REPOS_DIR = CACHE / "apache-repos"
OUTPUT    = CACHE / "apache_git_index.json"

PROJECTS = {
    "camel":  {
        "url":     "https://github.com/apache/camel.git",
        "prefix":  "CAMEL",
        "exts":    {".java", ".xml"},
    },
    "spark":  {
        "url":     "https://github.com/apache/spark.git",
        "prefix":  "SPARK",
        "exts":    {".java", ".scala", ".py"},
    },
    "hadoop": {
        "url":     "https://github.com/apache/hadoop.git",
        "prefix":  "HADOOP",
        "exts":    {".java"},
    },
}

# Source-only: skip test paths
TEST_HINTS = ("/test/", "/tests/", "/test-", "Test.java", "Tests.java",
              "Spec.scala", "Suite.scala", "test_", "_test.py")

def is_source(path: str) -> bool:
    return not any(h in path for h in TEST_HINTS)


def run(cmd, cwd=None):
    r = subprocess.run(cmd, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", cwd=cwd)
    if r.returncode != 0 and r.stderr:
        print(f"  [warn] {' '.join(str(c) for c in cmd[:4])}: {r.stderr[:200]}")
    return r.stdout


def clone_or_fetch(name: str, url: str, repo_dir: Path, skip_fetch: bool):
    if repo_dir.exists():
        if skip_fetch:
            print(f"  {name}: repo exists, skipping fetch.")
        else:
            print(f"  {name}: fetching latest …", end="", flush=True)
            run(["git", "--git-dir", str(repo_dir), "fetch", "--quiet", "--all"])
            print(" done.")
    else:
        print(f"  {name}: cloning (bare, no blobs) from {url} …")
        REPOS_DIR.mkdir(parents=True, exist_ok=True)
        r = subprocess.run(
            ["git", "clone", "--bare", "--filter=blob:none",
             "--quiet", url, str(repo_dir)],
            capture_output=True, text=True,
        )
        if r.returncode != 0:
            print(f"  Clone failed:\n{r.stderr[:400]}")
            sys.exit(1)
        print(f"  {name}: clone complete.")


def scan_commits(repo_dir: Path, prefix: str):
    """Yield (sha, year, message) for commits mentioning PREFIX-####."""
    pattern = f"{prefix}-"
    out = run(
        ["git", "--git-dir", str(repo_dir),
         "log", "--all", "--format=%H|%ai|%s", f"--grep={pattern}"]
    )
    for line in out.splitlines():
        parts = line.split("|", 2)
        if len(parts) < 3:
            continue
        sha, date_str, msg = parts
        sha = sha.strip()
        msg = msg.strip()
        try:
            year = int(date_str.strip()[:4])
        except ValueError:
            continue
        if sha and year >= 2000:
            yield sha, year, msg


def get_source_files(repo_dir: Path, sha: str, exts: set) -> list:
    """Return source file paths changed in this commit (no test files)."""
    out = run(
        ["git", "--git-dir", str(repo_dir),
         "diff-tree", "--no-commit-id", "-r", "--name-only", sha]
    )
    return [
        f.strip() for f in out.splitlines()
        if Path(f.strip()).suffix in exts and is_source(f.strip())
    ]


KEY_RE_CACHE = {}

def make_re(prefix):
    if prefix not in KEY_RE_CACHE:
        KEY_RE_CACHE[prefix] = re.compile(rf"\b{prefix}-(\d+)\b", re.IGNORECASE)
    return KEY_RE_CACHE[prefix]


def extract_key(msg: str, prefix: str) -> str | None:
    m = make_re(prefix).search(msg)
    return f"{prefix}-{m.group(1)}" if m else None


def process_project(name: str, cfg: dict, skip_fetch: bool) -> dict:
    repo_dir = REPOS_DIR / f"{name}.git"
    url      = cfg["url"]
    prefix   = cfg["prefix"]
    exts     = cfg["exts"]

    print(f"\n[{name.upper()}]")
    clone_or_fetch(name, url, repo_dir, skip_fetch)

    print(f"  Scanning commits for {prefix}-#### references …")
    candidates = {}   # key → (sha, year, msg)  — keep first occurrence
    total = 0

    for sha, year, msg in scan_commits(repo_dir, prefix):
        total += 1
        key = extract_key(msg, prefix)
        if key and key not in candidates:
            candidates[key] = (sha, year, msg)

    print(f"  {total:,} commits scanned, {len(candidates):,} unique ticket keys found")

    print(f"  Extracting changed source files …")
    index   = {}
    no_src  = 0

    for i, (key, (sha, year, msg)) in enumerate(candidates.items(), 1):
        if i % 500 == 0:
            print(f"    {i:,}/{len(candidates):,} …", end="\r", flush=True)
        files = get_source_files(repo_dir, sha, exts)
        if not files:
            no_src += 1
            continue
        index[key] = {
            "sha":     sha,
            "year":    year,
            "message": msg,
            "files":   files[:20],   # cap at 20 to keep file small
        }

    print(f"    {len(index):,} tickets with source changes "
          f"({no_src} skipped — no source files)")
    return index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-fetch", action="store_true",
                        help="Skip git fetch (offline/faster rerun)")
    parser.add_argument("--projects", nargs="+",
                        choices=list(PROJECTS.keys()),
                        default=list(PROJECTS.keys()),
                        help="Which projects to process (default: all three)")
    args = parser.parse_args()

    CACHE.mkdir(exist_ok=True)

    print("Building Apache git index")
    print(f"Projects: {', '.join(args.projects)}")
    print(f"Output:   {OUTPUT}")

    # Load existing index if present (so we can add to it incrementally)
    if OUTPUT.exists():
        with open(OUTPUT, encoding="utf-8") as f:
            full_index = json.load(f)
        print(f"\nLoaded existing index: {len(full_index):,} entries")
    else:
        full_index = {}

    for name in args.projects:
        cfg = PROJECTS[name]
        project_index = process_project(name, cfg, args.skip_fetch)
        full_index.update(project_index)
        print(f"  Total index size so far: {len(full_index):,}")

    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(full_index, f)   # no indent — keeps file small
    size_mb = OUTPUT.stat().st_size / 1_048_576
    print(f"\nSaved: {OUTPUT}  ({size_mb:.1f} MB)")
    print(f"Total entries: {len(full_index):,}")

    # Summary by project prefix
    by_prefix = {}
    for key in full_index:
        pfx = key.split("-")[0]
        by_prefix[pfx] = by_prefix.get(pfx, 0) + 1
    print("\nBreakdown:")
    for pfx, n in sorted(by_prefix.items(), key=lambda x: -x[1]):
        print(f"  {pfx:10s}  {n:>6,}")

    print("\nNext: run apache_learning_signals.py")


if __name__ == "__main__":
    main()
