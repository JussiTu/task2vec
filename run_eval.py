"""
run_eval.py
===========
For a sample of labeled tickets that have git commits, asks Claude to fix the
ticket given only the ticket text + before-state source files, then scores
Claude's answer against the actual commit diff.

Scoring per ticket:
  file_hit      : Claude's response mentions the correct file / class name
  token_overlap : Jaccard of code tokens (ground-truth added lines vs answer)
  pass          : file_hit AND token_overlap >= 0.15

Results are saved to .cache/eval_results.json.

Usage:
    python run_eval.py               # 5 tickets per tier = 15 total
    python run_eval.py --n 10        # 10 per tier
    python run_eval.py --tier Automate
    python run_eval.py --seed 7
    python run_eval.py --model claude-haiku-4-5-20251001   # cheaper
    python run_eval.py --dry-run     # show sample without calling Claude

Requirements:
    pip install anthropic pymongo
    set ANTHROPIC_API_KEY=...
"""
import argparse, json, os, random, re, subprocess, textwrap
from pathlib import Path

CACHE        = Path(__file__).parent / ".cache"
REPO_DIR     = CACHE / "spring-framework"
GIT_INDEX    = CACHE / "git_index.json"
SIGNALS_FILE = CACHE / "outcome_signals.json"
RESULTS_FILE = CACHE / "eval_results.json"

MONGO_URI   = os.getenv("MONGO_URI", "mongodb://localhost:27017")
TIERS       = ["Automate", "Assist", "Escalate"]

MAX_FILE_CHARS = 6000   # truncate very large source files
MAX_FILES      = 3      # max changed files shown to Claude

DEFAULT_MODEL = "claude-sonnet-4-6"

SYSTEM_PROMPT = (
    "You are a senior Spring Framework engineer doing a code review. "
    "You are given a Jira ticket and the current (unfixed) source file(s). "
    "Write the minimal code fix — show exactly which lines change. "
    "Format your fix as a unified diff or clearly mark old/new lines. "
    "Be specific. Do not ask for more information."
)


# ── Git helpers ───────────────────────────────────────────────────────────────

def git(cmd, cwd=REPO_DIR):
    r = subprocess.run(cmd, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", cwd=cwd)
    return r.stdout


def file_at_commit(sha, path):
    """File content at a given commit (before the fix = parent SHA)."""
    out = git(["git", "show", f"{sha}:{path}"])
    if len(out) > MAX_FILE_CHARS:
        out = out[:MAX_FILE_CHARS] + "\n// [file truncated — rest not shown]\n"
    return out


def diff_between(parent, sha):
    """Unified diff of the actual fix."""
    return git(["git", "diff", parent, sha])


# ── Ticket text from MongoDB ──────────────────────────────────────────────────

def load_ticket(key):
    """Return (summary, description) from MongoDB or ("", "")."""
    try:
        from pymongo import MongoClient
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=3000)
        doc = client["jiradump"]["Spring"].find_one(
            {"key": key},
            {"fields.summary": 1, "fields.description": 1},
        )
        client.close()
        if doc:
            f   = doc["fields"]
            desc = (f.get("description") or "")[:2000]
            return f.get("summary", key), desc
    except Exception:
        pass
    return key, ""


# ── Scoring ───────────────────────────────────────────────────────────────────

TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")

def code_tokens(text):
    return set(TOKEN_RE.findall(text))


def score(answer: str, ground_truth_diff: str):
    """
    Returns (file_hit: bool, token_overlap: float, passed: bool).

    file_hit      — does the answer mention the class name of any changed file?
    token_overlap — Jaccard of identifier tokens vs ground-truth added lines
    passed        — file_hit AND token_overlap >= 0.15
    """
    # Class names from diff header, e.g. "b/spring-core/.../RedisTemplate.java"
    diff_files  = re.findall(r"b/(.+\.java)", ground_truth_diff)
    class_names = {Path(f).stem for f in diff_files}   # {"RedisTemplate", …}

    file_hit = bool(class_names) and any(cn in answer for cn in class_names)

    # Added lines from the diff (skip file header lines starting with +++)
    added = "\n".join(
        l[1:] for l in ground_truth_diff.splitlines()
        if l.startswith("+") and not l.startswith("+++")
    )
    gt_tok  = code_tokens(added)
    ans_tok = code_tokens(answer)
    overlap = (
        len(gt_tok & ans_tok) / len(gt_tok | ans_tok)
        if gt_tok | ans_tok else 0.0
    )
    return file_hit, round(overlap, 3), file_hit and overlap >= 0.15


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",    type=int,  default=5,
                        help="tickets per tier (default 5)")
    parser.add_argument("--seed", type=int,  default=42)
    parser.add_argument("--tier", choices=TIERS, help="run a single tier only")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"Claude model ID (default: {DEFAULT_MODEL})")
    parser.add_argument("--dry-run", action="store_true",
                        help="show what would be evaluated, don't call Claude")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    # Load indexes
    print("Loading indexes …")
    with open(GIT_INDEX, encoding="utf-8") as f:
        git_index = json.load(f)
    with open(SIGNALS_FILE, encoding="utf-8") as f:
        signals = json.load(f)["signals"]
    print(f"  {len(git_index):,} tickets in git index")

    # Group by tier
    buckets = {t: [] for t in TIERS}
    for key, info in git_index.items():
        buckets[info["label"]].append(key)

    tiers_to_run = [args.tier] if args.tier else TIERS

    # Sample
    sample = {}
    for tier in tiers_to_run:
        pool = buckets[tier]
        n    = min(args.n, len(pool))
        sample[tier] = rng.sample(pool, n)
        print(f"  {tier:<12} {n} tickets selected  (pool = {len(pool)})")

    total = sum(len(v) for v in sample.values())
    print(f"\nTotal tickets to evaluate: {total}")
    if args.dry_run:
        print("\n[DRY RUN — not calling Claude]\n")

    # Claude client (only if not dry-run)
    claude = None
    if not args.dry_run:
        from anthropic import Anthropic
        claude = Anthropic()   # reads ANTHROPIC_API_KEY from env

    results = []

    for tier in tiers_to_run:
        for key in sample[tier]:
            info = git_index[key]
            print(f"\n{'='*60}")
            print(f"[{tier}]  {key}   sha={info['sha'][:8]}")

            summary, description = load_ticket(key)
            print(f"  Summary : {summary[:70]}")
            print(f"  Files   : {', '.join(Path(f).name for f in info['files'])}")

            if args.dry_run:
                continue

            # Build prompt: ticket + before-state source files
            parts = [
                f"**Ticket:** {key}",
                f"**Summary:** {summary}",
                f"**Description:**\n{description or '(no description)'}",
                "",
                "**Source files (before fix):**",
            ]
            files_shown = info["files"][:MAX_FILES]
            for fpath in files_shown:
                content = file_at_commit(info["parent"], fpath)
                parts.append(f"\n--- {fpath} ---\n```java\n{content}```")

            user_msg = "\n".join(parts)

            # Ground truth
            ground_truth = diff_between(info["parent"], info["sha"])

            # Ask Claude
            try:
                resp = claude.messages.create(
                    model=args.model,
                    max_tokens=1500,
                    system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": user_msg}],
                )
                answer       = resp.content[0].text
                in_tok       = resp.usage.input_tokens
                out_tok      = resp.usage.output_tokens
            except Exception as exc:
                print(f"  API error: {exc}")
                answer = ""
                in_tok = out_tok = 0

            # Score
            file_hit, token_overlap, passed = score(answer, ground_truth)

            status = "PASS" if passed else "FAIL"
            print(f"  {status}   file_hit={file_hit}   "
                  f"token_overlap={token_overlap:.2f}   "
                  f"tokens in={in_tok} out={out_tok}")

            results.append({
                "key":             key,
                "tier":            tier,
                "summary":         summary,
                "sha":             info["sha"],
                "files":           info["files"],
                "file_hit":        file_hit,
                "token_overlap":   token_overlap,
                "pass":            passed,
                "input_tokens":    in_tok,
                "output_tokens":   out_tok,
                "answer_preview":  answer[:400],
                "gt_diff_preview": ground_truth[:400],
            })

    if results:
        with open(RESULTS_FILE, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        total_cost_est = sum(
            r["input_tokens"] * 3e-6 + r["output_tokens"] * 15e-6
            for r in results
        )
        print(f"\nResults saved: {RESULTS_FILE}")
        print(f"Estimated cost (Sonnet pricing): ${total_cost_est:.3f}")
        print("Run:  python eval_report.py")
    elif not args.dry_run:
        print("\nNo results — check --n and --tier flags.")


if __name__ == "__main__":
    main()
