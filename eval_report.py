"""
eval_report.py
==============
Reads .cache/eval_results.json and prints a human-readable summary of the
git-based eval: how well Claude reproduces actual Spring Framework fixes,
broken down by AI-readiness tier.

Usage:
    python eval_report.py
    python eval_report.py --verbose   # show all individual tickets
"""
import argparse, json
from collections import defaultdict
from pathlib import Path

RESULTS_FILE = Path(__file__).parent / ".cache" / "eval_results.json"
TIERS        = ["Automate", "Assist", "Escalate"]
W            = 70   # console width


def hr(c="-"):
    return c * W


def pct(n, d):
    return f"{n/d*100:.0f}%" if d else "n/a"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show every ticket result")
    args = parser.parse_args()

    if not RESULTS_FILE.exists():
        print(f"No results found at {RESULTS_FILE}")
        print("Run:  python run_eval.py")
        return

    with open(RESULTS_FILE, encoding="utf-8") as f:
        results = json.load(f)

    by_tier = defaultdict(list)
    for r in results:
        by_tier[r["tier"]].append(r)

    # ── Header ────────────────────────────────────────────────────────────────
    print()
    print(hr("="))
    print("GIT-BASED EVAL: Can Claude reproduce actual Spring Framework fixes?")
    print(hr("="))
    print(f"{'Tier':<12} {'N':>4}  {'Pass%':>6}  {'FileHit%':>9}  "
          f"{'AvgOverlap':>11}  {'Tokens(avg)':>12}")
    print(hr())

    totals = {"n": 0, "pass": 0, "fhit": 0, "ovlp": 0.0,
              "in_tok": 0, "out_tok": 0}

    for tier in TIERS:
        recs = by_tier[tier]
        if not recs:
            continue
        n        = len(recs)
        passes   = sum(1 for r in recs if r["pass"])
        fhits    = sum(1 for r in recs if r["file_hit"])
        avg_ovlp = sum(r["token_overlap"] for r in recs) / n
        avg_in   = sum(r["input_tokens"]  for r in recs) / n
        avg_out  = sum(r["output_tokens"] for r in recs) / n

        totals["n"]      += n
        totals["pass"]   += passes
        totals["fhit"]   += fhits
        totals["ovlp"]   += sum(r["token_overlap"] for r in recs)
        totals["in_tok"] += sum(r["input_tokens"]  for r in recs)
        totals["out_tok"]+= sum(r["output_tokens"] for r in recs)

        print(f"{tier:<12} {n:>4}  {pct(passes,n):>6}  {pct(fhits,n):>9}  "
              f"{avg_ovlp:>10.2f}  "
              f"{avg_in:>5.0f}in/{avg_out:>4.0f}out")

    print(hr())
    n = totals["n"]
    if n:
        avg_ovlp = totals["ovlp"] / n
        avg_in   = totals["in_tok"]  / n
        avg_out  = totals["out_tok"] / n
        print(f"{'TOTAL':<12} {n:>4}  {pct(totals['pass'],n):>6}  "
              f"{pct(totals['fhit'],n):>9}  {avg_ovlp:>10.2f}  "
              f"{avg_in:>5.0f}in/{avg_out:>4.0f}out")

    # ── Scoring legend ────────────────────────────────────────────────────────
    print()
    print("Scoring key:")
    print("  Pass      = identified correct file/class  AND  token_overlap >= 15%")
    print("  FileHit   = Claude's answer mentions the name of the changed class")
    print("  Overlap   = Jaccard of code tokens: Claude's answer vs. ground-truth")
    print("              added lines in the actual commit diff")

    # ── Sample results ────────────────────────────────────────────────────────
    print()
    print(hr("="))
    print("SAMPLE RESULTS BY TIER")
    print(hr("="))

    for tier in TIERS:
        recs = by_tier[tier]
        if not recs:
            continue
        show = recs if args.verbose else recs[:3]
        print(f"\n[{tier}]  ({len(recs)} tickets evaluated)")
        print(hr("-"))
        for r in show:
            status = "PASS" if r["pass"] else "FAIL"
            fh     = "Y" if r["file_hit"] else "N"
            print(f"  {status}  {r['key']:<14}  overlap={r['token_overlap']:.2f}  "
                  f"file_hit={fh}")
            print(f"       Summary : {r['summary'][:60]}")
            print(f"       Files   : {', '.join(Path(f).name for f in r['files'][:2])}")
            if args.verbose and r.get("answer_preview"):
                print(f"       Claude  : {r['answer_preview'][:120].strip()!r}")
            print()
        if not args.verbose and len(recs) > 3:
            print(f"  … {len(recs)-3} more — run with --verbose to see all")

    # ── Interpretation ────────────────────────────────────────────────────────
    print()
    print(hr("="))
    print("INTERPRETATION")
    print(hr("="))
    print("""
The three tiers predict how easy a ticket is for AI:

  Automate  — fast to resolve, low watcher count, routine contributor.
               Hypothesis: Claude should reproduce these fixes most reliably
               (well-understood bug patterns, single-file changes).

  Assist    — middle ground; Claude can draft a fix but may miss edge cases.

  Escalate  — slow, watched by many, often tackled by senior contributors.
               Hypothesis: Claude should struggle most here — these are
               architectural, ambiguous, or require deep codebase knowledge.

If pass-rates track Automate > Assist > Escalate, the outcome-based tier
labels are empirically validated against actual code quality.
""")


if __name__ == "__main__":
    main()
