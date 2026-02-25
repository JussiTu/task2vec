"""
eval_compare.py
===============
Reads all .cache/eval_results_*.json files and prints a side-by-side
comparison table of pass rates and token overlap by model and tier.

Usage:
    python eval_compare.py
    python eval_compare.py --verbose   # also print per-ticket detail per model
"""
import argparse, glob, json
from collections import defaultdict
from pathlib import Path

CACHE  = Path(__file__).parent / ".cache"
TIERS  = ["Automate", "Assist", "Escalate"]
W      = 80


def hr(c="-"):
    return c * W


def pct(n, d):
    return f"{n/d*100:.0f}%" if d else " n/a"


def load_all():
    """Return list of (model, results_list) sorted by model name."""
    files = sorted(CACHE.glob("eval_results_*.json"))
    if not files:
        return []
    runs = []
    for f in files:
        with open(f, encoding="utf-8") as fh:
            data = json.load(fh)
        model = data[0].get("model", f.stem.replace("eval_results_", "")) if data else "?"
        runs.append((model, data))
    return runs


def tier_stats(records):
    """Return dict of tier → {n, passes, fhits, ovlp, in_tok, out_tok}."""
    stats = {}
    for tier in TIERS:
        recs = [r for r in records if r["tier"] == tier]
        if not recs:
            continue
        n = len(recs)
        stats[tier] = {
            "n":       n,
            "passes":  sum(1 for r in recs if r["pass"]),
            "fhits":   sum(1 for r in recs if r["file_hit"]),
            "ovlp":    sum(r["token_overlap"] for r in recs) / n,
            "in_tok":  sum(r["input_tokens"]  for r in recs) / n,
            "out_tok": sum(r["output_tokens"] for r in recs) / n,
        }
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    runs = load_all()
    if not runs:
        print(f"No eval results found in {CACHE}")
        print("Run:  python run_eval.py --model <model>")
        return

    models = [m for m, _ in runs]
    all_stats = {m: tier_stats(r) for m, r in runs}

    # ── Summary table ─────────────────────────────────────────────────────────
    print()
    print(hr("="))
    print("MODEL COMPARISON — Spring Framework git-based eval")
    print(hr("="))

    # Header row
    col = 14
    tier_col = 10
    print(f"\n{'Pass rate by tier':}")
    print(f"{'Tier':<{tier_col}}", end="")
    for m in models:
        short = m[-col:].ljust(col)
        print(f"  {short}", end="")
    print()
    print(hr())

    for tier in TIERS:
        print(f"{tier:<{tier_col}}", end="")
        for m in models:
            s = all_stats[m].get(tier)
            if s:
                cell = f"{pct(s['passes'], s['n']):>5} (n={s['n']})"
            else:
                cell = "  —"
            print(f"  {cell:<{col}}", end="")
        print()

    # Total row
    print(hr())
    print(f"{'TOTAL':<{tier_col}}", end="")
    for m in models:
        recs = dict(runs)[m]
        n      = len(recs)
        passes = sum(1 for r in recs if r["pass"])
        print(f"  {pct(passes, n):>5} (n={n}){'':<{col-11}}", end="")
    print()

    # ── Token overlap table ────────────────────────────────────────────────────
    print(f"\n{'Avg token overlap by tier':}")
    print(f"{'Tier':<{tier_col}}", end="")
    for m in models:
        short = m[-col:].ljust(col)
        print(f"  {short}", end="")
    print()
    print(hr())

    for tier in TIERS:
        print(f"{tier:<{tier_col}}", end="")
        for m in models:
            s = all_stats[m].get(tier)
            cell = f"{s['ovlp']:.3f}" if s else " —"
            print(f"  {cell:<{col}}", end="")
        print()

    print(hr())
    print(f"{'TOTAL':<{tier_col}}", end="")
    for m in models:
        recs = dict(runs)[m]
        n    = len(recs)
        avg  = sum(r["token_overlap"] for r in recs) / n if n else 0
        print(f"  {avg:.3f}{'':<{col-5}}", end="")
    print()

    # ── Cost summary ──────────────────────────────────────────────────────────
    print(f"\n{'Avg tokens per ticket':}")
    print(f"{'Tier':<{tier_col}}", end="")
    for m in models:
        short = m[-col:].ljust(col)
        print(f"  {short}", end="")
    print()
    print(hr())

    for tier in TIERS:
        print(f"{tier:<{tier_col}}", end="")
        for m in models:
            s = all_stats[m].get(tier)
            cell = f"{s['in_tok']:.0f}in/{s['out_tok']:.0f}out" if s else "—"
            print(f"  {cell:<{col}}", end="")
        print()

    # ── Scoring reminder ──────────────────────────────────────────────────────
    print()
    print(hr("="))
    print("Scoring:  Pass = identified correct file/class  AND  token_overlap >= 0.15")
    print("Overlap:  Jaccard of code tokens — AI answer vs. ground-truth added lines")
    print(hr("="))

    # ── Verbose: per-ticket detail per model ──────────────────────────────────
    if args.verbose:
        for model, records in runs:
            print(f"\n{'='*W}")
            print(f"DETAIL: {model}")
            print("="*W)
            by_tier = defaultdict(list)
            for r in records:
                by_tier[r["tier"]].append(r)
            for tier in TIERS:
                recs = by_tier[tier]
                if not recs:
                    continue
                print(f"\n[{tier}]")
                print(hr())
                for r in recs:
                    st = "PASS" if r["pass"] else "FAIL"
                    print(f"  {st}  {r['key']:<14} ovlp={r['token_overlap']:.2f}  "
                          f"fhit={'Y' if r['file_hit'] else 'N'}")
                    print(f"       {r['summary'][:65]}")


if __name__ == "__main__":
    main()
