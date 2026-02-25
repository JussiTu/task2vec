"""
simulate_solve.py
=================
Picks one resolved ticket from each AI-readiness tier (Automate / Assist /
Escalate), asks GPT-4o-mini to solve it cold, and prints the results
side-by-side so you can see how answer quality tracks with tier.

Usage:
    python simulate_solve.py
    python simulate_solve.py --seed 99   # different ticket sample
"""
import argparse, json, os, random, textwrap
from pathlib import Path
from pymongo import MongoClient
from openai import OpenAI

# ── Config ───────────────────────────────────────────────────────────────────
CACHE          = Path(__file__).parent / ".cache"
SIGNALS_FILE   = CACHE / "outcome_signals.json"
MONGO_URI      = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MODEL          = "gpt-4o-mini"
MAX_DESC_CHARS = 1200   # truncate very long descriptions
WRAP_WIDTH     = 80

TIERS = ["Automate", "Assist", "Escalate"]

SYSTEM_PROMPT = (
    "You are an experienced Spring Framework engineer. "
    "A colleague has filed a Jira ticket and you need to resolve it. "
    "Read the ticket and respond with a concrete solution — code snippet, "
    "config change, explanation of the fix, or a clear diagnosis. "
    "Be specific. Do not ask for more information."
)

# ── Helpers ──────────────────────────────────────────────────────────────────

def load_signals():
    with open(SIGNALS_FILE, encoding="utf-8") as f:
        data = json.load(f)
    return data["signals"], data["calibration"]

def pick_ticket_per_tier(signals, rng):
    """Return {tier: key} with one random key per tier."""
    buckets = {t: [] for t in TIERS}
    for key, sig in signals.items():
        buckets[sig["label"]].append(key)
    return {t: rng.choice(buckets[t]) for t in TIERS}

def fetch_tickets(keys):
    """Return {key: {summary, description}} from MongoDB."""
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    coll   = client["jiradump"]["Spring"]
    result = {}
    for doc in coll.find({"key": {"$in": list(keys)}},
                          {"key": 1, "fields.summary": 1, "fields.description": 1}):
        f    = doc["fields"]
        desc = f.get("description") or ""
        if len(desc) > MAX_DESC_CHARS:
            desc = desc[:MAX_DESC_CHARS] + "\n[…truncated]"
        result[doc["key"]] = {
            "summary":     f.get("summary", ""),
            "description": desc,
        }
    client.close()
    return result

def ask_gpt(client, ticket):
    user_msg = f"**Summary:** {ticket['summary']}\n\n**Description:**\n{ticket['description']}"
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ],
        temperature=0.3,
        max_tokens=500,
    )
    return resp.choices[0].message.content.strip()

def hr(char="-", width=WRAP_WIDTH):
    return char * width

def wrap(text, indent=4):
    lines = []
    for para in text.splitlines():
        if para.strip() == "":
            lines.append("")
        else:
            lines.extend(textwrap.wrap(para, width=WRAP_WIDTH - indent,
                                       initial_indent=" " * indent,
                                       subsequent_indent=" " * indent))
    return "\n".join(lines)

TIER_ICONS = {"Automate": "[Automate]", "Assist": "[Assist]", "Escalate": "[Escalate]"}

# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    print("\nLoading outcome signals …")
    signals, cal = load_signals()
    print(f"  {len(signals):,} signals loaded. "
          f"Calibration: p33={cal['p33_days']}d, p67={cal['p67_days']}d, "
          f"p75_assignee={cal['p75_assignee']} tickets\n")

    picks = pick_ticket_per_tier(signals, rng)
    print("Selected tickets:")
    for tier, key in picks.items():
        sig = signals[key]
        print(f"  {tier:10s}  {key}  "
              f"(days={sig['days']}, watches={sig['watches']}, "
              f"assignee_exp={sig['assignee_exp']})")

    print("\nFetching ticket text from MongoDB …")
    ticket_data = fetch_tickets(set(picks.values()))

    print("Calling GPT-4o-mini …\n")
    client = OpenAI()

    print(hr("="))
    print("SIMULATION: Can an AI solve these tickets?")
    print(f"Model: {MODEL}   Seed: {args.seed}")
    print(hr("="))

    for tier in TIERS:
        key    = picks[tier]
        ticket = ticket_data.get(key, {"summary": "(not found)", "description": ""})
        sig    = signals[key]
        icon   = TIER_ICONS[tier]

        print(f"\n{icon}  TIER: {tier.upper()}   |   {key}")
        print(f"    days={sig['days']}  watches={sig['watches']}  "
              f"assignee_exp={sig['assignee_exp']}")
        print(hr())

        print("TICKET:")
        print(wrap(f"Summary: {ticket['summary']}"))
        if ticket["description"].strip():
            print(wrap(ticket["description"]))

        print(hr("."))
        print("GPT RESPONSE:")
        answer = ask_gpt(client, ticket)
        print(wrap(answer))
        print(hr())

    print("\nDone. Notice:")
    print("  [Automate] -> specific, confident, actionable answer")
    print("  [Assist]   -> reasonable draft, may need review")
    print("  [Escalate] -> vague, hedging, or asks for more context\n")

if __name__ == "__main__":
    main()
