import os
import json
import time
from openai import OpenAI

# -----------------------------
# Key + client (robust)
# -----------------------------
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError(
        "OPENAI_API_KEY not found.\n"
        "PowerShell (temporary):  $env:OPENAI_API_KEY=\"sk-...\"\n"
        "PowerShell (permanent):  setx OPENAI_API_KEY \"sk-...\" ثم افتح نافذة جديدة"
    )

client = OpenAI(api_key=api_key)

SYSTEM = """You generate plausible but incorrect student answers.

You MUST:
- Follow the requested misconception (root cause).
- Stay incorrect in a diagnostically meaningful way.
- Sound confident and plausible.
- NOT mention that the answer is wrong.
- NOT add meta commentary.

Return JSON only, strictly matching the schema.
"""

ROOT_CAUSES = [
    {"id": "RC1", "desc": "Confuse the concept with a nearby concept (e.g., leakage = overfitting)."},
    {"id": "RC2", "desc": "Assume the most common example is the only possible case."},
    {"id": "RC3", "desc": "Claim a standard checklist or procedure guarantees correctness."},
    {"id": "RC4", "desc": "Fixate on low-level technical operations instead of the real mechanism."},
    {"id": "RC5", "desc": "Use everyday or security meaning instead of technical ML meaning."},
    {"id": "RC6", "desc": "Invert cause and effect."},
    {"id": "RC7", "desc": "Ignore boundaries between train, validation, and test data."},
    {"id": "RC8", "desc": "Treat a tool or method as universally safe or unsafe."},
    {"id": "RC9", "desc": "Ignore timing, scale, or deployment constraints."},
    {"id": "RC10", "desc": "Assume optimizing one metric solves the whole system."}
]

JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "root_cause_id": {"type": "string"},
        "answers": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1
        }
    },
    "required": ["root_cause_id", "answers"],
    "additionalProperties": False
}

def gen_wrong_answers(question: str, gold: str, rc_id: str, rc_desc: str, n: int = 15) -> dict:
    prompt = f"""
Question:
{question}

Gold answer (reference):
{gold}

Root cause to simulate: {rc_id}
Description: {rc_desc}

Generate {n} DISTINCT incorrect open answers that reflect ONLY this root cause.

Rules:
- Each answer should be 2–5 sentences.
- Each answer must be wrong in a meaningful way.
- Vary phrasing across answers.
- Do not include disclaimers or meta language.

Return JSON only.
"""

    r = client.responses.create(
        model="gpt-5",
        input=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt}
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "wrong_answers",
                "schema": JSON_SCHEMA
            }
        }
    )

    raw = r.output_text or ""
    print(f"\nRAW OUTPUT ({rc_id}, first 200 chars):\n{raw[:200]}\n")

    # Parse strict JSON
    return json.loads(raw)

def save_partial(path: str, payload: dict):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)

if __name__ == "__main__":
    QUESTION = "Explain data leakage and give two examples: one obvious and one subtle."
    GOLD = (
        "Data leakage is when information that would not be available at prediction time "
        "influences training, feature engineering, or model selection, inflating evaluation. "
        "Obvious: including the target label (or a direct proxy) as a feature. "
        "Subtle: fitting preprocessing/feature selection on the full dataset before splitting, "
        "or using future information in time-series."
    )

    OUT_PATH = "llm_wrong_answers.json"

    results = {
        "question": QUESTION,
        "gold": GOLD,
        "by_root_cause": {},   # rc_id -> list[str]
        "errors": {}           # rc_id -> error string
    }

    # If a partial file exists, resume
    if os.path.exists(OUT_PATH):
        try:
            with open(OUT_PATH, "r", encoding="utf-8") as f:
                prev = json.load(f)
            if isinstance(prev, dict) and "by_root_cause" in prev:
                results = prev
                print(f"Resuming from existing {OUT_PATH}. Already have: {list(results['by_root_cause'].keys())}")
        except Exception:
            print("Warning: could not load existing output, starting fresh.")

    for rc in ROOT_CAUSES:
        rc_id = rc["id"]
        if rc_id in results["by_root_cause"]:
            print(f"{rc_id} already done, skipping.")
            continue

        # Retry a couple times for transient errors
        max_tries = 3
        for attempt in range(1, max_tries + 1):
            try:
                out = gen_wrong_answers(QUESTION, GOLD, rc_id, rc["desc"], n=15)
                results["by_root_cause"][rc_id] = out["answers"]
                print(f"{rc_id} generated {len(out['answers'])}")
                save_partial(OUT_PATH, results)
                break
            except Exception as e:
                msg = f"{type(e).__name__}: {e}"
                print(f"ERROR on {rc_id} attempt {attempt}/{max_tries}: {msg}")
                if attempt == max_tries:
                    results["errors"][rc_id] = msg
                    save_partial(OUT_PATH, results)
                else:
                    time.sleep(2.0 * attempt)  # backoff

    print("\n✅ Done. Output saved to llm_wrong_answers.json")
    if results["errors"]:
        print("Some root causes failed:", results["errors"])
