import argparse
import json
import os
import re
from collections import Counter

from tools import same_call

NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def save_jsonl(rows, path):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def numbers_in(text):
    text = re.sub(r"(?<=\d),(?=\d{3})", "", text)
    text = re.sub(r"[−–—](?=\d)", "-", text)
    return [float(x) for x in NUM_RE.findall(text)]


def answer_ok(final, check):
    if not final or not final.strip():
        return False
    found = numbers_in(final)
    for e in check.get("numbers", []):
        tol = max(0.011, 0.002 * abs(e))
        if not any(abs(x - e) <= tol or (e < 0 and abs(x + e) <= tol) for x in found):
            return False
    low = final.lower()
    return all(any(k in low for k in alts) for alts in check.get("keywords", []))


def match_calls(made, expected):
    # every expected call matched exactly once, order inside a step doesn't matter
    left = list(made)
    for e in expected:
        hit = next((m for m in left if same_call(m["name"], m["arguments"], e["name"], e["arguments"])), None)
        if hit is None:
            return False
        left.remove(hit)
    return not left


# per-test-case scoring + a failure category
def score(row, trace):
    expected = [c for step in row["steps"] for c in step]
    made = [c for s in trace["steps"] for c in s["calls"]]
    needs_tool = bool(expected)
    attempted = [s["attempted"] for s in trace["steps"]]
    final = trace["final"]
    last_is_call = bool(trace["steps"]) and trace["steps"][-1]["attempted"]

    out = {"id": row["id"], "kind": row["kind"], "needs_tool": needs_tool}
    out["success"] = final is not None and not last_is_call and answer_ok(final, row["check"])
    if needs_tool:
        out["valid"] = attempted[0] and all(s["valid"] for s in trace["steps"] if s["attempted"])
        out["accurate"] = match_calls(made, expected)
        out["passed"] = out["accurate"] and out["success"]
    else:
        out["abstained"] = not any(attempted)
        out["passed"] = out["abstained"] and out["success"]

    cat = None
    if not out["passed"]:
        exp_names, made_names = Counter(c["name"] for c in expected), Counter(c["name"] for c in made)
        if not needs_tool:
            cat = "unnecessary_call" if any(attempted) else "wrong_answer"
        elif not any(attempted):
            cat = "no_call"
        elif not out["valid"] and not made:
            cat = "wrong_format"
        elif any(n not in exp_names for n in made_names):
            cat = "wrong_tool"
        elif not out["accurate"] and made_names == exp_names:
            cat = "wrong_argument"
        elif not out["accurate"] and sum(made_names.values()) < sum(exp_names.values()):
            cat = "missing_call"
        elif not out["accurate"]:
            cat = "extra_call"
        elif final is None:
            cat = "no_final_answer"
        else:
            cat = "ignored_tool_result"
    out["category"] = cat
    return out


def summarize(scores):
    tool = [s for s in scores if s["needs_tool"]]
    none = [s for s in scores if not s["needs_tool"]]
    mean = lambda xs: round(100 * sum(xs) / max(1, len(xs)), 1)
    m = {
        "call_validity": mean([s["valid"] for s in tool]),
        "call_accuracy": mean([s["accurate"] for s in tool]),
        "abstention": mean([s["abstained"] for s in none]),
        "task_success": mean([s["success"] for s in scores]),
        "task_success_tool": mean([s["success"] for s in tool]),
        "task_success_no_tool": mean([s["success"] for s in none]),
        "fully_correct": mean([s["passed"] for s in scores]),
        "n_tool": len(tool), "n_no_tool": len(none),
        "per_kind_success": {},
        "failures": dict(Counter(s["category"] for s in scores if s["category"])),
    }
    for k in sorted({s["kind"] for s in scores}):
        m["per_kind_success"][k] = mean([s["passed"] for s in scores if s["kind"] == k])
    return m


def evaluate(model, tok, rows, name, system=None, out_dir="results"):
    from agent import SYSTEM_PROMPT, run_agent
    traces = run_agent(model, tok, [r["query"] for r in rows], system=system or SYSTEM_PROMPT)
    scores = [score(r, t) for r, t in zip(rows, traces)]
    os.makedirs(out_dir, exist_ok=True)
    save_jsonl([{**s, "query": r["query"], "trace": t} for r, t, s in zip(rows, traces, scores)],
               os.path.join(out_dir, f"{name}_cases.jsonl"))
    metrics = summarize(scores)
    with open(os.path.join(out_dir, f"{name}_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=1)
    return metrics, scores


def compare(before, after):
    # which failures LoRA fixed / kept / introduced, grouped by the "before" or "after" category
    fixed, kept, introduced = Counter(), Counter(), Counter()
    for b, a in zip(before, after):
        if not b["passed"] and a["passed"]:
            fixed[b["category"]] += 1
        elif not b["passed"] and not a["passed"]:
            kept[(b["category"], a["category"])] += 1
        elif b["passed"] and not a["passed"]:
            introduced[a["category"]] += 1
    return fixed, kept, introduced


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--adapter", default=None)
    p.add_argument("--prompt", choices=["basic", "detailed"], default="basic")
    p.add_argument("--test", default="data/test.jsonl")
    a = p.parse_args()

    from agent import DETAILED_PROMPT, SYSTEM_PROMPT, load_model
    model, tok = load_model(a.adapter)
    m, _ = evaluate(model, tok, load_jsonl(a.test), a.name,
                    system=DETAILED_PROMPT if a.prompt == "detailed" else SYSTEM_PROMPT)
    print(json.dumps(m, indent=1))
