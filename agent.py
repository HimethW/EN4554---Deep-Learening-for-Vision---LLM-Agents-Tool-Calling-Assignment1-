import json
import re
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from tools import FUNCTIONS, TOOL_SCHEMAS, run_tool

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

SYSTEM_PROMPT = "You are a helpful assistant. Use the tools when a request needs them, otherwise just answer."

# longer prompt used for the "better prompt only" comparison
DETAILED_PROMPT = """You are a helpful assistant with access to tools.
Rules:
- Call a tool for: arithmetic (calculator), unit conversions (convert_units), current weather (get_weather), stock prices (get_stock_price) and the user's calendar (get_events). Never guess these values.
- For anything else (general knowledge, explanations, chit-chat, writing) answer directly without calling a tool.
- A tool call must be exactly: <tool_call>
{"name": "<tool name>", "arguments": {...}}
</tool_call>
- Weather is returned in celsius. If the user wants another unit, call convert_units on the temperature.
- To price several shares, get the price first, then use the calculator.
- After you receive a tool result, reply to the user in one short sentence that uses the numbers from the result."""

TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.S)
BARE_CALL_RE = re.compile(r'\{\s*"name"\s*:\s*"(%s)"' % "|".join(FUNCTIONS))


def load_model(adapter=None, dtype=torch.float16):
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=dtype).to("cuda")
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    model.eval()
    return model, tok


def parse_tool_calls(text):
    blocks = TOOL_CALL_RE.findall(text)
    calls, broken = [], 0
    for b in blocks:
        try:
            obj = json.loads(b.strip())
            args = obj.get("arguments", {})
            if isinstance(args, str):
                args = json.loads(args)
            if not isinstance(obj["name"], str) or not isinstance(args, dict):
                raise ValueError
            calls.append({"name": obj["name"], "arguments": args})
        except Exception:
            broken += 1
    unclosed = text.count("<tool_call>") > len(blocks)
    bare = not blocks and bool(BARE_CALL_RE.search(text))
    attempted = bool(blocks) or unclosed or bare
    valid = attempted and broken == 0 and not unclosed and not bare
    return {"attempted": attempted, "valid": valid, "calls": calls}


def render(tok, messages):
    return tok.apply_chat_template(messages, tools=TOOL_SCHEMAS, add_generation_prompt=True, tokenize=False)


@torch.no_grad()
def generate(model, tok, prompts, max_new_tokens=256, batch_size=16):
    outs = []
    for i in range(0, len(prompts), batch_size):
        enc = tok(prompts[i:i + batch_size], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                             temperature=None, top_p=None, top_k=None, pad_token_id=tok.pad_token_id)
        for seq in tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=False):
            outs.append(seq.split("<|im_end|>")[0].split("<|endoftext|>")[0].strip())
    return outs


# the agent loop: generate -> parse -> run tools -> feed results back, until a plain answer
def run_agent(model, tok, queries, system=SYSTEM_PROMPT, max_steps=4, batch_size=16):
    convs = [[{"role": "system", "content": system}, {"role": "user", "content": q}] for q in queries]
    traces = [{"query": q, "steps": [], "final": None} for q in queries]
    active = list(range(len(queries)))
    for _ in range(max_steps):
        if not active:
            break
        outs = generate(model, tok, [render(tok, convs[i]) for i in active], batch_size=batch_size)
        still = []
        for i, out in zip(active, outs):
            p = parse_tool_calls(out)
            step = {"output": out, "attempted": p["attempted"], "valid": p["valid"], "calls": []}
            traces[i]["steps"].append(step)
            if not p["calls"]:
                traces[i]["final"] = out
                continue
            text = TOOL_CALL_RE.sub("", out).strip()
            convs[i].append({"role": "assistant", "content": text, "tool_calls": [
                {"type": "function", "function": c} for c in p["calls"]]})
            for c in p["calls"]:
                result = run_tool(c["name"], c["arguments"])
                step["calls"].append({**c, "result": result})
                convs[i].append({"role": "tool", "name": c["name"], "content": json.dumps(result)})
            still.append(i)
        active = still
    return traces


if __name__ == "__main__":
    # quick manual test: python agent.py "what's the weather in oslo?" [adapter_dir]
    model, tok = load_model(sys.argv[2] if len(sys.argv) > 2 else None)
    t = run_agent(model, tok, [sys.argv[1]])[0]
    for s in t["steps"]:
        print(">>", s["output"])
        for c in s["calls"]:
            print("   result:", c["result"])
    print("final:", t["final"])
