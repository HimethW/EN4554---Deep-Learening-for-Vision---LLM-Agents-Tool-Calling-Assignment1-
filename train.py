import argparse
import json
import math
import os

import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed

from agent import MODEL_ID, SYSTEM_PROMPT
from tools import TOOL_SCHEMAS


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def to_messages(row, system=SYSTEM_PROMPT):
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": row["query"]}]
    for step in row["steps"]:
        msgs.append({"role": "assistant", "content": "", "tool_calls": [
            {"type": "function", "function": {"name": c["name"], "arguments": c["arguments"]}} for c in step]})
        for c in step:
            msgs.append({"role": "tool", "name": c["name"], "content": json.dumps(c["result"])})
    msgs.append({"role": "assistant", "content": row["answer"]})
    return msgs


# tokenize turn by turn so only the assistant tokens get labels (everything else is -100)
def encode(row, tok, max_len=1024):
    msgs = to_messages(row)
    ids, labels, done = [], [], ""
    for i, m in enumerate(msgs):
        if m["role"] != "assistant":
            continue
        prompt = tok.apply_chat_template(msgs[:i], tools=TOOL_SCHEMAS, add_generation_prompt=True, tokenize=False)
        full = tok.apply_chat_template(msgs[:i + 1], tools=TOOL_SCHEMAS, tokenize=False)
        assert prompt.startswith(done) and full.startswith(prompt)
        reply = full[len(prompt):]
        tail = "\n" if reply.endswith("\n") else ""
        ctx_ids = tok(prompt[len(done):], add_special_tokens=False)["input_ids"]
        reply_ids = tok(reply[:len(reply) - len(tail)], add_special_tokens=False)["input_ids"]
        tail_ids = tok(tail, add_special_tokens=False)["input_ids"] if tail else []
        ids += ctx_ids + reply_ids + tail_ids
        labels += [-100] * len(ctx_ids) + reply_ids + [-100] * len(tail_ids)
        done = full
    return {"input_ids": ids[:max_len], "labels": labels[:max_len]}


class ChatData(Dataset):
    def __init__(self, rows, tok):
        self.items = [encode(r, tok) for r in rows]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def make_collator(pad_id):
    def collate(batch):
        n = max(len(x["input_ids"]) for x in batch)
        ids = torch.full((len(batch), n), pad_id)
        labels = torch.full((len(batch), n), -100)
        mask = torch.zeros((len(batch), n), dtype=torch.long)
        for i, x in enumerate(batch):
            k = len(x["input_ids"])
            ids[i, :k] = torch.tensor(x["input_ids"])
            labels[i, :k] = torch.tensor(x["labels"])
            mask[i, :k] = 1
        return {"input_ids": ids, "labels": labels, "attention_mask": mask}
    return collate


def train(out_dir="outputs/lora", epochs=2, lr=2e-4, rank=16, alpha=32, batch_size=2, grad_accum=8, seed=42,
          train_file="data/train.jsonl", val_file="data/val.jsonl"):
    set_seed(seed)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    # frozen base in fp16, the LoRA weights get kept in fp32 by peft
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float16).to("cuda")

    config = LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=0.05, task_type="CAUSAL_LM",
                        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    train_ds = ChatData(load_jsonl(train_file), tok)
    val_ds = ChatData(load_jsonl(val_file), tok)
    lens = [len(x["input_ids"]) for x in train_ds.items]
    print(f"train {len(train_ds)} / val {len(val_ds)} examples, tokens per example: "
          f"mean {sum(lens) / len(lens):.0f}, max {max(lens)}")

    total_steps = math.ceil(len(train_ds) / (batch_size * grad_accum)) * epochs
    args = TrainingArguments(
        output_dir=out_dir, num_train_epochs=epochs, learning_rate=lr, lr_scheduler_type="cosine",
        warmup_steps=int(0.05 * total_steps),
        per_device_train_batch_size=batch_size, per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum, fp16=True, logging_steps=5, eval_strategy="epoch",
        save_strategy="epoch", save_total_limit=1, report_to="none", seed=seed, remove_unused_columns=False,
    )
    trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds,
                      data_collator=make_collator(tok.pad_token_id))
    trainer.train()

    final = os.path.join(out_dir, "final")
    model.save_pretrained(final)
    with open(os.path.join(out_dir, "log_history.json"), "w") as f:
        json.dump(trainer.state.log_history, f, indent=1)
    return final, trainer.state.log_history


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="outputs/lora")
    p.add_argument("--epochs", type=float, default=2)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--bs", type=int, default=2)
    p.add_argument("--accum", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    train(a.out, a.epochs, a.lr, a.rank, 2 * a.rank, a.bs, a.accum, a.seed)
