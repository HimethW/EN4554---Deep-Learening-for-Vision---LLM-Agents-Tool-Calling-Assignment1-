# EN4554 Assignment 1 - Part 2: LLM agent tool calling

Small agent built on **Qwen2.5-0.5B-Instruct**, LoRA-finetuned to call tools, and compared against the untouched model.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HimethW/EN4554---Deep-Learening-for-Vision---LLM-Agents-Tool-Calling-Assignment1-/blob/main/tool_calling.ipynb)

## Files

| file | what it does |
|---|---|
| `tool_calling.ipynb` | runs everything: data, agent demo, baseline eval, LoRA training, LoRA eval, failure analysis |
| `tools.py` | the 5 tools (calculator, convert_units, get_weather, get_stock_price, get_events) + their JSON schemas |
| `agent.py` | agent loop: generate -> parse `<tool_call>` -> run tool -> feed result back, stops on a plain answer or after 4 steps |
| `make_data.py` | generates train / val / test from templates (seed 42) |
| `train.py` | LoRA finetuning, loss only on assistant turns |
| `evaluate.py` | metrics (call validity, call accuracy, abstention, task success) and failure categories |
| `data/` | generated datasets |
| `results/` | per-case traces and metrics for each model, summary table, loss curve |
| `writeup.md` | the 1 page write-up |

## Results (100 held-out test requests)

| | Baseline | Baseline + detailed prompt | LoRA |
|---|---|---|---|
| call validity | 85.0 | 77.5 | **100.0** |
| call accuracy | 67.5 | 57.5 | **86.2** |
| abstention | 75.0 | 80.0 | **95.0** |
| task success | 67.0 | 62.0 | **91.0** |

Details and failure analysis in [writeup.md](writeup.md). Seed 42 everywhere.

## Running

Open the notebook in Colab (T4 GPU) and run all. It clones this repo, installs the requirements and saves the
LoRA checkpoint to Google Drive. Takes about 30-40 min.

Locally (needs a CUDA GPU):

```
pip install torch -r requirements.txt
python make_data.py
python evaluate.py --name baseline
python train.py --out outputs/lora          # use --bs 1 --accum 16 on an 8GB GPU
python evaluate.py --name lora --adapter outputs/lora/final
```
