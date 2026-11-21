# Part 2 - Tool calling with LoRA

**Setup.** Model: Qwen2.5-0.5B-Instruct, using its own `<tool_call>{json}</tool_call>` chat template. Five tools:
`calculator` (safe AST eval), `convert_units` (length/mass/volume/temperature), and three lookups on made-up data so
the model can't answer from memory: `get_weather(city)`, `get_stock_price(symbol)`, `get_events(day)`. Agent loop:
generate, parse every tool call, run the tools, append the results as `tool` messages and repeat. It stops on a
plain-text reply or after 4 steps (greedy decoding, same short system prompt in every run).

**Data** (templates, seed 42). Train 1000 / val 100 / test 100. Train has 150 calculator, 150 conversion,
140 weather, 120 stock, 110 calendar, 80 two-step (weather→convert, price→calculator), 50 parallel (two lookups in
one turn) and 200 no-tool requests (chit-chat, general knowledge, including near-misses like "what is a bear
market?"). Test has 80 tool requests (14/14/14/12/12, 8 two-step, 6 parallel) and 20 no-tool ones. All test
templates are different phrasings (e.g. "Do I need an umbrella in Kandy today?"), and 4 cities and 3 tickers only appear in test.

**LoRA.** r=16, α=32, dropout 0.05 on q,k,v,o,gate,up,down (8.8M trainable params, 1.75%). 2 epochs, lr 2e-4
cosine, 5% warmup, effective batch 16, fp16 (frozen base in fp16, adapters in fp32). The loss is only on assistant tokens
(tool calls and final answers). Train loss 0.53 → 0.02, val loss 0.042 → 0.023, about 17 min of training.

| test set (100 requests) | Baseline | Baseline + detailed prompt | LoRA |
|---|---|---|---|
| call validity (80 tool requests) | 85.0 | 77.5 | **100.0** |
| call accuracy (80 tool requests) | 67.5 | 57.5 | **86.2** |
| abstention (20 no-tool requests) | 75.0 | 80.0 | **95.0** |
| task success, end to end | 67.0 | 62.0 | **91.0** |
| two-step requests solved | 0/8 | 0/8 | 6/8 |

**Failure analysis.** The baseline failed 34 cases and LoRA 12 (each case gets one category):

| category | no call | missing 2nd call | unnecessary call | wrong tool | wrong argument | wrong answer (no-tool) |
|---|---|---|---|---|---|---|
| baseline | 12 | 8 | 5 | 3 | 3 | 3 |
| LoRA | 0 | 2 | 1 | 3 | 6 | 0 |

- **Fixed by LoRA (25):**
  - **Didn't call a needed tool (9).** For indirect phrasings like "I'm flying to Tokyo tonight, what are the conditions like there?" the base model asked for information it already had ("Could you please tell me your city or zip code").
  - **Never chained two calls (6).** It solved 0/8 two-step requests and made up the second number instead (99 × $388.12 became "$3,881.20").
  - **Called a tool for no-tool questions (4).** For example it called `get_weather` for "What is the capital of Australia?".
  - **Refused general questions (3).** It said it needed a tool for questions like "Who wrote the Harry Potter books?".
  - **Wrong tool (3).** For example "where is KO trading right now" became `get_weather(city="Ko")`.
- **Not fixed (9):**
  - **Unseen ticker (3).** NKE gets mangled to "NEK"/"NEO", and the model then invents a price from the error message.
  - **Stopped after the price lookup (2).** "price for 250 x AMZN stock?" still never reaches the calculator.
  - **Made-up tool (1).** "Which days are the weekend?" calls a tool that doesn't exist.
  - **"How much would one share of Apple cost?" (3).** It moved from *no call* to an extra `calculator(1 / 214.37)`. The phrasing is close to the training template "How much would {n} shares of {t} cost?", so LoRA matched it to the two-step chain.
- **Introduced by LoRA (3):**
  - **New company name (1).** "Latest Coca-Cola stock price?" became symbol "COG"; that name only appears in the test set.
  - **"Add a and b, then multiply the result by c" (2).** It produced `a + b` or `a + b * c`. That sentence structure never appears in training, and the base model got both right.

Overall, LoRA taught the format and *when* to call (validity 100%, zero missed calls). The remaining errors are inputs
that don't look like the 1000 templated examples: new tickers, new company names and new sentence structures. It
partly overfits to surface patterns, so more varied training phrasings would be the next fix.

**Would a better prompt have been enough?** Not for this model. A detailed system prompt spelled out when to call each tool, the exact
call format and how to chain calls, and it made the 0.5B model *worse* at calling tools (accuracy 67.5 → 57.5, missed
calls 12 → 18). It still solved 0/8 two-step requests, and only abstention improved (75 → 80). A model this small doesn't reliably follow
long instructions, so the behaviour had to be trained in. A larger model or few-shot examples might close more of the gap.
