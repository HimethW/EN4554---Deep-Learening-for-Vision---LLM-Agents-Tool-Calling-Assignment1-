# EN4554 Assignment 1, Part 2: Tool Calling with LoRA

## 1. Setup

I used Qwen2.5-0.5B-Instruct, since its chat template already supports tool calls. The agent loop generates a reply, runs any
tool calls, adds the results to the conversation and repeats until the model answers in plain text or four steps have passed.
Decoding is greedy and all runs use the same system prompt.

- Tools: a calculator, a unit converter, and weather, stock and calendar lookups that return made-up data.
- Data (templates, seed 42): 1000 training, 100 validation and 100 test requests. Training has 670 single calls, 80 two-step
  requests, 50 with two lookups in one turn and 200 that need no tool. The test set (80 tool, 20 no-tool) uses different
  templates and 4 cities and 3 tickers never seen in training.
- LoRA: rank 16, alpha 32 and dropout 0.05 on all attention and MLP projections (8.8M trainable parameters, 1.75% of the
  model), 2 epochs, learning rate 2e-4 with a cosine schedule, effective batch size 16, fp16, and loss only on the assistant
  turns. Training took about 17 minutes and the validation loss went from 0.042 to 0.023.

## 2. Results

Table 1 compares the untouched model, the same model with a longer system prompt explaining when and how to call each tool,
and the LoRA model. Validity and accuracy are measured on the 80 tool requests and abstention on the 20 no-tool ones. LoRA
improves every metric, most of all on two-step requests, where the base model never made the second call.

**Table 1.** Results on the test set (%).

| | Base | Base + prompt | LoRA |
|---|---|---|---|
| Call validity | 85.0 | 77.5 | 100.0 |
| Call accuracy | 67.5 | 57.5 | 86.2 |
| Abstention | 75.0 | 80.0 | 95.0 |
| Task success | 67.0 | 62.0 | 91.0 |
| Two-step solved | 0/8 | 0/8 | 6/8 |

## 3. Failure analysis

Table 2 gives each failed case one category. LoRA fixed 25 of the 34 baseline failures, left 9 and introduced 3.

**Table 2.** Failed test cases by category.

| | Base | LoRA |
|---|---|---|
| No call when one was needed | 12 | 0 |
| Stopped before second call | 8 | 2 |
| Unnecessary call | 5 | 1 |
| Wrong tool | 3 | 3 |
| Wrong arguments | 3 | 6 |
| Wrong answer (no-tool) | 3 | 0 |
| **Total** | **34** | **12** |

- Fixed: most baseline failures were missed tool calls. For indirect questions such as "Do I need an umbrella in Kandy
  today?" the base model asked the user for the weather instead of calling the tool, and it never chained calls (for 99
  Microsoft shares it looked up the price and then guessed the total). It also called tools when none was needed and
  refused some general questions. After training the model never skipped a needed call.
- Not fixed: 3 cases come from the ticker NKE, which only appears in the test set. The model misspells it as NEK and
  makes up a price. Another 3 are "How much would one share of Apple cost?", which is close to the training template for
  buying several shares, so the model adds an unneeded calculator call. The rest are two "price for 250 x AMZN stock?"
  requests that stop after the price lookup and a question about the weekend.
- Introduced: Coca-Cola mapped to the wrong ticker, and two "add a and b, then multiply by c" requests turned into the
  wrong expression, which the base model got right.

Overall, LoRA learned the format and when to use a tool, but it is still weak on inputs unlike the templates. More varied
training phrasings would help.

## 4. Would a better prompt have been enough?

Probably not for a model this small. The detailed prompt made tool use worse (call accuracy fell from 67.5% to 57.5% and missed
calls rose from 12 to 18) and still solved none of the two-step requests. Only abstention improved. A 0.5B model does not follow
long instructions reliably, so the behaviour had to be trained in.
