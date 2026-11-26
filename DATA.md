# Training data

The data are not part of this repository. This file lists what the two post-training stages were trained on and how each part is built. Every public dataset is downloaded from the Hugging Face Hub by `data_prep/download_sft.py`, which also records the licence listed there on 2026-11-22.

## Public datasets

| Dataset | Used for | Licence |
|---|---|---|
| nvidia/OpenMathInstruct-2 (the 1M split) | math with step-by-step reasoning | CC BY 4.0 |
| openai/gsm8k (train) | math; also rewritten as Python-tool conversations | MIT |
| open-thoughts/OpenThoughts-114k | reasoning (math, code and other problems) | Apache 2.0 |
| HuggingFaceTB/smoltalk | everyday chat, long conversations, summaries | Apache 2.0 |
| glaiveai/glaive-function-calling-v2 | function calling | Apache 2.0 |
| NousResearch/hermes-function-calling-v1 | function calling | Apache 2.0 |
| Team-ACE/ToolACE | function calling | Apache 2.0 |
| bigcode/self-oss-instruct-sc2-exec-filter-50k | code problems | ODC-BY |
| codeparrot/apps | code problems with tests | MIT |
| microsoft/rStar-Coder (synthetic_sft) | competition-style code with verified solutions | CC BY 4.0 |
| deepmind/narrativeqa (train) | questions on long books | Apache 2.0 |
| HuggingFaceH4/MATH-500 | evaluation only | not stated on the Hub (MATH is MIT) |

HumanEval, MBPP's test split, GSM8K's test split and MATH-500 are evaluation sets. Training problems that share a 13-gram with one of them are dropped (`data_prep/decontam.py` in the base repository, used by `sft_convert.py`, `sft_code_pool.py` and `sft_code_seeds.py`).

## Teacher data

A teacher model (DeepSeek-V4.1-Flash through the DeepSeek API, `data_prep/distill_gen.py`) wrote the parts that public data do not provide:

- **Code**: solutions with a short reasoning for code problems. A solution is kept only if it passes the problem's tests in a sandbox (`data_prep/code_check.py`); for a problem without tests the teacher writes the tests too, and its solution must pass them.
- **New code problems**: written by the teacher from snippets of real Python code, then solved and checked as above.
- **Long documents**: questions and answers on 950 real documents of 70K to 110K tokens (PG19 books and long web articles); 50 more documents are held out for evaluation.
- **Thoughts for existing answers**: public tool-calling and chat conversations have no reasoning, so the teacher writes the short thought behind each reply and the replies stay as they are. Long public reasoning (OpenThoughts) is rewritten shorter, with question and answer unchanged.

## Synthetic data

- **Long inputs** (`data_prep/sft_long_synth.py`): many short documents with a question about one of them, and tables of random records with a lookup question, at lengths from a few thousand to over 100K tokens.
- **Agent data** (`data_prep/agent_pool.py`, `agent_traj.py`, `agent_env.py`): function problems with runnable examples taken from the teacher's code set; conversations in which code is written, run and fixed; and the small-project tasks (a verified solution with one injected bug, or a function to implement). Every tool result in these conversations is what the sandbox actually returned. Attempts that fail and are corrected in a later turn are shown to the model but are not training targets.

## Mixes

Shares are by tokens. Both stages also read the pretraining bins of the base repository for replay.

**SFT** (`sft_v3` in `scripts/train.py`, 1.5B tokens):

| Part | Share |
|---|---|
| Math (OpenMathInstruct-2, GSM8K) | 22% |
| Reasoning (OpenThoughts, teacher math) | 8% |
| Code in chat format (teacher solutions 7%, public solutions with a real reasoning 3%, rStar-Coder 14%) | 24% |
| Hard problems with long reasoning, packed whole | 10% |
| Chat | 4% |
| Function calling and the Python tool | 4.4% |
| General reasoning | 0.6% |
| Long instruction packs (long-document questions, multi-document and record lookups, long conversations) | 17% |
| Plain long documents (pretraining data) | 6% |
| Plain code (pretraining data) | 4% |

**Agent fine-tune** (`runs/posttrain/agent_mix.py`, 0.18B tokens): 40% agent data and 60% the SFT mix above (with packs of 64K tokens), so that what the SFT stage taught is replayed.

| Agent source | Tokens | Share |
|---|---|---|
| Small-project tasks (`sft_agent_env`) | 25.8M | 23.6% |
| Write, run, fix with the Python tool (`sft_agent_tool`) | 12.1M | 11.1% |
| Fix after an error report, no tool (`sft_agent_fix`) | 2.6M | 2.4% |
| The same tasks with no tool offered (`sft_agent_plain`) | 2.5M | 2.3% |
| A failed attempt followed by the model's own passing one (`sft_agent_self`) | 0.7M | 0.7% |
