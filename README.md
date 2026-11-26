# Ember-2B-preview

Ember-2B-preview is the post-trained version of [Ember-2B-base](https://github.com/Zihao-Wu06/Ember-2B-base), a 1.97B-parameter MoE language model (0.39B parameters active per token, 128K context) that I pretrained from scratch on only 25B tokens with a single RTX 4090. This repository holds the post-training code only: the chat format, the data pipeline, the fine-tuning recipes, the evaluation scripts and the training records. The architecture, the pretraining code and the pretraining results are in the base repository.

Post-training had two stages:

1. **SFT.** Chat format with a thinking block in every assistant turn and Hermes-style tool calls. 1.5B tokens at 128K context on one rented B300, starting from the base model's final 128K checkpoint.
2. **Agent fine-tune.** 0.18B tokens at 64K context on one RTX 4090, starting from the SFT model. It teaches the model to work in a small project over several turns: run the tests, read a file, edit it, run the tests again.

A third stage, reinforcement learning (GRPO with answer-match rewards for math and unit-test rewards for code), was planned. **It was not run: I did not have enough compute for it.** The released model is the result of the two supervised stages.

Weights: [Liquor-06/Ember-2B-preview](https://huggingface.co/Liquor-06/Ember-2B-preview) on Hugging Face.

## Results

All numbers are for the released model, greedy decoding, chat format with thinking on. Only the final answer is scored, never the thinking.

| Task | Size | Score |
|---|---|---|
| GSM8K | 1,319 | 34.8 |
| MATH-500 | 500 | 21.0 |
| HumanEval | 164 | 26.2 |
| MBPP (test) | 500 | 31.2 |
| Function calling: right function / right arguments | 210 calls | 81.9 / 68.1 |
| Function calling: no call when none is needed | 90 | 54.4 |
| Agent tasks in a small project (below) | 360 | 29.4 |
| Long-document questions, documents of 74K to 85K tokens (token F1) | 12 | 32.1 |
| Passkey retrieval at 4K and 32K | 8 | 8 / 8 |

**Agent tasks.** The model gets `solution.py` and `test_solution.py`, four tools (`run_tests`, `read_file`, `edit_file`, `write_file`) and up to 10 turns. A task counts as done when the tests pass on the files as the model left them. The 360 tasks are 120 bug fixes built from MBPP's test split (23.3% done), 120 bug fixes built from HumanEval (25.0%) and 120 implement tasks from MBPP's test split (40.0%; 33.3% also pass the tests the model did not see). None of these problems is in the training data. The SFT model before the agent fine-tune finishes none of the 360.

The raw results are in `runs/eval/`.

## Compared with models pretrained on far more data

With 25B tokens of pretraining, about 1% of what the models below were trained on, Ember-2B-preview scores higher than both on math and code:

| Model | Pretraining tokens | Context | GSM8K | HumanEval | MBPP |
|---|---|---|---|---|---|
| **Ember-2B-preview** | **25B** | **128K** | **34.8** | **26.2** | **31.2** |
| Gemma-2B | 3T | 8K | 17.7 | 22.0 | 29.2 |
| Llama-2-7B | 2T | 4K | 14.6 | 12.8 | 20.8 |

The Gemma and Llama 2 numbers are the ones published in their papers ([Gemma](https://arxiv.org/abs/2403.08295), Sections 2 and 4.1, Table 6; [Llama 2](https://arxiv.org/abs/2307.09288), Tables 1, 21 and 25).

## Usage

The code in this repository is used on top of the base repository. Copy it over a checkout of the base (two files, `scripts/train.py` and `moe_llm/data.py`, replace the base versions: they add the SFT data mixes and the loss mask):

```bash
git clone https://github.com/Zihao-Wu06/Ember-2B-base
git clone https://github.com/Zihao-Wu06/Ember-2B-preview
rsync -a --exclude .git --exclude README.md --exclude LICENSE Ember-2B-preview/ Ember-2B-base/
pip install torch safetensors tokenizers huggingface_hub
```

Chat with the model (custom architecture, so it is loaded with this code and not with `transformers`):

```python
import json, os, sys
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from tokenizers import Tokenizer

sys.path[:0] = ["Ember-2B-base", "Ember-2B-base/runs/mac_eval"]
import decode, mac_lm
from moe_llm import chat
from moe_llm.config import ModelConfig
from moe_llm.model import MoELLM

path = snapshot_download("Liquor-06/Ember-2B-preview")
cfg = {k: v for k, v in json.load(open(f"{path}/config.json")).items() if k in ModelConfig.__dataclass_fields__}
model = MoELLM(ModelConfig.from_dict(cfg).replace(**mac_lm.CPU_OVERRIDES))  # pure PyTorch, runs on CPU
for f in sorted(os.listdir(path)):
    if f.endswith(".safetensors"):
        model.load_state_dict(load_file(f"{path}/{f}"), strict=False)
model.eval()
tok = Tokenizer.from_file(f"{path}/tokenizer.json")

conv = {"messages": [{"role": "user", "content": "A train travels 180 km in 2.5 hours. What is its average speed?"}]}
prompt = [chat.SPECIAL_IDS[chat.EOS]] + tok.encode(chat.render(conv, generation_prompt=True)).ids
out = decode.generate_ids(model, prompt, max_new=512, eos=chat.SPECIAL_IDS[chat.IM_END])
reply = chat.parse_assistant(tok.decode(out, skip_special_tokens=False))
print(reply["think"])    # the reasoning
print(reply["content"])  # the answer
```

A conversation is `{"messages": [...], "tools": [...]}`. Messages have the roles `system`, `user`, `assistant` and `tool`; an assistant message may carry `think` and `tool_calls`. To offer tools, pass a list of JSON function descriptions as `tools`; `parse_assistant` returns the calls the model made in `reply["tool_calls"]`, and their results go back as `tool` messages. `moe_llm/chat.py` documents the format. Load one model per Python process.

## Reproducing

Every step is a script in this repository. The data are not redistributed; `DATA.md` lists the sources and their licences. The teacher data came from a paid API and the agent data depend on the model's own attempts, so a rerun gives a model of the same kind, not the same weights.

1. **Pretraining bins and the base checkpoint**: from the base repository. The SFT mix replays plain code and long documents from the pretraining bins (`data/bins_main`).
2. **Public datasets**: `data_prep/download_sft.py`, then `data_prep/sft_convert.py` (one conversation format for all of them, with decontamination against the evaluation sets) and `data_prep/sft_select.py`.
3. **Teacher data** (DeepSeek API, `data_prep/distill_gen.py`): solutions to code problems, kept only if they pass the tests (`code_check.py`); new code problems written from real code snippets (`sft_code_seeds.py`, `distill_codeprob.py`, `sft_code_pool.py`); questions on real long documents (`sft_longdocs.py`); short thoughts for public tool-calling and chat conversations that have none, and shorter rewrites of long public reasoning (`distill_think.py`). `distill_convert.py` turns the results into conversations.
4. **Synthetic long inputs** without the API: `data_prep/sft_long_synth.py`. Held-out sets: `data_prep/sft_eval_sets.py`.
5. **Tokenize and pack**: `data_prep/sft_tokenize.py`, then `data_prep/sft_long_packs.py` (packs of whole conversations for the long windows; `--pack-len 65536 --name sft_long64k` for the 4090).
6. **Starting weights**: the base model never saw the chat tokens, so their output rows are fitted first (`runs/posttrain/probe_special_rows.py`, `runs/posttrain/cloud/make_init.py`; `runs/posttrain/special_init.py` explains why).
7. **SFT**: `runs/posttrain_sft/command.txt` (mix `sft_v3` in `scripts/train.py`).
8. **Agent data**: a pool of function problems with runnable examples (`data_prep/agent_pool.py`), the SFT model's own attempts at them (`runs/mac_eval/agent_gen.py`), multi-step conversations built from real executions (`data_prep/agent_traj.py`; a first fine-tune on these was used to collect runs in which the model fixes its own error, `runs/mac_eval/agent_gen2.py`, and they were added), and the small-project environment (`data_prep/agent_env.py`).
9. **Agent fine-tune and evaluation**: `runs/posttrain/agent_finetune.sh` (`runs/posttrain_agent/command.txt` is the resulting launch command).

Wrong turns that a later turn corrects stay in the training conversations as context but are not training targets (`no_loss`; the loader's mask in `moe_llm/data.py`).

## Layout

- `moe_llm/chat.py`: the chat format, rendering and parsing
- `moe_llm/data.py`, `scripts/train.py`: the base versions plus the loss mask and the SFT and agent data mixes
- `data_prep/`: downloads, conversion, teacher distillation, code checking, tokenization, long packs, agent data
- `runs/posttrain/`: starting weights, the agent fine-tune scripts and its data mix rule
- `runs/mac_eval/`: chat-format evaluation (`chat_eval.py`, `chat_score.py`) and the agent evaluations
- `runs/posttrain_sft/`, `runs/posttrain_agent/`: model config, launch command and training log of the two stages
- `runs/eval/`: evaluation results of the released model
- `scripts/to_safetensors.py`: exported weights to the files on the Hub
- `tests/`: unit tests of the above (run them in the merged checkout: `python -m pytest -q tests`)

## License

Apache 2.0. The training data have their own licences (`DATA.md`).
