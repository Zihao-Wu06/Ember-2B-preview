#!/bin/bash
# The agent fine-tune that produced Ember-2B-preview: the small-project data, the data mix, 350 steps from the SFT
# model on one RTX 4090, then the evaluations. Each phase leaves a marker in runs/posttrain/agent_finetune/, so the
# script can be started again after a failure. It expects (README, "Reproducing"):
#   data/bins_sft          the SFT sources (data_prep/sft_tokenize.py, sft_long_packs.py --pack-len 65536)
#   data/bins_main         the pretraining bins of the base repository (replay of plain code and long documents)
#   data/agent/pool_v2.jsonl, attempts.jsonl, attempts_samples.jsonl     data_prep/agent_pool.py, runs/mac_eval/agent_gen.py
#   data/bins_sft/sft_agent_tool, sft_agent_fix                          data_prep/agent_traj.py + sft_tokenize.py
#   ckpt/posttrain_sft/init_for_4090.pt                                  runs/posttrain/slim_for_4090.py on the SFT checkpoint
cd "$(dirname "$0")/../.."
N=runs/posttrain/agent_finetune
O=runs/eval
NEW=runs/mac_eval/agent
mkdir -p $N $O
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
phase() {
  local name=$1; shift
  if [ -f $N/$name.done ]; then echo "== $name: done before"; return 0; fi
  echo "== $name $(date '+%F %T')"
  if "$@"; then touch $N/$name.done; echo "== $name finished $(date '+%F %T')"
  else echo "== $name FAILED $(date '+%F %T')"; exit 1; fi
}
chat() {
  local name=$1; shift
  python runs/mac_eval/chat_eval.py --export $NEW --device cuda --moe-impl bmm "$@" --out $O/$name.json > $O/$name.log 2>&1 || return 1
  echo "   $name: $(grep -a '^{' $O/$name.log | tr '\n' ' ' | cut -c1-700)"
}
data() {
  # bug fixes (one injected bug), two-function bug fixes, a wrong first fix that is undone, a first edit that quotes
  # text not in the file, and implement tasks; every tool result in the data is what the sandbox returned
  python data_prep/agent_env.py train --bugfix 9000 --bugfix2 2500 --bugfix-retry 6000 --bugfix-editfail 3000 --implement 7200 \
    --workers 6 > $N/envdata.log 2>&1 || return 1
  python data_prep/sft_tokenize.py --only sft_agent_env sft_agent_self sft_agent_plain > $N/tokenize.log 2>&1 || return 1
  python data_prep/agent_env.py eval > $N/envtasks.log 2>&1   # data/agent/env_eval.jsonl: HumanEval and MBPP's test split
}
train() {
  local mix
  # every agent source is read twice in 350 steps, the agent sources together at most 40%; the rest is the SFT mix
  mix=$(python runs/posttrain/agent_mix.py --steps 350 sft_agent_tool sft_agent_fix sft_agent_env sft_agent_self sft_agent_plain) || return 1
  echo "   mix: $mix"
  OUT=runs/posttrain/agent CK=ckpt/posttrain_agent EXPORT=$NEW STEPS=350 MIX="$mix" \
    bash runs/posttrain/agent_train.sh > $N/train.log 2>&1
  grep -aq "AGENT_TRAIN_DONE" $N/train.log
}
evaluate() {
  python runs/mac_eval/agent_env_eval.py --export $NEW --device cuda --limit 120 --turns 10 --max-rows 96 --rows-at-32k 7 \
    --out $O/agent_env.json > $O/agent_env.log 2>&1 || return 1
  chat short --tasks gsm8k,math500,humaneval,mbpp,tools || return 1
  chat passkey_4k_32k --tasks passkey --lengths 4096,32768 --n-long 4 || return 1
  chat passkey_128k --tasks passkey --lengths 131072 --n-long 6 || return 1
  chat longdoc --tasks longdoc --limit 6 --per-doc 2
}
phase data data
phase train train
phase eval evaluate
echo "AGENT_FINETUNE_DONE $(date '+%F %T')"
