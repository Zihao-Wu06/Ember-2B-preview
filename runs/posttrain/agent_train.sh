#!/bin/bash
# Small-agent fine-tuning on one RTX 4090: from the SFT model of the B300 run, a mix of the multi-step agent
# conversations and the SFT mix as replay (MIX: a name in scripts/train.py MIXES, or the JSON runs/posttrain/agent_mix.py
# prints), 64K windows, 4K rows for the short sources, full recomputation, a lower learning rate than the SFT run's 1e-4
# because the model is already trained. Then the inference weights are exported.
#   bash runs/posttrain/agent_train.sh      (STEPS, LR, MIX, OUT, CK, EXPORT, INIT override)
cd "$(dirname "$0")/../.."
OUT=${OUT:-runs/posttrain/agent}
CK=${CK:-ckpt/posttrain_agent}
INIT=${INIT:-ckpt/posttrain_sft/init_for_4090.pt}
EXPORT=${EXPORT:-runs/mac_eval/agent}
MIX=${MIX:-agent_v1_64k}
STEPS=${STEPS:-300}            # optimizer steps of 524,288 tokens: 157M tokens, about 1.9 hours
LR=${LR:-5e-5}
WARMUP=${WARMUP:-10}
CKPT_EVERY=${CKPT_EVERY:-100}  # also the interval of the validation losses
[ -f "$INIT" ] || { echo "missing $INIT"; exit 1; }
mkdir -p "$OUT"
OV='{"moe_impl": "triton", "ce_impl": "fused", "attnres_impl": "fused", "ce_blas": "cublaslt", "kda_keep_layers": 12, "conv_impl": "fla_gluon", "moe_shared_fused": "router_lean", "moe_sort": "counting", "attn_logn": 4096, "max_seq_len": 65536}'
echo "$(date +%T) train: $STEPS steps of 524,288 tokens from $INIT, mix $MIX, learning rate $LR"
python scripts/supervise.py --out $OUT --max-restarts 2 -- python scripts/train.py --preset main_2b \
  --data data/bins_sft,data/bins_main --out $OUT --ckpt-dir $CK --ckpt-every $CKPT_EVERY --backup-every 0 --steps $STEPS \
  --seq-len 65536 --short-seq-len 4096 --short-micro-bsz 2 --short-ckpt-frac 0 --micro-bsz 1 --global-tokens 524288 \
  --lr $LR --warmup $WARMUP --decay-frac 0.3 --eval-every $CKPT_EVERY --grad-ckpt --ckpt-frac 1.0 --offload-master --fused-grad-acc \
  --muon-impl batched_fused --override "$OV" --engram-early --state-min-bytes 1048576 --engram-native --async-ckpt \
  --offload-opt --fuse --muon-bf16 --mix $MIX --init-from $INIT --init-no-opt
rc=$?; echo "$(date +%T) train: exit $rc"
grep -aq "^TRAIN_DONE" $OUT/stdout.log || { echo "training did not finish"; exit 1; }
LAST=$(ls $CK/ckpt_*.pt 2>/dev/null | sort | tail -1)
[ -n "$LAST" ] || { echo "no checkpoint was written"; exit 1; }
echo "$(date +%T) export $LAST"
CUDA_VISIBLE_DEVICES= python runs/mac_eval/export_weights.py --ckpt "$LAST" --out $EXPORT
rc=$?; echo "$(date +%T) export: exit $rc"
[ $rc = 0 ] && echo "AGENT_TRAIN_DONE $(date +%T)"
