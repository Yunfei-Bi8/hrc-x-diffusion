#!/usr/bin/env bash
set -uo pipefail
cd /home/admin_025/X-Diffusion
export WANDB_API_KEY=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
P=/home/admin_025/miniconda3/envs/mtpi/bin/python
S=/tmp/claude-1000/-home-admin-025-mt-pi-codebase/78a38b32-b231-4a83-ab10-6c5d577f1100/scratchpad
echo "=== [1/2] handover policy v2 (neg-dropout 0.5; wandb: handover-robot-first-v2-negdrop) $(date)"
$P scripts/train_policy.py --config_path configs/policy_handover_bowl_v2.yaml || { echo "POLICY FAILED"; exit 1; }
echo "=== [2/2] GATES A+B $(date)"
$P $S/gates_ab_handover.py && echo "=== V2 COMPLETE — ALL GATES PASSED $(date)" || echo "=== V2 COMPLETE — GATE A/B FAILED $(date)"
