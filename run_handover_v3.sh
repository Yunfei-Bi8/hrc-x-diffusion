#!/usr/bin/env bash
set -uo pipefail
cd /home/admin_025/X-Diffusion
export WANDB_API_KEY=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
P=/home/admin_025/miniconda3/envs/mtpi/bin/python
S=/tmp/claude-1000/-home-admin-025-mt-pi-codebase/78a38b32-b231-4a83-ab10-6c5d577f1100/scratchpad
echo "=== [1/2] handover policy v3 (transition-free neg-dropout; wandb: handover-robot-first-v3) $(date)"
$P scripts/train_policy.py --config_path configs/policy_handover_bowl_v3.yaml || { echo "POLICY FAILED"; exit 1; }
echo "=== [2/2] GATES A+B $(date)"
$P $S/gates_ab_handover.py && echo "=== V3 DONE — ALL GATES PASSED $(date)" || echo "=== V3 DONE — GATE A/B FAILED $(date)"
