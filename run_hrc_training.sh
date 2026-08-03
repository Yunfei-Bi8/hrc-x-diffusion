#!/usr/bin/env bash
set -uo pipefail
cd /home/admin_025/X-Diffusion
export WANDB_API_KEY=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
P=/home/admin_025/miniconda3/envs/mtpi/bin/python
S=/tmp/claude-1000/-home-admin-025-mt-pi-codebase/78a38b32-b231-4a83-ab10-6c5d577f1100/scratchpad
echo "=== NOTE: classifier already trained; Gate C measured 14.1% mean admission (marginal)"
echo "=== -> vis-forced admission patch active in train_policy.py (graspall grasp-dim precedent)"
echo "=== [1/3] HRC policy (wandb: 1st-hrc-robot-avoid-human) $(date)"
$P scripts/train_policy.py --config_path configs/policy_hrc_green.yaml || { echo "POLICY FAILED"; exit 3; }
echo "=== [2/3] ablation cell (no-cond) $(date)"
$P scripts/train_policy.py --config_path configs/policy_hrc_ablation_green.yaml || echo "ABLATION FAILED (non-blocking)"
echo "=== [3/3] GATES A+B $(date)"
$P $S/gates_ab_hrc.py && echo "=== HRC CHAIN COMPLETE — ALL GATES PASSED $(date)" || echo "=== HRC CHAIN COMPLETE — GATE A/B FAILED, see above $(date)"
