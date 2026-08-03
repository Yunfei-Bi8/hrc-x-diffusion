#!/usr/bin/env bash
set -uo pipefail
cd /home/admin_025/X-Diffusion
export WANDB_API_KEY=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
P=/home/admin_025/miniconda3/envs/mtpi/bin/python
S=/tmp/claude-1000/-home-admin-025-mt-pi-codebase/78a38b32-b231-4a83-ab10-6c5d577f1100/scratchpad
echo "=== [1/5] classifier (7-dim) $(date)"
$P scripts/train_classifier.py --config_path configs/classifier_handover_bowl.yaml || { echo "CLASSIFIER FAILED"; exit 1; }
echo "=== [2/5] GATE C: release-window admission $(date)"
$P $S/gate_c_handover.py || { echo "GATE C FAILED — enable vis_force_admission fallback and rerun"; exit 2; }
echo "=== [3/5] handover policy (wandb: 1st-handover-bowl) $(date)"
$P scripts/train_policy.py --config_path configs/policy_handover_bowl.yaml || { echo "POLICY FAILED"; exit 3; }
echo "=== [4/5] ablation (no-cond) $(date)"
$P scripts/train_policy.py --config_path configs/policy_handover_ablation.yaml || echo "ABLATION FAILED (non-blocking)"
echo "=== [5/5] GATES A+B $(date)"
$P $S/gates_ab_handover.py && echo "=== HANDOVER CHAIN COMPLETE — ALL GATES PASSED $(date)" || echo "=== HANDOVER CHAIN COMPLETE — GATE A/B FAILED $(date)"
