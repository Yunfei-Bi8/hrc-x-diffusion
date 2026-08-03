#!/usr/bin/env bash
set -uo pipefail
cd /home/admin_025/X-Diffusion
export WANDB_API_KEY=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
P=/home/admin_025/miniconda3/envs/mtpi/bin/python
S=/tmp/claude-1000/-home-admin-025-mt-pi-codebase/78a38b32-b231-4a83-ab10-6c5d577f1100/scratchpad
echo "=== [1/3] v4 policy (flip x2) $(date)"
$P scripts/train_policy.py --config_path configs/policy_handover_bowl_v4.yaml || { echo "POLICY FAILED"; exit 1; }
echo "=== [2/3] BAND PROBE $(date)"
$P $S/band_probe.py policy_handover_bowl_v4 || echo "BAND PROBE FAILED"
echo "=== [3/3] GATES A+B $(date)"
$P $S/gates_ab_handover.py && echo "=== V4 DONE $(date)" || echo "=== V4 DONE (gate B first-3 informational) $(date)"
