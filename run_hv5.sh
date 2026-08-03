#!/usr/bin/env bash
set -uo pipefail
cd /home/admin_025/X-Diffusion
export WANDB_API_KEY=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
P=/home/admin_025/miniconda3/envs/mtpi/bin/python
echo "=== [1/1] v5 policy (obs_horizon=2) $(date)"
$P scripts/train_policy.py --config_path configs/policy_handover_bowl_v5.yaml && echo "=== V5 TRAINED $(date)" || echo "=== V5 FAILED $(date)"
