#!/usr/bin/env bash
# Train the three green_lego policy cells sequentially (one GPU).
# robot_only (5-demo baseline) -> naive (all human mixed in) -> xdiffusion (classifier-gated).
# Continue past individual failures so one bad cell doesn't block the comparison.
cd /home/admin_025/X-Diffusion
P=/home/admin_025/miniconda3/envs/mtpi/bin/python
for mode in robot_only naive xdiffusion; do
  echo "=== [$mode] start $(date)"
  $P scripts/train_policy.py --config_path configs/policy_${mode}_green_lego.yaml \
      > policy_${mode}_green_full.log 2>&1 \
      && echo "=== [$mode] DONE $(date)" \
      || echo "=== [$mode] FAILED $(date) (see policy_${mode}_green_full.log)"
done
echo "=== ALL CELLS FINISHED $(date)"
