# TUM UR10e Robot→Human Bowl Handover — X-Diffusion HRC (2026-08-03)

**Best working policy for the robot-moving-first handover, ROBOT-VERIFIED**: the robot
descends, grasps the bowl by the rim, lifts up-right, HOLDS until the human RIGHT hand
has actually grasped the bowl (verified frame-by-frame on deploy videos: releases only
onto a closed fist on the rim; never releases with no receiver — 0% false-release
offline AND far-receiver runs on the robot), then releases and retreats.

## THE deployment command (canonical)

```bash
cd ~/Tum_lsy_ur10e_pipeline/ur10_clearpath/Yunfei/crisp_gym
pixi run -e jazzy-lerobot python /home/admin_025/mt_pi_codebase/mt_pi/scripts/deploy_xdiffusion_ur10e.py \
  --run-dir /home/admin_025/X-Diffusion-Data/_private/runs/policy_handover_bowl_v2 \
  --ckpt latest --no-safety-observer \
  --hamer-interferer --open-horizon 8 --open-settle-cycles 2 \
  --no-home --goto-start
```

- checkpoint = **v2** (`handover-robot-first-v2-negdrop`, wandb x-diffusion-hrc)
- `--hamer-interferer`: live HaMeR RIGHT-hand receiver → 4-dim condition (camera_01)
- `--open-horizon 8`: release reads the FULL chunk (policy expresses release at steps
  4-8; first-3 latch caused 10-31s delays)
- `--open-settle-cycles 2`: release accepted only after the receiver dist is stable
  (WINDOW-SPREAD < 20px over 3 samples — per-step deltas let slow approaches through)
- `--no-home --goto-start`: skip the failure-prone joint home entirely; SAFE 3-segment
  cartesian glide (rise → traverse at z≥0.72 → descend) to the mean robot-demo start
  pose (auto-derived, cached in the run dir's `start_pose.json`)

## Data (task `handover_bowl`)

`07-29-robot-first-handover` (10 teleop eps, 848×480, real human receiver; 5 train /
demo00005 val) + `29-07-two-human-handover` (70 eps, 1280×720; LEFT hand = robot role,
RIGHT = receiver; 68 train / 2 val). Preprocessing (see `~/hamer/`):
side-locked two-hand ID (`process_two_human_handover.py` — arrival-order is WRONG here,
receiver hovers first), SAM3 bowl track (`process_bowl_sam3.py`), bowl-contact grasp
signal with motion-anchored close + ≥8-frame-loss open (`build_handover_grasp.py`,
70/70 semantics close < r_contact ≤ open), receiver emit on robot videos via
projected-TCP rejection (`process_robot_receiver_hand.py`), exports
(`export_xdiffusion_h5_handover.py`, `inject_handover_interferer_robot.py`).

## Training (what worked and what didn't — measure, don't guess)

Recipe base = HRC-avoidance winning cell; deltas: flip ×5 (transition density 7% not
1.5%), interferer-oversample ×1 (vis saturated), `vis_force_admission: false` (robot
demos carry the conditioned behavior; Gate C measured 33% release-window admission —
no forcing needed, unlike the avoidance task).

| cell | change | result |
|---|---|---|
| v1 | base | **condition IGNORED**: release 72% real = 72% sentinel. Receiver visible in ~all hold frames ⇒ zero counterfactual support |
| **v2 (DEPLOYED)** | `interferer_neg_dropout: 0.5` (closed windows → sentinel swap) | release 67% real vs **0% sentinel** — causality established; robot-verified |
| v3 | dropout on all transition-free windows | sentinel-descent hypothesis was wrong; no gain |
| v4 | flip ×5→×2 | band unchanged, release degraded (39%) |
| v5 | obs_horizon 2 | band unchanged (pipeline now SUPPORTS oh>1 though — 3 latent bugs fixed) |

**The "slow above the bowl" saga (5 hypotheses, final answer: the DATA)**: deploy
sometimes dwells at z≈0.54 before the final descent. Sentinel-thinness, flip
oversampling, mode-hedging, and phase-aliasing were each eliminated by targeted probes
(condition-swap 2×2, direction-consistency 0.99, oh=2 retrain). Ground truth: the
DEMOS meander there — net 8-frame displacement in z[0.52,0.58] is 9-18mm (demo00002:
56% of frames <10mm; human agg 35%) while path-speed looks fast (32mm/s) — align-over-
bowl wander. The policy faithfully imitates its training median. Real fixes if wanted:
recollect decisive-descent robot demos (best), or pipeline deploy inference (~1.7×
wall-clock everywhere).

## Offline gates

Gate A: open-loop pos_err 7.7mm; Gate B (deploy-equivalent): full-chunk release 67%
real vs 0% sentinel. Release-latch operating points measured per-state-group
(approach 13%/draw vs settled 82%/draw at N=8) — the basis for horizon+interlock.

## Known limits

- Occasional few-second dwell above the bowl (data-typical; see saga above).
- Close fires ~2s early in open-loop probes (32mm high) — on-robot grasp works
  (settle-then-close absorbs it); watch on new setups.
- Post-release behavior beyond retreat is data-undefined.
- Cycle time ~1.3s (0.79s inference dead time incl. 3 release draws + shared GPU with
  the SAM3 daemon) — pipelining is the known lever.
