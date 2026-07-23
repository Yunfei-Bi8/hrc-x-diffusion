# TUM UR10e Cross-Embodiment Few-Shot Success — X-Diffusion `green_lego` (2026-07-23)

**First fully successful cross-embodiment few-shot deployment on our rig**: ~100% success
on the FULL task cycle — descend → grasp at z≈0.44 → push right → **release (re-open,
robot-verified 2026-07-23)** — with a policy trained on **5 robot demos + 100 human
demos**. Winning cell: `policy_xdiffusion_graspall_flip_green` ("experiment 3").

## 1. Data

| source | episodes | notes |
|---|---|---|
| robot | `06-20-teleop` ep0–4 train, ep5 val | UR10e + 2F-85 teleop, green lego push-right |
| human | `25-06-single-human` all 100 (98 train / 2 val) | single LEFT hand, same task/table |

H5 contract (`X-Diffusion-Data/retargeted/green_lego/{robot,human}/demoNNNNN.h5`):
`ee_pos (T,3)`, `ee_euler (T,3) 'xyz'`, `gripper_open (T,) {0,1}`, `3d_tracks (T,5,3)`
— absolute robot-base-frame poses. Grasp points agree across embodiments to 2mm
(robot z=0.440±3mm, human z=0.443±4mm). 98/100 human demos contain the release
(re-open) event; all 6 robot demos do (median 20 frames before episode end).

### Robot export — `mt_pi/scripts/export_xdiffusion_h5_robot.py`
Recorded `robotiq_tcp` cartesian (rotvec→euler) + 0613 hand-eye calib (same frame as the
human triangulation) + 2F-85 keypoint template + leading/trailing OOB trim; gripper_open
= physical opening ≥ 50mm.

### Human export — `~/hamer/export_xdiffusion_h5_human.py`
SAM3 "hand" track → HaMeR(forced LEFT) 2D keypoints (2 cams) → triangulation (0613 calib)
→ fingertip cross-camera disagreement filter (15px) → explicit gripper retarget
(**anchor = thumb/index fingertip midpoint** = paper's grasp point; real wrist tilt KEPT)
→ SAM3 object-contact gripper (open iff either cam, 5-frame confirm) →
**temporal speed alignment S=2.021** (median 16-frame ee-displacement ratio human/robot;
matches the validated 2D C-align ×1.889 independently). Release signal = SAM3 contact
ending when the hand lets go (98/100 demos).

## 2. Training (two stages, env `mtpi` + pip pyarrow)

**Stage 1 — classifier** (`configs/classifier_green_lego.yaml`, ~25min):
HalfUnet1D on (noised action 8×7, noised state 7, timestep), BCE, 50/50 balanced,
bs256 × 60ep × 100 steps. Noise-sweep sanity: human P(robot) 0.00@t0 → 0.49@t100 where
it meets the robot curve at 0.494 ≈ the theoretical 0.5 merge. **State-only — no images
(per released code AND paper's own stub note; the discriminative signal is action dynamics).**

**Stage 2 — policy, the WINNING recipe (experiment 3)**
(`configs/policy_xdiffusion_graspall_flip_green_lego.yaml`, ~17min):
- action = state_cond = 7-dim absolute `[ee_pos, ee_euler, gripper]`; obs_h 1 / pred_h 8;
  ConditionalUnet1D; DDIM 101 train / 16 infer; min-max norm over merged pool.
- masked denoising: robot rows ALWAYS in loss; human rows only when the FROZEN classifier
  says P(robot)>0.5 at that sampled noise step (per-sample Monte-Carlo of the paper's
  k≥k\*; admission ≈22%).
- **grasp dim: human rows DO supervise it** (`grasp_dim_robot_only: false`) — same joint
  supervision as position dims, classifier-gated. The release transitions are supervised
  the same way (both embodiments; flip oversampling boosts them too).
- **flip-window oversampling ×15** (`flip_oversample_factor: 15`, our addition to
  `dataset_utils/h5_dataset.py`): windows whose action grasp dim contains ANY open↔close
  transition get 15× sampler weight — the grasp-initiation events are ~1.5% of windows
  and get diluted into a copy-through shortcut otherwise.
- 50/50 human/robot batch sampling (paper Eq.4 = unweighted sum of per-dataset
  expectations; also their released config), Adam 1e-4, bs128 × 200ep × 300 steps.

Repo patches (all `PATCH (2026-07-2x)`-tagged, git-diffable): dead `baselines` import
stubs; per-embodiment demo-id split (`human_train/val_demo_ids`); Checkpointer in-flight
pruning (the release saved 630MB/epoch with pruning commented out → filled the disk);
per-dim loss masks; `flip_oversample_factor`.

## 3. Deployment — `mt_pi/scripts/deploy_xdiffusion_ur10e.py`

```bash
cd ~/Tum_lsy_ur10e_pipeline/ur10_clearpath/Yunfei/crisp_gym
pixi run -e jazzy-lerobot python /home/admin_025/mt_pi_codebase/mt_pi/scripts/deploy_xdiffusion_ur10e.py \
  --run-dir /home/admin_025/X-Diffusion-Data/_private/runs/policy_xdiffusion_graspall_flip_green \
  --ckpt latest --no-safety-observer
```

Architecture (each element earned by a measured failure):
- **inference worker process, spawned BEFORE rclpy** (in-process = GIL starvation by ~270
  executor threads; 35ms round-trip vs multi-second).
- **semantic anchor**: the policy is conditioned on the rollout of its own predicted chunk
  ends, NOT the measured pose (measured-state feedback + ~50-60mm impedance gravity droop
  = a stall fixed point).
- **integral droop compensation**: command = semantic target + comp, comp += 0.5·(desired −
  measured), clip ±12cm (training actions are MEASURED-pose trajectories; the controller
  needs targets ~60mm lower to realize them). MAX_LEAD 15cm vs measured; workspace box.
- **grasp latch, direction-specific over the first 3 planned steps** (final, robot-verified):
  - CLOSE (currently open): `min(first 3) ≤ 0.5` — "the plan closes soon". Then settle
    0.8s (stop-then-close like the teleop demos) before gripping.
  - OPEN (currently closed): **median over 3 DDIM draws** of `max(first 3) > 0.65`,
    single vote. At the release boundary the draws are bimodal (1.0/0.0 flicker ≈
    P(release)); consecutive-vote rules keep resetting on the zeros and delayed the open
    by ~8s. Median-of-3 (35ms/draw) averages the sampling noise inside one cycle; during
    transport P≈0.02 so the median ≈0 — more robust AND faster.
  - Failed latch designs, kept for the record: mean-of-2 ("open now, close next" averages
    to exactly 0.5 → threshold flicker, never fires); single min for both directions
    (pins at 0 while holding → release structurally blocked); consecutive-2 votes on raw
    draws (reset-on-flicker → ~8s release delay).
- euler_x +2π canonicalization at Rx≈π; yaw lock; per-run recording
  (`deploy_logs/xdiff_*/commands.pkl` + dual mp4).

## 4. Lessons (chronological, each verified)

1. Naive co-training hurts; classifier gating turns 100 uncurated human demos into
   positive transfer (paper's claim, reproduced: position error 4.6mm vs 7.6mm robot-only;
   xdiffusion reached the block every run, robot-only intermittently).
2. Human grasp TIMING labels are good (2mm agreement) — but grasp INITIATION is a rare
   event (~6 robot transitions) that dilutes into a copy-through shortcut without
   flip oversampling.
3. Offline probes that feed recorded states can be confounded (the recorded grip input
   flips exactly at the labeled close) — trust held-grip probes and, above all, on-robot
   logs (`commands.pkl`).
4. A state-only policy is open-loop in task space: grasp fires where ITS internal
   trajectory says, and inter-run DDIM drift is ±1-2cm. Vision (paper's full form) is the
   principled fix; the semantic-anchor design is the state-only workaround that proved
   sufficient at a fixed block position.
5. Impedance-controller droop is invisible in training data (actions are measured poses)
   and must be compensated at the command layer.
6. The released X-Diffusion has no vision path (withheld `baselines` pkg) and no deploy
   code — both were rebuilt here.
7. **Diffusion policies output BIMODAL samples at discrete-decision boundaries** (close?
   release?): any single-draw-plus-threshold reading flickers there. Read decisions as
   statistics over draws (median/vote ensembles) and use direction-specific step
   statistics (min for close, max for open). Three latch bugs in one day traced to this.

## 5. Open items

- ~~Release (re-open) after push~~ **SOLVED & robot-verified (2026-07-23)**: the signal
  was present in BOTH datasets (robot 6/6, human 98/100 via SAM3 contact ending) and
  jointly supervised all along; the blocker was the deploy latch (a single min statistic
  structurally pins at 0 while holding). Direction-specific stats + median-of-3 draw
  ensemble fixed it — **full cycle including prompt release robot-verified 2026-07-23**.
- Vision variant (paper's full form): rgb_seg export + encoder graft — planned.
- Formal 10-rollout A/B (xdiffusion vs robot_only) for the record.
