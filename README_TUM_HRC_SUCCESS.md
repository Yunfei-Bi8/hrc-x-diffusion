# TUM UR10e Human-Robot-Collaboration Success — X-Diffusion HRC `green_lego` (2026-07-28)

**First working HRC policy on our rig, ROBOT-VERIFIED 2026-07-28**: the robot performs the
full task alone (descend → grasp → push right → release), and when a live human RIGHT hand
approaches the lego mid-descent it **retreats upward instead of continuing** — driven by a
live HaMeR hand detector feeding a 4-dim interferer condition. Trained WITHOUT any
robot-human HRC demonstrations: the avoidance behavior comes entirely from **human-human**
data in which one person's LEFT hand plays the robot's role. Run:
`policy_hrc_1st_robot_avoid_human` (wandb `x-diffusion-hrc/hg9qo2hv`).

Builds directly on the cross-embodiment baseline — read
`README_TUM_CROSSEMBOD_SUCCESS.md` first; everything there (7-dim action space, retarget
chain, classifier gating, graspall_flip recipe) is inherited unchanged. This file records
only the HRC layer on top.

## 0. Why human-human data can train a robot-human policy

- **Actor channel** (left hand → 7-dim EEF actions): has an embodiment gap, but it is the
  SAME gap the cross-embodiment machinery already solved (retarget + classifier gating).
- **Interferer channel** (right hand → condition): human hand at training time, human hand
  at deploy time — **zero embodiment gap by construction**. Only the actor changed.
- The model learns: condition = sentinel → do the task; condition = hand visible & close →
  retreat. Gate B proved the mapping is causal, the robot proved it physically.

## 1. Data (3 sources, one merged `green_lego` pool)

| source | episodes used | role | condition rows |
|---|---|---|---|
| `06-20-teleop` (robot) | demo00000–04 train, 00005 val | real dynamics + grasp | sentinel |
| `25-06-single-human` | 98 train / 2 val (demo00000–99) | task itself | sentinel |
| `24-07-two-human` | **87 train / 2 val** of 100 (demo00100–199) | avoidance reaction | live features |

- 24-07 task script: LEFT hand starts descending as in 25-06; RIGHT hand (interferer)
  approaches the lego; LEFT retreats UP without grasping. Left NEVER grasps in 24-07 →
  `gripper_open ≡ 1`, position-level fidelity suffices (the load-bearing realization that
  allowed interpolation over sparse stereo anchors).
- **Excluded 11 "spontaneous retreat" demos** (interferer visible <30% — retreat with an
  invisible cause would teach contradiction): demo00122, 124, 129, 130, 134, 162, 164,
  166, 168, 171, 184.
- H5 contract adds one dataset: `interferer (T,4) = [cx_px, cy_px, vis, dist_px]` in
  **camera_01 1280×720 px**; sentinel row `(0, 0, 0, 1500)`. Files without the key (robot,
  25-06) get sentinel at load time — one loader, one human dir, demo numbering disjoint.

### 1a. Two-hand identity pipeline — `~/hamer/process_two_human_sam3.py`
The hard part: keep LEFT=actor / RIGHT=interferer stable per camera (every choice measured):
- **camera_01**: SAM3 concept track ("hand", always 2 stable ids) + **arrival-order slot
  lock** (left enters first in every episode — task-script invariant) → forced `is_right`
  per slot into HaMeR. Coverage 83–100%.
- **camera_02** (hostile view: near-simultaneous arrivals, SAM3 forearm leaks): candidates
  = MediaPipe(conf .05) ∪ SAM3 boxes (labels ignored) → batched HaMeR(left) → accept the
  fit whose WRIST lies within 100px of the epipolar line of camera_01's left wrist
  (F from the 0613 calib; true hand 3–6px vs wrong hand 280px+). Residual medians 4.9–22.7px.
- Auto-chain `~/hamer/run_24_07_auto.sh`: 3-episode probe → residual QC gate (<25px) →
  full 100-episode emit. Dead ends (do NOT retry): ViTPose chain (coverage), box-center
  epipolar (50–500px off the wrist line), depth back-projection (24-07 depth not
  pixel-aligned to color), ≥40% cam_02 coverage gate (self-occlusion makes it impossible).

### 1b. Dual-stream export — `~/hamer/export_xdiffusion_h5_two_human.py`
- LEFT → the PROVEN cross-embod chain (triangulate at both-visible anchors → fingertip
  cross-cam filter → workspace-box filter x∈[0.70,1.15] y∈[−0.05,0.40] z∈[0.30,0.95] →
  interpolate → explicit retarget → speed resample **S=2.034** vs robot demos) → appended
  as `human/demo00100+`.
- RIGHT → camera_01-only 2D features per REAL detection frame: pinch center = mean of
  HaMeR index_tip(row 8) + thumb_tip(row 4); `dist` = px distance to the LEFT hand's pinch
  center (the robot-role reference); nearest-neighbor resampled on the same timeline;
  `vis=1` only at real detections. 2D-only was deliberate: exactly reproducible online.

## 2. Conditioning + training design (the HRC-specific decisions)

- **11-dim state_cond** = action(7) ⊕ interferer(4). Action/prediction stays 7-dim — the
  interferer is never predicted. Loader: `use_interferer: true` in `h5_dataset.py`.
- **D1 — classifier never sees the interferer dims**: trained with `use_interferer: false`
  (7-dim), and policy training slices `state_cond[..., :7]` at BOTH classifier call sites
  (and instantiates it 7-dim). An 11-dim classifier would learn the trivial "vis=1 ⇒
  human" shortcut and gate out exactly the avoidance supervision.
- **Gate C (ran, FAILED, by design)**: the 7-dim classifier admits 24-07 retreat windows
  only 1.2%@k=10 … 26%@k=80 (mean k∈[20,60] = 14.1% < 15% threshold). Structural cause:
  the 5 robot demos contain NO retreat, so low-noise discrimination must call avoidance
  "human". Pure gating would dilute the one behavior HRC exists to learn.
- **The fix — vis-forced admission** (`compute_xdiffusion_loss_masks`, train_policy.py):
  human rows with vis (`state_cond[...,9] > 0.5`) are ALWAYS supervised — the same
  precedent as graspall's grasp-dim bypass (human data is the sole source of a behavior ⇒
  don't gate that behavior). Sentinel rows (robot, 25-06) gate as before; the 7-dim
  ablation is untouched. Final-epoch masking: 12744/21111 human rows included (60%), of
  which 7449 vis-forced (would be ≈25% without the patch). Logged as `human_vis_forced`.
- **Interferer-visible window oversampling ×3** (`interferer_oversample_factor`, MT-π
  h_visible precedent) on top of flip ×15.
- Everything else = the winning graspall_flip recipe verbatim (50/50 sampling, DDIM
  101/16, bs128 × 200ep × 300 steps ≈ 20 min, `grasp_dim_robot_only: false`).

**Configs**: `configs/classifier_hrc_green.yaml`, `configs/policy_hrc_green.yaml`,
`configs/policy_hrc_ablation_green.yaml` (no-condition control, wandb `7be1trs8`).
**Chain runner**: `run_hrc_training.sh` (classifier → Gate C hard-stop → policy →
ablation → Gates A+B).

## 3. Offline gates (all PASSED before the robot ever moved)

- **Gate A — solo preserved**: sentinel condition on robot demo00005: open-loop pos_err
  **5.0mm**; grasp fires 21 steps early = closes 8.6mm high (inside baseline envelope).
- **Gate B — counterfactual causality**: same 24-07 val states (interferer close):
  REAL condition → chunk-end **Δz = +28mm (retreat)**; SENTINEL on identical states →
  −2mm (descend). The reaction is condition-driven, not state-memorized.
- Checkpoints kept: `latest.pth` (ep199, deployed) and best-val ep33.

## 4. Deploy — live HaMeR interferer (`mt_pi/scripts/deploy_xdiffusion_ur10e.py`)

```bash
cd ~/Tum_lsy_ur10e_pipeline/ur10_clearpath/Yunfei/crisp_gym
pixi run -e jazzy-lerobot python /home/admin_025/mt_pi_codebase/mt_pi/scripts/deploy_xdiffusion_ur10e.py \
  --run-dir /home/admin_025/X-Diffusion-Data/_private/runs/policy_hrc_1st_robot_avoid_human \
  --ckpt latest --no-safety-observer \
  --hamer-interferer          # omit for SOLO mode (constant sentinel = on-robot Gate A)
```

- Worker auto-detects state_dim (7/11) from `dataset_stats/train.npy` — old 7-dim runs
  deploy unchanged.
- `--hamer-interferer` spawns `~/hamer/online_hand_daemon.py --is-right 1` (flag added;
  default 0 keeps MT-π left-hand compat) in the `hrcyf-gpu` conda env via the
  `hrc_hand_shm` double-buffered bridge; deploy feeds camera_01 @10Hz; robot motion waits
  for daemon ALIVE.
- **Feature bit-exactness with training**: pinch = mean(kps 8,4); shm 848×480 kpts scaled
  ×(1280/848, 720/480) into training px (same-FOV pure rescale, template-verified ±2px);
  `dist` = to the MEASURED TCP projected via 0613 hand-eye + factory 1280×720 intrinsics
  + 8-coeff distortion; detection stale >1.5s (`--hand-stale-ms`) → sentinel `(0,0,0,1500)`
  bit-exact. Known bias: deploy dist reads −17px vs the training reference (median) —
  conservative direction (slightly earlier reaction).
- Human must interfere with the **RIGHT hand** (matches training handedness).
- Cycle log prints `hand=(cx,cy)d=…px` / `hand=absent`; `commands.pkl` records the
  interferer vector per cycle.

## 5. Verified behavior + known limits

- ✅ Solo: full task cycle (unchanged vs baseline). ✅ Right hand approaches mid-descent →
  upward retreat. Robot-verified 2026-07-28.
- **Post-withdrawal behavior is UNDEFINED in the data** — every 24-07 episode ends at the
  retreat, none shows resuming the task. Observed behavior after the hand leaves is
  whatever the sentinel condition implies from the current (elevated) state. If "resume
  task after interferer leaves" is wanted, collect episodes that show it.
- Open follow-ups: formal solo A/B (HRC vs graspall baseline, 10×2); reaction-distance
  statistics vs training distribution; vision variant.
