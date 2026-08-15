# Unified Jar Task — Human/Robot Cross-Embodiment Policy + Label-Free Feasibility Shield

**Status: ROBOT-VERIFIED WORKING (2026-08-15, user-declared best).** One policy drives
both interaction modes of a jar-opening collaboration, and a label-free k\* discriminator
vetoes the physically-infeasible sub-motion online at deploy time.

- **Policy** `X-Diffusion-Data/_private/runs/policy_jar_uni_v10`
- **Shield** `X-Diffusion-Data/_private/runs/kstar_uni/kstar_cls.pth` (`feat="eul_grip_rel"`, `tau=17`)

---

## 1. The task and the science

Two collaboration modes, one unified naive-cotrained diffusion policy (state-only EEF,
7-dim action, 5-dim interferer condition = `cx,cy,vis,dist,side`):

- **Mode B (robot-first):** nobody in view → the robot descends and grasps the jar HOLDER →
  a human right hand rotates the lid → the robot releases and rises. Every robot motion here
  is feasible.
- **Mode A (human-first):** a human LEFT hand grips the holder first → the robot approaches
  the lid → it **would** try to rotate → **that rotation is kinematically infeasible for the
  gripper** → the k\* shield vetoes it and freezes the arm above the lid with a banner.

**Contribution:** the infeasibility is never labeled. The discriminator only ever sees the
free `y∈{robot=1, human=0}` embodiment label; feasibility is the *emergent* answer to
"how much diffusion noise `k` must be added before the classifier stops calling this a robot
motion" (k\* = min k with P(robot) < 0.5, Eq. 3 of X-Diffusion). Rotation is absent from the
robot data by construction → low k\* → veto. This is the whole mechanism.

## 2. Data (both from double-human recording, plus 10 robot demos)

Datasets: `08-14-jar-rotating-human-robot` (20 eps → 10 best robot demos) and
`14-08-double-human-rotating-jar-` (73 eps, dual-hand).

Dual-role reading of the double-human data (`hamer/export_xdiffusion_h5_jar_uni.py`):
- **Role B** = LEFT-hand retarget (grasps holder), condition = right hand, `demo00000+`.
- **Role A** = RIGHT-hand retarget (rotates lid, mirrored fingertip slots `[0,1,2,4,3]`),
  condition = left hand, `feas_mask` marks the rotation span, `demo00100+`.

Data surgery (`hamer/jar_uni_focus.py`, `hamer/jar_uni_prep.py`):
- `--rotate-focus`: re-cut role-A rotation spans as standalone episodes (`demo00200+`) so the
  policy actually commits to rotation emission (rotation is only ~14% of role-A frames).
- `--trim-b` / `--trim-a-tails`: cut the leave/lift tails that form competing attractors.
- `--sentinel-b`: hand-agnostic approach copies (`demo00400+`, pre-close interferer forced to
  sentinel) so the robot descends **without waiting for a detected hand** (fixes arrival-timing
  fragility). Closing still requires the partner (causality preserved).
- `--shift-a "0,-0.027,-0.016"` + `--retrim-leads`: **physical-frame calibration** of the
  A side. The A stream is pure HaMeR retarget with no robot anchor, so it floated ~2.7cm left
  / 1.6cm high; the offset was measured by re-triangulating the RAW rotation-phase fingertips
  (fingers on the real lid) and subtracted. **Lesson: retargeted streams with no robot anchor
  need an explicit physical-frame calibration; raw triangulated contact phases are the free
  anchor.**
- Unified start moved onto the physical approach column top `[0.978, 0.14, 0.645]`.

## 3. The discriminator (`scripts/kstar_uni.py`) — POSITION-BLIND

The decisive redesign. Every false veto during bring-up was **position-driven** — but position
is feasibility-irrelevant in this task (all workspace positions are reachable; only wrist
rotation is infeasible). So the classifier was rebuilt to be blind to position:

- **Features = euler(3) + grip(1)**, euler made **chunk-relative** (subtract row 0) with
  **fixed physical scaling** (rel-euler / 0.5 rad). Quiet chunks of both classes become
  class-ambiguous (P≈0.5, k\*≈0 = feasible); coherent rotation is the sole discriminant.
- **Training task = rotation-like (0) vs robot-executed (1).** Class 0 = activity-mined real
  rotate windows (self-supervised euler-rate segmentation, *not* feasibility labels) +
  rotation counterfactuals injected onto robot content. **Real-human QUIET windows are
  deliberately excluded** — under position-blind features they are indistinguishable from
  robot quiet, and that contradictory gradient drowns the signal.
- Online aggregation: executed deploy chunks fold into D_R (euler-filter keeps rotation out
  regardless of provenance — absence of rotation from D_R *is* the mechanism).
- **`tau = 17`** gives real margin: at the same magnitude, coherent rotation (ramp 0.35 →
  k\* 48) vetoes while incoherent wobble (0.44 → k\* 12) passes. B executed chunks 0/45 flagged;
  rotation-intent 52-72 vetoed. AUROC 0.92.

> **PROCESS RULE (learned the hard way):** never retrain/swap the shield checkpoint
> mid-session. A mid-session aggregation once swapped tau 17→10 under the working system and
> froze mode-B approach (wobble chunks re-flagged → never-execute freeze-cascade). Aggregate
> **only between sessions and only with a rehearsal re-gate.**

## 4. Deploy machinery (`mt_pi/scripts/deploy_xdiffusion_ur10e.py`)

k\* shield in `--shield-kstar` mode (auto-detected from `feat="eul_grip_rel"` → 4-dim classifier
+ relative-euler transform). Beyond the core veto, the demo required a stack of deploy-side
robustness pieces, each traced to a specific robot-run failure:

| flag | what it fixes |
|---|---|
| `--wait-hand 3` | hold at start until the hand is detected+locked (kills the detector-warmup sentinel-drift window); A locks <1s pre-placed, B pays only 3s |
| `--cond-a-proto "928,250,22"` | feed the data-prototype LEFT-hand condition (the human export's `dist` is a ~22px artifact; live ~150px is OOD and mode-collapses A toward B) |
| `--a-descend-lock 0.585` + `--a-yaw-trip 0.06` | on entering the lid zone, freeze xy on the jar axis and trip the veto on cumulative yaw drift (catches slow pre-rotation that per-chunk k\* can't see — a persistence-score ρ monitor) |
| `--shield-sanitize 0.550` | flagged chunks execute TRANSLATION only (euler locked, z-floored) so the freeze lands AT the lid, not 4-6cm above; **anchor advances in policy-space only** |
| height-scoped latch + evidence verdict | rotation intent above the lock is sanitized-through without voting; the honest verdict fires once the gripper (MEASURED z) arrives at the workpiece |
| `--grasp-z-max 0.538` `--grasp-z-assist 3` | suppress premature close while the arm is high; nudge down during suppression |
| `--hold-pin` | pin state+targets to the close-moment pose during the static hold (open-loop imagination drift walks off the release manifold) |
| `--auto-release 3` | force-open after the hand leaves for 3 cycles (stochastic release head) |
| `--speedup 1.6` | time-rescale execution (training data is slow teleop; input distribution unchanged) |
| `--cond-hold 4` | bridge hand-detector flickers of the static gripping hand |

**General deploy lesson:** any user-visible trigger must gate on MEASURED, not imagined
(commanded-propagation), kinematics — and `sem_anchor` must only ever carry policy-space values.

## 5. Final deploy command (user-approved)

```bash
cd ~/Tum_lsy_ur10e_pipeline/ur10_clearpath/Yunfei/crisp_gym
pixi run -e jazzy-lerobot python /home/admin_025/mt_pi_codebase/mt_pi/scripts/deploy_xdiffusion_ur10e.py \
    --run-dir /home/admin_025/X-Diffusion-Data/_private/runs/policy_jar_uni_v10 \
    --ckpt latest --no-safety-observer \
    --hamer-interferer --interferer-hand auto \
    --shield-run /home/admin_025/X-Diffusion-Data/_private/runs/kstar_uni/kstar_cls.pth \
    --shield-m 2 --shield-n 3 --shield-latch --no-home --goto-start --demo \
    --cond-a-proto "928,250,22" --wait-hand 3 --goto-start-speed 0.10 \
    --shield-sanitize 0.550 --a-descend-lock 0.585 --a-yaw-trip 0.06 \
    --grasp-z-max 0.538 --grasp-z-assist 3 --hold-pin \
    --speedup 1.6 --open-horizon 8 --open-settle-cycles 3 \
    --auto-release 3 --cond-hold 4
```

Add `--exit-glide 0.12` to auto-park at start after a B release (single-shot demo); omit it for
a continuous/looping run. **A-mode operating note:** the left hand must grip the holder BEFORE
launch (during goto-start) so the detector locks by cycle 0.

## 6. Reproduce

Data + policy chain (from a clean retarget): `hamer/run_jar_uni_chain*.sh` variants, then
`scripts/kstar_uni.py` for the shield. The v10 policy config is `configs/policy_jar_uni_v10.yaml`.
The blow-by-blow iteration log (v2→v10 policy, every classifier redesign, every deploy fix, and
the physical-frame calibration) lives in the project memory `jar-uni-pipeline.md`.

## 7. Why the bowl-handover task never hit these failures

Handover recorded ROBOT teleop demos for *both* interaction directions, so its geometry was
physically correct by construction, its interferer condition matched the deploy projection, and
it has no feasibility notion at all. The jar task's Mode A is deliberately a **zero-robot-data
mode** (that absence IS the science) — which is exactly what forced the physical-frame
calibration, the condition-prototype shim, and the position-blind actuation-signature
discriminator built here.
