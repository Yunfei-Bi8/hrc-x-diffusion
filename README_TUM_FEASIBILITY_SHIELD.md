# TUM Feasibility Shield — full record (2026-08-06 → 08-13, ROBOT-VERIFIED)

An online **feasibility critic** that watches every predicted action chunk during
policy deployment and **vetoes infeasible motion before it executes** — demonstrated
live on the UR10e jar-opening demo: the policy (trained purely on two-human data)
descends to the jar a partner is holding, and the moment it commits to the
lid-rotation behaviour (affordance-infeasible for a 2F-85 parallel gripper) the
shield fires a full-screen INFEASIBLE banner and holds the pose permanently.

Companion design docs: `RESEARCH_PROPOSAL_FEASIBILITY.md` (three-contribution
proposal + literature verdicts) and `IMPL_FEASIBILITY_CRITIC.md` (critic
implementation plan + all executed experiment results §4.9–4.11).

---

## 1. The diagnosis that started it (08-06)

X-Diffusion trains a human-vs-robot classifier and *interprets* it as feasibility
(k\*, Eq. 3 of the paper). A corruption pilot (`scripts/feascritic_synth_exp.py`
ancestor, doc §0) dissected the two notions: **identity AUROC 0.98, but dynamics
violations at chance** (speed ×4 = 0.51, teleport 0.55 — scoring *more* robot-like
than clean data, whipsaw 0.51). The classifier reads embodiment *style*, not
executability — the two coincide in the paper's tasks only because infeasible
motion there happens to look human. A shield built on it would pass dangerous
outputs and reject benign human-styled ones (88%). Feasibility supervision must be
injected explicitly; that is this whole work.

## 2. Upstream bug found on the way (08-07)

`models/policy_nets/unet.py::HalfUnet1D` (classifier trunk) contained `cond *= 0`
since the public release — **every classifier ever trained on it was blind to
state_cond** (verified: identical weights, output invariant to state before the
fix, responsive after). The paper's `c_θ(k, A^k, s)` was effectively `c_θ(k, A^k)`.
Fixed here; old classifier checkpoints need retraining, not reloading. The policy
net (`ConditionalUnet1D`) never had the bug.

## 3. The FeasibilityCritic (`models/xdiffusion/hr_classifier.py`)

Additive classes (old checkpoints keep loading):
- 3 heads `[identity, kin_feas, afford_feas]`, per-head masked BCE;
- **dual pooling** (avg‖max — max preserves single-step spikes GAP dilutes 1/T);
- **state-channel input** (state broadcast ⊕ action, 7→14 dims): FiLM alone never
  learned state-action comparison (state-blind ablation ≈ unblinded on start_jump;
  the channel variant lifted it 0.735→0.949);
- optional finger branch (HaMeR fingertips, labeler-only) with modality dropout —
  for trace-invisible affordance classes (chopsticks/chords); not needed for jar;
- T-agnostic (fully conv+pooled): trained on 24-step windows, deployed on
  `[16 executed ‖ 8 predicted]`.

Training recipe (each item validated by an ablation that failed without it):
- **low-k noise sampling** (50% k=0 + 50% U[0,21)): uniform-k training *caused*
  the jitter/teleport blindness (iid Gaussian ≡ the forward-diffusion noise; a
  noise-conditioned critic is trained to look through it) — k=0 columns went
  0.64/0.75 → 0.99/0.97. The noise axis is only needed for train-time admission;
  the deployment shield operates at k≈0.
- **mixed-span corruptions** (full / suffix-8 / random suffix): full-window-only
  training degraded 8-step-slice detection 0.97→0.61-0.88 (E-B cells 3≈4 proved
  slice-confinement, not policy texture — generated chunks are texture-neutral,
  AUROC 0.50); mixed spans recovered 0.86-0.99 incl. seam_jump 0.95.
- **still-holding positives**: without them the (slowed) rotation class collided
  with legitimate holds — v1 vetoed a sentinel hover.
- **transition negatives** ([feasible context ‖ rotate suffix] — the deploy
  first-detection shape): without them onset windows scored mid-range → slow veto
  at the lid; with them P_aff(onset) = 0.22, single-window veto 87.5%.
- kin labels for real windows come from the ANALYTIC envelope check (physics, not
  guesswork); afford labels from task-phase segmentation.
- thresholds: **gap-midpoint calibration** (feasible q05 vs negative q95) — the
  naive 95%-TPR quantile sat at 0.954 on a saturated feasible pool and fired on
  benign descent dips 0.008 below it.

Laundering test (E-C): a policy trained purely on affordance-infeasible synthetic
episodes emits chunks the critic still flags (P_aff 0.24 vs 0.90 for a good
policy; euler churn amplified, not laundered) — diffusion policies preserve modes,
so the shield premise survives the data→policy→sampler pipeline. No adversarial
loop exists by construction (the policy never receives critic gradients).

## 4. The 10-08 rotating-jar dataset pipeline (`hamer/export_xdiffusion_h5_jar.py`)

52 two-human episodes: RIGHT hand enters first and fixes the jar (condition
stream, 4-dim [cx,cy,vis,dist]); LEFT hand descends, rotates the lid fast
(infeasible), pulls it off. Retarget via the proven two-human chain. Hard-won
lessons, all robot- or data-verified:

- **Pinch grasp is unreliable on wide grips** (a 6-7 cm lid ≈ open-hand aperture;
  closure bimodality nonexistent). Kept as a weak binary channel; NEVER used for
  segmentation.
- **Segmentation = euler activity** (rotation IS the label): largest contiguous
  span of smoothed wrap-diff euler rate > 0.06 rad/step, 12-step gap bridging.
  (z-profile segmentation failed twice: `np.convolve(mode='same')` zero-pads —
  edge frames become "lowest z" of every episode; and z profiles are
  heterogeneous, some hands enter below the lid.)
- **Phase-aware euler**: approach = CONST canonical tool-down + 0.008 jitter
  (const channel would blow up min-max normalization); rotate/lift rebased so
  rotate-START = canonical, relative rotation (the signature) preserved
  frame-by-frame; whole episode `np.unwrap` (the canonical roll 3.06 sits 0.08 rad
  from ±π — churn crossed it 93 times = teleports in the action channel).
  A whole-trajectory entry rebase had failed first: human grip orientations are
  intrinsically diverse (at-lid scatter median 45°, wrist rotates ~86°
  entry→grip) — never anchor a rigid rebase at the highest-variance moment.
- **Approach straightening**: TIME-paced min-jerk xy line from entry to each
  episode's own lid center (z untouched) — removes the 19-30 mm human lateral
  wander that made deployed descents drift. (Z-progress pacing degenerated on the
  ~1/3 of episodes entering below the lid.) Post: wander 0.0 mm median & p90.
- **Speed**: `--speed-scale 1.5` (auto vs teleop median gave S=2.266 = 24 mm/s,
  robot-verified too slow) → 34-40 mm/s descent, rotation keeps its dynamics.
- Per-episode `feas_mask` (approach/lift = 1, rotate = 0) + 30-frame sentinel
  still-prepend (`make_jar_prepend.py`): nobody → hold; partner grasps → descend.
- Jar placement guidance: lid center at base-frame **[0.927, 0.188, 0.530]**
  (data scatter only ±10 mm — the partner holds it steady).

Critic gates (held-out 8 episodes, `scripts/feascritic_jar.py`): afford AUROC
0.97, rotate veto **100%** @ approach false-vote 5.4%, transition-window veto
87.5%, synthetic families no regression. Policy `policy_jar_v1`: naive 11-dim,
trained on ALL data *including* the infeasible segments — the demo needs the
policy to attempt the rotation so the shield has something to veto.

## 5. Deployment shield (`mt_pi/scripts/deploy_xdiffusion_ur10e.py`)

Worker scores every draw's `[16 executed ‖ 8 predicted]` window at k=0 (~2 ms,
thresholds auto-loaded from the ckpt). Veto machinery, each rule robot-motivated:
- **A flagged chunk is NEVER executed** (single flag = quiet 1-cycle hold +
  redraw; the original execute-until-M-of-N pressed the lid during vote
  accumulation);
- M-of-N (2-of-3) declares the veto episode; **clear hysteresis**
  (`--shield-clear 3` consecutive passes) prevents single passing draws from
  flashing "feasible again" mid-rotation;
- **`--shield-latch`**: for tasks that can never become feasible (this demo),
  the veto is permanent — passing draws ignored, banner stays until operator stop;
- **`--shield-draws 3` pessimistic voting**: near the transition the policy is a
  descend/rotate MIXTURE and one draw per cycle samples it geometrically slowly —
  any rotation-committed draw among ≥3 votes immediately (execution still uses the
  selected passing chunk).

## 6. Demo presentation (`--demo`)

Three-color full-screen state language, auto-scaled to the terminal (block font,
main word up to 6×, subtitles in 1× block font):
**BLUE `GOTO START`** (policy not running; move stats) → **GREEN `INFERENCE`**
(policy live; bottom-row in-place status) → **RED `INFEASIBLE`** (holding pose;
latched). All routine logs → `deploy_logs/demo_*.log`; the hand daemon's output
→ `deploy_logs/daemon_*.log`; camera stale warnings silenced. Non-demo mode is
byte-for-byte the old behavior.

## 7. THE deployment command (jar demo, robot-verified)

```bash
cd ~/Tum_lsy_ur10e_pipeline/ur10_clearpath/Yunfei/crisp_gym
pixi run -e jazzy-lerobot python /home/admin_025/mt_pi_codebase/mt_pi/scripts/deploy_xdiffusion_ur10e.py \
    --run-dir /home/admin_025/X-Diffusion-Data/_private/runs/policy_jar_v1 \
    --ckpt latest --no-safety-observer \
    --hamer-interferer --interferer-hand right \
    --shield-run /home/admin_025/X-Diffusion-Data/_private/runs/feascritic_synth/jarcrit.pth \
    --shield-m 2 --shield-n 3 --shield-latch --no-home --goto-start --demo
```

## 8. Result-evolution table (why each number moved)

| version | change | heldout aff AUROC | rotate veto @ FPR | rehearsal |
|---|---|---|---|---|
| v1-v2 | grasp-threshold + z-profile segmentation (broken) | 0.75 | 24% @ 9% | fail |
| v3 | euler-activity segmentation | 0.938 | 83% @ 8.4% | descend 170mm, veto@22 |
| v4 | entry-anchored orientation rebase | 0.943 | 93% @ 9.5% | + but strange poses on robot |
| v5 | phase-aware euler (const approach + unwrap) | 0.965 | 96% @ 8.1% | veto@20 |
| v6 | S=1.5 + straightening (z-paced, 1/3 broken) + never-execute-flagged | 0.955 | 99% @ 8.9% | veto@10 |
| v7 | time-paced straightening | 0.952 | 97% @ 6.3% | veto@9, z stops at lid top |
| final | + transition negatives + pessimistic draws + gap-mid thresholds | 0.971 | **100% @ 5.4%** (onset 87.5%) | veto@9 |

Every euler/xy nuisance-variance cleanup ALSO sharpened the critic — the class
boundary and the data hygiene are the same fight.

## 9. Repo map

- `X-Diffusion`: critic model + training (`scripts/feascritic_jar.py`,
  `scripts/feascritic_synth_exp.py`), configs, design docs, this README.
- `hamer`: jar preprocessing (`export_xdiffusion_h5_jar.py`,
  `make_jar_prepend.py`, `run_10_08_jar.sh`, `run_jar_chain*.sh`).
- `mt_pi`: deployment (`scripts/deploy_xdiffusion_ur10e.py` — shield + demo UX).
