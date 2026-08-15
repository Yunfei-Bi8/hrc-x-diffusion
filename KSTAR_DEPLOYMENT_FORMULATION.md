# k\*-based Deployment Feasibility — formulation for the unified jar task (2026-08-13)

Goal: replace the supervised (phase-labeled) shield critic with the ORIGINAL
X-Diffusion mechanism — an identity classifier whose **minimum
indistinguishability step k\*** is computed per policy-predicted chunk at
deployment; k\* > τ ⇒ INFEASIBLE ⇒ hold. No feasibility labels anywhere in
training. This document fixes the formulation, the required structural
modifications, the threshold calibration, and the failure modes — each grounded
either in the paper (verified against the PDF, not memory) or in our own
measured experiments.

---

## 1. Setup and notation (the unified jar task)

One two-human recording, read under BOTH role assignments (the handover-uni
playbook):

- **hand-1** enters first, descends, grasps the 3D-printed **holder** on the
  jar body, holds. (Robot-feasible content.)
- **hand-2** enters second, descends to the lid, **rotates** it. (Infeasible
  for the 2F-85.)
- Mode A (demo): human holds → robot approaches the lid → policy emits
  rotation-like chunks → shield vetoes → hold + banner.
- Mode B (demo): robot grasps the holder (executes feasibly) → human rotates
  the lid.
- Additionally: **5–10 robot teleop demos** of the feasible side (descend →
  pinch holder → hold → release). These exist for the mode-B policy anyway;
  they are also exactly what makes k\* well-posed (§5, F1).

Notation: state s ∈ R^7 (EEF pos, euler, gripper), action chunk
A = a_{t:t+S} ∈ R^{S×7} (S = 8, the policy's pred horizon), forward-diffusion
noising A^k at step k ∈ [0, K), K = 101, same squaredcos schedule as the
policy.

## 2. The original mechanism (verified against the paper text)

- **Classifier (Eq. 2)**: c_θ(k, A^k, s) → P(embodiment = robot), trained with
  BCE on samples drawn **equally** from D_R (robot demos, y = 1) and D_H
  (human retargets, y = 0), k sampled uniformly, noise applied to A (and s).
  Identity labels only — free at collection; no feasibility annotation.
- **k\* (Eq. 3)**: k\*(A) = min{ k : c_θ(k, A^k, s) ≥ 0.5 } — the earliest
  noise step at which the (noised) human action is indistinguishable from
  robot actions.
- **Gated policy loss (Eq. 4)**: robot samples supervise at all k; human
  samples only at k ≥ k\*. ("Coarse task guidance while avoiding the transfer
  of physically infeasible behaviors" — Discussion, verbatim.)
- The paper uses the classifier at TRAINING TIME ONLY. The deployment rule
  below is our extension.
- Code-level caveat we verified: the public `HalfUnet1D` zeroed the state
  conditioning (`cond *= 0`); with our fix, s actually participates. Any
  classifier must be trained post-fix.

## 3. Deployment formulation

### 3.1 Admission curve and two equivalent scores

For a policy-predicted chunk (A, s), define the **admission curve**

  g(k) = E_ε [ c_θ(k, A^k(ε), s^k(ε)) ]        (estimated with n_ε ≈ 6 draws)

Two scores:

1. **k\*** (paper-faithful): first k with g(k) ≥ ½, using a 2-consecutive-
   grid-point crossing rule (our kstar experiments showed raw crossings
   flicker). If never: k\* = K.
2. **Feasibility persistence** ρ = (1/K) Σ_k 1{g(k) ≥ ½} — the fraction of the
   noise axis that admits the chunk. Under a monotone g, ρ = (K − k\*)/K;
   unlike k\*, ρ is robust to non-monotone curves and is the score we
   recommend logging alongside k\*.

### 3.2 Decision rule

  INFEASIBLE(A, s)  ⇔  k\*(A, s) > τ        (equivalently ρ < 1 − τ/K)

Integrated into the EXISTING shield plumbing unchanged: pessimistic
aggregation over the cycle's draws (vote on **max k\*** across draws — any
rotation-committed draw votes), a flagged chunk is NEVER executed, M-of-N
(2-of-3) declares the veto episode (full-screen banner), clear-hysteresis /
`--shield-latch` as before.

### 3.3 Threshold calibration — label-free

  τ = q_{1−α}( { k\*(A, s) : (A, s) ∈ held-out ROBOT windows } ) + Δ

Robot demos are feasible **by construction**, so this touches no feasibility
labels. α = 0.01, margin Δ ≈ 10 noise steps. Deployment-validity gate: the
achieved gap must satisfy q05(k\* of rotation-phase windows) − τ ≥ Δ (phase
labels used for this GATE only, never for training) — the gap-midpoint lesson
from the supervised critic (razor-thin thresholds fired on benign dips)
transposed to the k\* axis.

### 3.4 Cost

Coarse-to-fine search over k (g is regularized toward monotone, §4.3): ~7
evaluations × 6 draws × n_chunks ≈ 10–20 ms on the deploy GPU — fits the
pipelined cycle. Fixed-grid sweep (26 × 6) ≈ 40–60 ms also fits.

## 4. Training specification (and the structural modifications)

### 4.1 Data classes (no feasibility labels)

- D_R (y=1): ALL windows of the jar-task teleop demos.
- D_H (y=0): ALL windows of BOTH retargeted human streams — holder-hand AND
  rotator-hand. (The holder-hand windows are supposed to be confusable with
  D_R — that is the mechanism, not a bug: they get low k\* and, in Eq. 4
  co-training, are admitted almost everywhere.)

### 4.2 Recipe (paper + two additions our experiments proved necessary)

- Uniform k ~ U[0, K) (k\* needs the whole axis — do NOT reuse the low-k
  recipe of the supervised shield critic; that one optimizes a k=0 operating
  point, which is a different estimator).
- Balanced 50/50 class sampling; BCE; T = 8 windows (policy-chunk shape, so
  one classifier serves both the deployment shield and Eq. 4 co-training);
  ~3000 steps, bs 256, lr 1e-4; freeze after training.
- **[Addition 1 — texture equalization]** apply an IDENTICAL small noise
  augmentation to BOTH classes (euler jitter σ ≈ 0.008 rad, position residual
  σ ≈ 1.2 mm, matching the human-export texture), and match teleop speed to
  the human-export approach speed (~30–40 mm/s). Measured motivation: without
  this, the classifier separates the shared (feasible) content by processing
  texture and k\* saturates high everywhere — our three jar experiments
  (robot class from another task / + workspace synthetics / + texture match at
  T=24) scored k\*-AUROC 0.515 / 0.591 / 0.491. Same-task robot demos remove
  the domain gap; equalization removes the texture gap.
- **[Addition 2 — monotone regularizer]** k\* is well-defined only if g
  crosses ½ once. Theory says g(k) → ½ as k → K for both classes (the noised
  distributions merge — the Ambient premise), rising for human samples and
  falling for robot samples; the raw network only approximates this. Add

    L_mono = E_{y=0} Σ_k max(0, g(k) − g(k+δ)) + E_{y=1} Σ_k max(0, g(k+δ) − g(k))

  (a few sampled k pairs per batch, weight ~0.1). This makes the crossing
  unique, stabilizes k\*, and licenses binary search at deployment.
- **[Optional — ensemble]** 3 seeds; veto requires k\*-agreement (2 of 3 over
  τ). Epistemic guard for chunks resembling NEITHER class (true OOD): a lone
  member's spurious crossing cannot fire the banner.

### 4.3 Validation gates (labels only here)

1. Held-out ROBOT windows: median k\* ≤ 10 (sanity; feeds §3.3).
2. Held-out human episodes, per-window k\* profile: holder-phase median LOW
   (target ≤ ~25), rotation-phase median HIGH (target ≥ ~60); k\*-AUROC
   (rotation vs holder) ≥ 0.85.
3. Gap gate: q05(rotation k\*) − τ ≥ Δ.
4. Monotonicity: fraction of val curves with >1 crossing ≤ 5%.

## 5. Failure modes (each observed by us) and their mitigations

- **F1 cross-task class gap**: robot class from a different task ⇒ k\* measures
  workspace/domain, not feasibility (measured: AUROC ≈ 0.5). Mitigation:
  same-task teleop demos — now available by design.
- **F2 texture leak**: export-pipeline noise signatures separate the shared
  content ⇒ feasible-phase k\* inflates. Mitigation: equalization augmentation
  + speed matching (§4.2-1); gate 2 detects residual leak.
- **F3 non-monotone admission curves**: flickering crossings ⇒ unstable k\*.
  Mitigation: monotone regularizer + 2-consecutive rule + ρ as the logged
  robust score.
- **F4 razor thresholds**: calibrating τ on a saturated quantile fires on
  benign dips (robot-observed with the supervised critic). Mitigation:
  robot-only quantile + Δ margin + the gap gate.
- **F5 true-OOD chunks** (resemble neither class): g ill-defined. Mitigation:
  ensemble agreement; optionally keep the analytic kinematic-envelope check as
  a parallel, label-free guard (it needs no learning at all).

## 6. Interaction with policy training for the demo

- The DEMO policy should be trained **naive** (no Eq. 4 gating), exactly as
  the current jar policy: in mode A it must faithfully ATTEMPT the rotation so
  the shield has something to veto. (A gated policy learns rotation only as
  coarse guidance — its low-noise layers never see the fine execution — so its
  mode-A outputs would be vague drift, a weaker demo.)
- The SAME frozen classifier additionally provides Eq. 4 gating for an
  X-Diffusion-faithful co-trained variant (the paper's method column):
  holder-hand human data admitted almost everywhere (low k\*), rotation data
  only at high k. One classifier, two uses — the paper's mechanism plus our
  deployment extension.
- Unified-policy conditioning (mode selection, entry-lock, bridges) reuses the
  handover-uni playbook unchanged.

## 7. Related anchors

- Ambient Diffusion / Ambient Omni [19, 20 in the paper's refs]: the
  noise-level data-selection classifier — the training-time ancestor of k\*.
- Ambient Diffusion Policy (arXiv 2606.12365): per-dataset timestep-band
  gating — nearest neighbor of the continuous per-sample k\*.
- Diffusion-based OOD detection (multi-noise-level reconstruction/likelihood
  scoring, e.g. Graham et al. 2023): the deployment-time relatives of
  "score a sample by its behavior along the noise axis".
- Our internal evidence chain: 08-06 identity≠feasibility pilot; E-A noise-
  sampling ablation; the three label-free k\* jar experiments
  (`runs/kstar_jar*`); the supervised-critic robot deployments (the plumbing
  the k\* scorer drops into).

## 8. Implementation deltas (once the new data exists)

1. `scripts/kstar_identity_jar.py` → point D_R at the jar teleop export; add
   texture-equalization augmentation, L_mono, the four gates, τ calibration
   (writes τ, Δ, ensemble ids into the ckpt).
2. Deploy worker: `--shield-kstar` mode — replace the probability scorer with
   the k\* search (coarse-to-fine); votes/banner/latch/never-execute unchanged;
   thresholds auto-loaded from the ckpt.
3. Everything else (recording protocol, exporters, prepends, unified-policy
   configs) already exists from the handover-uni + jar pipelines.
