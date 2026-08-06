# Feasibility-Aware Cross-Embodiment Learning — research proposal (2026-08-06)

Target: leading robotics venue (CoRL / ICRA / RSS). Base system: our HRC extension of
X-Diffusion on the UR10e handover rig (unified two-direction handover, robot-verified).
This document answers three questions posed on 2026-08-06:

1. Can the X-Diffusion discriminator become a **deployment-time safety layer**? (detailed design)
2. Can it **label datasets** into feasible/infeasible parts for training? (design + extensions)
3. Where else in the X-Diffusion framework is there room for **scientific contribution**?

---

## 0. Pilot experiment (2026-08-06): the discriminator measures IDENTITY, not FEASIBILITY

Before designing anything we tested the core premise. X-Diffusion's classifier
`c_θ(k, A_t^k, s_t)` is *trained* on embodiment identity (BCE, human=0 / robot=1, Eq. 2
of the paper) but *interpreted* as kinematic/dynamic feasibility (k*, Eq. 3; Fig. 5
"naive co-training learns infeasible actions"). Those two notions coincide only when
infeasible motion happens to look human-styled. Do they dissociate?

**Protocol** (`scratchpad/shield_pilot.py`): frozen `classifier_handover_uni`
(val loss 0.514), val windows of `handover_bowl_uni` (347 robot, 603 human), 8
physically-motivated corruption families applied to ROBOT val chunks in raw space,
scored at diffusion steps k ∈ {0,5,10,20,40}, 8 noise draws averaged. AUROC =
clean-robot vs corrupted.

| family (on robot chunks) | violates | P(robot) @k0 | AUROC @k0 |
|---|---|---|---|
| clean robot | — | 0.788 | — |
| clean human (retargeted) | identity only | 0.018 | **0.98** |
| speed ×2 / ×4 | velocity limits | 0.785 / 0.778 | 0.50 / **0.51 (chance)** |
| teleport +5 cm mid-chunk | continuity | **0.861 (MORE robot!)** | 0.55 |
| jitter 10 mm iid | smoothness | 0.782 | 0.52 |
| whipsaw ±40 mm (the real OOD failure we measured on-robot 08-05) | dynamics | 0.818 | 0.51 |
| grasp flutter 0/1 each step | gripper semantics | 0.692 | 0.67 |
| euler spin 0.3 rad/step | rotation dynamics | 0.528 | 0.77 |
| still (freeze) | nothing (feasible) | 0.785 | 0.50 ✓ correctly kept |

**Findings.**
- Identity separation is nearly perfect (0.98) — the classifier does its trained job.
- It is **blind to dynamic infeasibility** (speed ×4 at chance; a mid-chunk teleport
  scores *more* robot-like than clean robot data). Root cause: our human data was
  deliberately speed-aligned to robot speed and both classes are continuous — the
  training classes never *differ* along velocity/continuity axes, so the classifier
  never learned them. In the paper's tasks, human style (side grasp) *correlates* with
  infeasibility, so identity works as a proxy; the proxy breaks exactly when infeasible
  motion is robot-styled — which is what a malfunctioning policy emits (our measured
  whipsaw bursts pass at 98%).
- A naive "use the discriminator as a runtime shield" is therefore **empirically wrong
  twice over**: it passes dangerous outputs (whipsaw 2% rejected) and rejects benign
  human-styled outputs (88% of clean human chunks rejected at the same threshold).
- Weak-but-real signal on grasp-flutter (0.67) and euler-spin (0.77): embodiment-STYLE
  channels (grasp patterns, rotation conventions) is what it actually learned.

This is the scientific wedge for the paper: **"embodiment identity ≠ execution
feasibility"** — a diagnosis of X-Diffusion's central mechanism, with a constructive
fix below. It also *validates the need for explicitly collected infeasible data*:
feasibility signal cannot be extracted for free from the H/R classifier because the
signal is not in its training classes.

---

## 1. Design A — the Feasibility Critic `D_feas` (upgraded discriminator)

One network, same architecture as the existing classifier (HalfUnet1D over the noised
8×7 action chunk, conditioned on the 7-dim proprio state + diffusion step k, GAP +
linear head) → drop-in for every place the current classifier is used. New training
classes:

**Positives (feasible):**
- all robot demonstrations (any task on the rig);
- optionally retargeted human demos that pass the analytic screen (below), label-smoothed 0.9.

**Negatives (infeasible), three complementary sources:**

1. **Synthetic dynamics corruptions** (free, unlimited): the 8 pilot families applied
   on-the-fly in the loader with randomized magnitudes (speed ∈ [1.5,6]×, teleport ∈
   [2,10] cm, jitter ∈ [5,25] mm, whipsaw ∈ [15,60] mm, euler ∈ [0.1,0.5] rad/step,
   grasp flutter). Covers *kinematic/dynamic* infeasibility. Curriculum: large
   magnitudes first, anneal down to the decision boundary.
2. **Analytic auto-labels** (free): replay retargeted human EEF traces through the
   UR10e differential IK + limits check (joint velocity/acceleration, workspace box,
   self-collision proxy). Violating windows → negative. This is the FILTERED baseline's
   manual step (paper §V-B, 50% discard) turned into an automatic *labeler* instead of
   a data-discarder.
3. **Collected affordance negatives** (the new two-human datasets, §4): human hand does
   things a 2F-85 cannot — these retarget to *in-limits* EEF traces that are still
   non-executable (in-hand reorientation → grasp flutter + euler noise signatures).
   This is the class NO analytic check can catch, and the pilot shows it is learnable
   (flutter/spin are precisely the families with signal already).

**Heads.** Multi-task heads sharing the trunk: `[identity, kin_feas, afford_feas]`
(3 sigmoids). Keeps the original identity function (needed for X-Diffusion training
admission) while adding calibrated feasibility outputs; ablation "single feas head vs
multi-head" is a paper table. Noise-conditioning is inherited → every head is defined
along the k-axis, preserving the framework's key idea (feasibility as a *function of
abstraction level*) but now measuring feasibility rather than identity.

Files: `models/xdiffusion/hr_classifier.py` (add head dim + loss), new
`dataset_utils/corruptions.py` (families, magnitudes, curriculum),
`scripts/train_classifier.py` (negative-source mixing weights in config),
`scripts/ik_screen.py` (analytic labeler → per-window feasibility h5 key).
Estimated effort: 2-3 days including retrain (classifier trains in ~30 min on our data).

---

## 2. Design B — deployment-time safety shield (question 1, detailed implementation)

### 2.1 Integration point

Our deploy is already pipelined with a **batched-draw worker**
(`deploy_xdiffusion_ur10e.py`): parent sends `(state, n_draws)`, worker returns
`(n, 8, 7)` action chunks. The shield lives in the worker (GPU already hot):

```
worker: chunks = policy.sample(state, n)                  # existing
        f      = D_feas(chunks_normalized, state7, k=0)   # NEW: one batched forward,
                                                          #      (n,) feasibility scores
        return chunks, f                                  # protocol adds one array
```

Cost: one HalfUnet1D forward on n≤4 chunks ≈ 1-2 ms — invisible next to the 0.2 s
diffusion sample; the pipelined cycle budget is untouched.

### 2.2 Decision policy (parent, per cycle)

Three composable levels, flags `--shield-mode {off,select,veto,both}`:

1. **SELECT** (soft): among the n draws, drop those with `f < τ_soft`, then apply the
   existing `--draw-consistency` anchor-nearest rule to the survivors. Degrades to
   current behavior when all draws pass. This alone would have filtered the 08-05
   whipsaw bursts *if* D_feas fires on them (pilot: current classifier does NOT →
   requires Design A; with synthetic-negative training, whipsaw is in-distribution
   negative by construction).
2. **VETO** (hard): if ALL draws have `f < τ_hard` for M of the last N cycles
   (default 3-of-4, hysteresis against flicker) → **do not advance the anchor**: hold
   the current pose (re-command anchor), latch gripper state, log `SHIELD-VETO`.
   Recovery: K consecutive passing cycles (default 4) resumes. This is "the robot does
   nothing when the predicted action is infeasible" from the original idea — made
   precise: *hold last commanded pose*, not zero-torque (compliant controller keeps
   gravity compensation; k_pos=400 holds against drift).
3. **ESCALATE**: sustained veto > T seconds (default 6 s) → slow retreat to start pose
   and require operator keypress. Composes with the existing settle interlock and
   grasp latch (shield veto while HOLDING an object must never open the gripper —
   explicit rule: veto freezes the grasp channel too).

Threshold calibration is empirical and honest: τ set so that ≥99% of chunks from our
*robot-verified successful* deploy sessions pass (we have those logs), and ≥95% of
synthetic corruptions + collected-infeasible val windows fail. Report the resulting
operating point as a precision/recall curve in the paper — no hand-waving.

### 2.3 Sampling-level integration (the algorithmically novel part)

Because D_feas is noise-conditioned (trained at all k), it can score the sampler's
**intermediate x̂₀ predictions** at every DDIM step — not just the final output:

- **Early veto / resample:** at DDIM steps k = {40, 20, 10}, score x̂₀; if the batch
  best is below τ, re-noise and re-draw that trajectory branch (up to R retries) —
  rejection sampling moved *inside* the sampler, cheaper than full re-draws.
- **Feasibility guidance:** add the discriminator-guidance term (Kim et al., ICML 2023,
  image domain) to the reverse update:
  `ε̂' = ε̂ − w(k) · √(1−ᾱ_k) · ∇_{A^k} log D_feas(A^k, s, k)`
  steering samples toward the feasible manifold instead of blocking them. w(k)
  annealed to 0 at low k (guide early, commit late). To our knowledge discriminator
  guidance has not been used for *embodiment feasibility* in robot diffusion policies —
  this is the cleanest algorithmic novelty of the proposal, and it directly reuses the
  framework's own component (elegant: X-Diffusion trains the critic anyway; we make it
  earn its keep at deployment).

Implementation: `models/xdiffusion/shield.py` (score + guidance hooks),
`deploy_xdiffusion_ur10e.py` worker (+~60 lines), flags `--shield-run-dir
--shield-mode --shield-thr --shield-guidance-w`. Offline replay harness first:
re-run the logged 08-05 OOD sessions (mid-task hand withdrawal → whipsaw) through the
shield and count vetoes/bursts before any robot test.

### 2.4 Honest limits (write these in the paper)

- The shield judges *chunks*, not long-horizon intent: a feasible-looking chunk can
  still be task-wrong. It is a dynamics/affordance filter, not a task monitor —
  compose with runtime task monitors (Sentinel/STAC-style) rather than compete.
- Veto = availability loss; report intervention rate alongside safety gains.
- Guidance can push off the policy's data manifold; bound w(k) and ablate.

---

## 3. Design C — the critic as a dataset labeler (question 2)

Offline pass over any demo pool: slide the 8-frame window, score all three heads at a
low-k grid → per-frame profiles `identity(t), kin_feas(t), afford_feas(t)` stored as
h5 keys. Cheap (one forward per window). Four training uses, ordered by
effort-to-payoff:

1. **Soft admission (replaces Eq. 4's hard gate).** X-Diffusion supervises a human
   sample iff `k ≥ k*` (binary Monte-Carlo gate). Replace the indicator with a
   continuous weight `w = P_feas(A, s, k)^γ` inside
   `compute_xdiffusion_loss_masks()` — one-line change, directly ablatable against
   the paper (γ sweep; γ→∞ recovers the gate). Removes the threshold=0.5
   arbitrariness (Eq. 3) and our measured admission-rate instabilities.
2. **Segment-masked co-training (the HRC-shaped extension).** Demos with infeasible
   *segments* (afford_feas(t) low, sustained): mask the action loss there but KEEP the
   frames as observation/condition context. The policy learns to *wait through /
   anticipate* partner phases it cannot imitate — a generalization of our handover
   design, where the partner hand is condition-only by construction. Contrast with the
   paper's FILTERED baseline (discards 50% of demos wholesale) and with X-Diffusion
   (admits whole windows only at high noise): we keep every frame, at the right role.
   This is what makes the *mixed* feasible/infeasible collections (§4) trainable at all.
3. **Feasibility-conditioned generation (stretch).** Feed `f` as an extra condition
   with dropout; at deploy clamp f=1 → classifier-free-guidance-style "generate the
   feasible mode". Interacts with Design B guidance — pick one for the paper, ablate.
4. **Data-quality dashboards / curricula:** rank demos by profile, surface collection
   errors (we found real ones — teleports from recording stalls — by hand; the critic
   automates the forensics).

---

## 4. Design D — the feasible/infeasible two-human collections (data spec)

Two matched datasets on the existing rig (same cameras/calib/HaMeR pipeline, ~1-2 h
each), *partner-conditioned* so the feasibility decision depends on the interaction
context, not just the motion:

| | FEASIBLE set | INFEASIBLE set |
|---|---|---|
| hand-1 (action, retargeted) | pick / place / push / handover of the bowl | in-hand pen spin; cap-twist with regrasp cycles; chopstick pickup; card slide-and-flip off the table edge; cloth fold with fingers; press small button WHILE holding an object |
| hand-2 (condition) | open-palm reach ("give it to me") | point at the object / rotate-gesture ("open it", "spin it") |
| retargeted signature | clean EEF traces | in-limits positions but grasp-flutter + euler-noise + micro-jitter |

Design rules learned from the pilot:
- Choose infeasible tasks whose retargeted EEF trace is **in-limits but semantically
  non-executable** — that is the class only a *learned* critic can catch (the analytic
  IK screen catches the rest for free; do not waste collection time on out-of-limits
  motion).
- Same objects, table region, lighting, subjects across both sets — otherwise the
  critic learns the scene, not the feasibility (balance check: train scene-classifier
  probe, demand chance accuracy).
- Grasp-channel ground truth via the existing SAM3 bowl-contact builders; for
  non-bowl objects add a per-object concept track (one line each).

**The robot demo this enables** (paper's money shot): partner makes the feasible
request → robot executes the handover; partner makes the infeasible request (point at
the pen, rotate-gesture) → the *same* policy's outputs score infeasible → shield holds
pose / declines, optionally with an LED/UI signal. "A robot that knows what it cannot
do" — collaborative safety, live on hardware, using the framework's own discriminator.

---

## 5. Question 3 — contribution map of everything we have

Ranked S/A/B by (novelty × evidence-in-hand ÷ remaining effort):

- **S1 | Identity ≠ feasibility + Feasibility Critic trio** (this proposal): diagnosis
  pilot done; critic + shield + labeler designs above; needs collection + retrain +
  robot A/B. The only piece with an *algorithmic* claim (feasibility-guided sampling).
- **S2 | One human-human dataset → both interaction directions → unified
  mode-selecting policy** (DONE, robot-verified 08-05): role reassignment (swap which
  hand is "robot"), side-bit conditioning, entry-edge lock, glide bridges for the
  decision point, branch-matrix evaluation. Nobody in the MT-π/X-Diffusion line trains
  *collaboration* (partner-conditioned) from HH data, let alone bidirectional from ONE
  recording. This is the paper's data-efficiency story.
- **S3 | Zero-shot: 0 robot demos** (trained 08-05, offline gates pass, robot test
  pending): if the robot run works even partially, S2 upgrades to "HH data alone
  suffices"; the robot-demo count (0 / 5 / 12) becomes the paper's x-axis.
- **A1 | Counterfactual condition dropout** (DONE + measured): conditions are
  quasi-constant within episodes → policies ignore them (our v1: 72% false release
  both-arms); scope-controlled sentinel dropout manufactures the counterfactual
  (→ 0% false release, reactive waiting robot-verified). General principle for
  conditioned IL; clean ablation already exists. Fold into S2 as a method section.
- **A2 | Soft admission replacing the hard k\* gate** (Design C-1): tiny change,
  directly ablates against the paper's Eq. 4 on public tasks + ours.
- **B | systems findings** (timestamp-uniform resampling forensics, pipelined
  deployment, obs-horizon/velocity-blindness, CRISP damping sentinel): honest
  workshop/appendix material — strengthens reproducibility, not headline claims.

**Recommended paper**: *"Feasibility-Aware Cross-Embodiment Learning from Human-Human
Collaboration"* = S2 (data story + robot-verified handover) + S1 (critic: diagnosis →
fix → three uses) + A1/A2 (method refinements) + S3 (robot-demo-count axis), evaluated
on the handover suite + the §4 feasible/infeasible suite.

---

## 6. Experiment matrix (paper-ready)

**Offline tables**: (i) the §0 dissection table (identity vs feasibility AUROC) for
the stock classifier vs D_feas (expect 0.5→0.9+ on dynamics families, keep 0.98
identity); (ii) admission-weight ablation (hard k* / soft γ / vis-forced) on handover
+ paper-style tasks; (iii) labeler precision vs the analytic IK screen and vs human
annotation on 200 windows.

**Robot A/B** (success rate + safety events + intervention rate, ≥10 trials/cell):
1. unified handover: shield off vs SELECT vs SELECT+VETO — including the scripted OOD
   trigger (mid-task hand withdrawal) that reproducibly caused 40-54 mm bursts.
2. feasible/infeasible request suite (§4): naive co-trained policy vs
   segment-masked policy vs segment-masked + shield. Metrics: feasible-request success,
   infeasible-request abstention rate, false-abstention on feasible requests.
3. zero-shot vs 5-demo vs 12-demo unified policy (S3 axis).

**Baselines**: X-Diffusion stock (hard gate, no shield), FILTERED (analytic-screen
discard), naive co-train, robot-only, + runtime-monitor baseline (STAC-style
consistency check) for the shield comparison.

---

## 7. Risks / open questions

- **Critic shortcut risk**: synthetic negatives are easy to overfit (detect "the
  corruption operator", not infeasibility). Mitigations: wide magnitude ranges,
  held-out corruption families in eval (train without teleport, test on teleport),
  the collected affordance set as the real test.
- **Shield vs stochastic policy**: diffusion outputs are legitimately multimodal;
  SELECT must not collapse diversity (keep τ_soft permissive; measure draw-diversity
  before/after).
- **Reviewer: "why not joint-limit checks?"** — answer is §1-source-2 + §4: we *use*
  analytic checks as free labels for the kinematic class; the affordance class
  (in-limits but non-executable) is invisible to them; the learned, noise-conditioned
  critic unifies both AND plugs into training admission, which no analytic check can.
- **Reviewer: "shield ≈ OOD detection?"** — position against runtime monitors
  (consistency/embedding OOD): D_feas is *supervised by construction* along named
  physical axes, noise-indexed (works mid-sampler), and doubles as the training gate —
  same object serving train-time and run-time; that unification is the claim.
- **Discriminator's own OOD reliability** (our 07-17 DANN lesson, first-hand: a
  discriminator can hit its ideal metric while features are pathological): report
  critic calibration under distribution shift; the held-out-corruption-family eval
  (§7 bullet 1) is the guard.

## 7.5 Related work & novelty verdicts (lit sweep 2026-08-06)

X-Diffusion itself: **arXiv 2511.04671, ICRA 2026** (Cornell); confirmed the
classifier is train-time only ("we only integrate human actions into the loss when
k ≥ k*"); stated future work = scale to uncurated video (orthogonal to us).

**Design B (collected infeasible contrast + labeling) — strongest novelty.** No prior
work collects *paired feasible/infeasible* human data to upgrade an embodiment
classifier into a feasibility critic. Nearest mechanisms: DWBC (ICML 22,
2207.10050) — discriminator-weighted BC, but expert-vs-suboptimal within ONE
embodiment; DemInf (2502.08623) / Re-Mix (CoRL 24) — quality/mixture weighting, not
feasibility. The entire human→robot line (Phantom CoRL 25 — *explicitly restricts
collection to pinch grasps because the robot is a parallel-jaw gripper*; Mirage,
EgoMimic, HumanPlus, DexCap, MT-π, Point Policy) handles infeasibility by **collection
protocol**, i.e. by avoiding it — none detects or models it. Direct argumentative
contrast for the paper.

**Design A (deployment shield / feasibility guidance) — novel, position carefully.**
Discriminator guidance (Kim, ICML 23, 2211.17091) has NO robot-policy application yet.
Nearest: SafeDiffuser (ICLR 25, 2306.00148) — constraints inside denoising but
hand-crafted CBFs, not learned; Learned Viability Filters (2502.19564) — best-of-N +
runtime filter, but character animation with an RL value filter. Runtime monitors
(Sentinel/STAC CoRL 24 2410.04640, FAIL-Detect RSS 25 2503.08558, FIPER NeurIPS 25)
alarm but never veto-and-redraw and carry no embodiment semantics — use as baselines.
Our angle: the filter is the framework's OWN frozen by-product, noise-conditioned so
it natively scores every DDIM step.

**Design C (continuous noise-indexed admission) — ablation-grade, not standalone.**
Bounded between X-Diffusion's hard k* and Ambient Diffusion Policy's (2606.12365)
per-dataset timestep segments. Sell as a component of A/B only.

## 8. Two-week execution checklist

1. `corruptions.py` + retrain critic w/ synthetic negatives; rerun §0 table (expect
   dynamics AUROC → 0.9+). [1-2 d]
2. IK screen auto-labeler; labeler-vs-analytic table. [1 d]
3. Shield in deploy worker + offline replay of the 08-05 OOD logs; then robot A/B-1. [2-3 d]
4. Collect §4 feasible+infeasible sets (with partner conditioning); afford head
   retrain; robot A/B-2. [3-4 d]
5. Soft-admission ablation training runs. [1 d, parallel]
6. Zero-shot robot test (already queued) → S3 cell. [0.5 d]
