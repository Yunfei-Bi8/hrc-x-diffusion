# Feasibility Critic — detailed implementation plan (2026-08-06)

Companion to `RESEARCH_PROPOSAL_FEASIBILITY.md`. This document turns Design B (the
strongest-novelty direction per the 08-06 literature sweep: *paired feasible/infeasible
two-human collection upgrading the embodiment classifier into a feasibility critic*)
into an executable engineering plan: mechanism, data spec, code changes, training
recipe, and — in full detail — the evaluation protocol that establishes whether the
critic can actually tell feasible from infeasible actions.

---

## 1. Mechanism: how the critic works

### 1.1 Inputs, architecture, and the noise axis

The critic keeps the exact I/O contract of the existing classifier
(`models/xdiffusion/hr_classifier.py`), so every consumer (training admission,
labeler, deploy shield) is drop-in:

- **Action chunk** `A ∈ R^{8×7}`: 8 future steps of `[ee_pos(3), ee_euler(3),
  gripper(1)]`, min-max normalized to [−1,1] with the critic's own dataset stats
  (stored in its run dir; consumers MUST load these stats, not the policy's — a
  silent stats mismatch is the #1 foreseeable integration bug, add an assert).
- **State** `s ∈ R^7`: the proprio row at observation time (same 7 dims).
- **Diffusion step** `k ∈ [0, 101)`: forward-noised inputs
  `A^k = √ᾱ_k·A + √(1−ᾱ_k)·ε`, same squaredcos_cap_v2 schedule as the policy;
  `k` enters via the sinusoidal timestep embedding (dim 256). Noise is applied to
  both `A` and `s` (matching `unified_forward`).
- **Trunk**: `HalfUnet1D` — temporal Conv1d stack (kernel 5, down_dims
  [256,512,1024]) FiLM-conditioned on (timestep-embedding ⊕ flattened state).
- **Heads** (the change): GAP over time → `Linear(1024, 3)` = three sigmoid heads
  `[identity, kin_feas, afford_feas]` instead of the single identity logit.

Why temporal convolution is the right inductive bias: finite differences are linear
convolutions, so a kernel-5 conv stack natively computes per-step velocity,
acceleration, jerk, sign-alternation and grasp-transition patterns at multiple
temporal scales — exactly the statistics along which infeasible motion differs from
feasible motion. The 08-06 pilot showed the failure of the stock classifier is a
*missing-supervision* problem, not a capacity problem: the same trunk already scores
0.67/0.77 AUROC on grasp-flutter/euler-spin — the only two corruption axes along
which its two training classes happen to differ. Give it classes that differ along
velocity/continuity axes and gradient descent will find those features immediately
(they are the largest-margin ones).

Why keep the noise axis (three reasons, each load-bearing):

1. **Deployment alignment**: during DDIM sampling the shield scores intermediate
   states at their actual noise level `k` (or the x̂₀ estimate at k≈0) with zero
   distribution shift — the critic trained on exactly such noised chunks.
2. **The k\* machinery generalizes**: X-Diffusion's minimum indistinguishability
   step (Eq. 3) becomes a *minimum feasibility step*
   `k*_feas(A) = min{ k : P_feas(A^k, s, k) ≥ 0.5 }` — "how much abstraction until
   this human behavior is robot-usable" — now measured against ground-truth
   feasibility labels rather than embodiment identity as a proxy.
3. **Regularization**: at high k the task is provably impossible (distributions
   merge — the Ambient Diffusion premise), which pushes the net toward robust
   low-k features instead of brittle shortcuts.

### 1.2 Training classes (three negative sources, one positive pool)

| head | positives (y=1) | negatives (y=0) | cost |
|---|---|---|---|
| `identity` | robot windows | human windows | free (existing) |
| `kin_feas` | clean robot + IK-screen-passing human | (a) synthetic corruptions of robot/feasible-human windows; (b) IK-screen-failing human windows | free |
| `afford_feas` | robot + FEASIBLE-task human | INFEASIBLE-task human (new collection, §2) | ~2 half-days |

Per-head masked BCE: `L = Σ_h λ_h · BCE(head_h, y_h) · defined_h` — each sample
carries labels only for the heads where its label is meaningful (a synthetic
corruption has `kin=0` but `afford` undefined → masked). λ all 1.0 initially.
Balanced per-head sampling via the existing `balanced_sampling_weights` machinery
(generalized to per-head pools). Timesteps `k ~ U[0,101)` as in the original (keep
comparability; low-k importance sampling is an ablation, not the default).

Synthetic corruption families (in `dataset_utils/corruptions.py`, applied in the
loader on RAW-space chunks *before* normalization, probability `p_corrupt` per
sample, magnitude randomized per draw):

| family | operator | magnitude range |
|---|---|---|
| speed | `p₀ + m·(p−p₀)` | m ∈ [1.5, 6] |
| teleport | step jump at random index | [2, 10] cm |
| jitter | iid pos noise | [5, 25] mm |
| whipsaw | alternating ±y offsets | [15, 60] mm |
| euler_spin | yaw ramp | [0.1, 0.5] rad/step |
| grasp_flutter | 0/1 toggling | 2–7 transitions per window |
| smooth-violation | resample with dropped frame (aliasing) | 1–2 frames |

Rationale for ranges: lower bounds sit just above our real deploy envelope (the
policy's verified-good sessions must not be inside the negative support); upper
bounds are the pilot magnitudes that today score at chance. A magnitude-sweep at
eval (§4, E1c) locates the learned boundary.

`afford_feas` label noise: labels are inherited from the task segment, so
still/transport sub-windows inside an infeasible episode are locally feasible —
weak labels. v1 mitigations: label smoothing 0.9/0.1 on afford negatives + report
segment-level metrics (robust to boundary noise) alongside window-level. Upgrade
path if window metrics disappoint: multiple-instance learning with attention
pooling over the episode's windows (Ilse et al., ICML 2018) — the bag label is
clean even when instance labels are not.

### 1.3 Outputs and calibration

Raw balanced-BCE sigmoids are not calibrated probabilities. Post-hoc **temperature
scaling** (Guo et al., ICML 2017) per head on held-out val, fit on a k-grid
{0,5,10,20,40} → `T_h(k)` table (linear interp between grid points), stored in the
run dir; report ECE before/after. Consumers:

- **Shield score**: `f = min(kin_feas, afford_feas)` at k=0 (conservative AND;
  identity head deliberately EXCLUDED — the pilot proved identity rejects benign
  human-styled outputs). 2–4 noise draws averaged (at k=0, ᾱ≈1, draw variance is
  tiny — measure and minimize draws for latency).
- **Labeler profile**: slide over a demo, per-frame
  `feas(t) = mean over windows covering t`, stored as h5 keys
  `crit_kin`, `crit_afford`, plus `k*_feas` per window.
- **Soft admission weight**: `w = P_feas(A^k, s, k)^γ` replacing the hard
  `1{k ≥ k*}` gate in `compute_xdiffusion_loss_masks()`.

### 1.4 Is the architecture sufficient? (added after 08-06 review)

The honest three-layer verdict on "can a 7-DoF-input HalfUnet tell feasibility":

**Layer 1 — kinematic/dynamic violations: architecture is sufficient as-is.**
Speed, teleport, jitter, whipsaw, spin are per-step difference statistics; temporal
Conv1d computes multi-order differences natively, and the pilot already proves this
trunk extracts such features when supervision demands it (flutter 0.67 / spin 0.77
with zero feasibility training). The pilot's chance-level results were a
missing-supervision problem, not a capacity problem.

**Layer 2 — cheap architectural upgrades (adopt all three).**
- **Dual pooling**: replace GAP with concat[avg-pool, max-pool] → `Linear(2048, 3)`.
  A single-step teleport produces one large temporal feature that GAP dilutes by
  1/8; max-pool preserves peak violations. One-line change.
- **Decouple the critic window from the policy's pred_horizon** (the important
  one): 8 steps @15 Hz = 0.53 s, but a cap-twist regrasp cycle or pen-spin period
  runs 1–2 s — inside one 8-step window, a regrasp fragment ("opens while wrist
  turns") locally resembles a feasible release. Train the critic on **24-step
  windows** (1.6 s). At deployment, score `[last 16 executed steps ‖ 8 predicted]`
  — the executed history is free in the deploy loop, and the concatenation puts
  **cross-chunk discontinuities** (whipsaw across chunk boundaries — invisible to
  any per-chunk scorer) inside the receptive field. The policy's horizon is
  untouched; only the critic's input length differs. For training-admission use,
  score the 24-step window ending at the policy sample's window.
- **3-member ensemble** (3 seeds × ~30 min): score disagreement = epistemic
  uncertainty, letting the shield separate "confidently infeasible" from "never
  saw anything like this" (distinct fallback policies; also the concrete answer to
  the discriminator-reliability/DANN concern).

**Layer 3 — the input-representation limit (no architecture on 7-DoF can fix
this).** Affordance infeasibility splits by whether its signature *survives
retargeting to the wrist+pinch trace*:
- *Trace-visible*: pen-spin (pos–euler decorrelation), cap-twist (grasp
  oscillation while "holding"), cloth folding (micro-regrasp flutter) — learnable
  from 7-DoF.
- *Trace-INVISIBLE*: chopstick pickup (wrist+pinch trace ≈ a perfectly feasible
  pinch-and-lift; the infeasibility lives in tool-mediated finger contact) and
  multi-key chords (wrist trace ≈ feasible hover+poke; the simultaneity of spread
  fingers is gone). Information-theoretically undetectable from the retargeted
  trace — **because retargeting itself destroyed the evidence**.

Fix = **dual-input critic**: shared temporal trunk on the 7-DoF core + an optional
**finger-feature branch** (from the HaMeR keypoints we already have upstream of
retargeting: 5 fingertip positions relative to the wrist frame, 15-dim/frame,
encoded by a parallel conv stack, fused before the heads), trained with
branch-dropout so the model works with or without it. The **labeler/training-gate
runs WITH fingers** (human data always has them); the **deployment shield runs
WITHOUT** (policy outputs have no fingers — and shield-relevant failures are
dynamics/OOD, which are 7-DoF-visible). This also cleans up the concept the paper
should separate anyway:

> infeasible = **executability failure** (arm cannot run the trajectory — 7-DoF
> critic) ∨ **effectiveness failure** (arm can run it, but the retarget lost the
> behavior's essence, so running it would not accomplish the task — finger-level
> critic; equivalently a *retargetability* critic).

Chopsticks/chords are effectiveness failures. Prediction registered for E2: the
7-DoF critic fails on exactly those two tasks and the finger-branch critic
recovers them — the per-task visibility table then *measures* which
infeasibilities survive retargeting, which no prior work quantifies.

### 1.5 Data level vs deployment level — and can the policy's outputs fool the critic?

The critic acts at BOTH levels, with a strict division of labor by failure source:

| failure source | example | caught at | by |
|---|---|---|---|
| policy would LEARN fine infeasible execution from a demo | pen-spin fine cycle | DATA (admission: low-k supervision blocked) | 7-DoF heads |
| infeasibility whose evidence retargeting destroyed | chopsticks, chords | DATA only (finger-branch labeler masks segments) | finger branch |
| runtime OOD generation, never in any data | 08-05 whipsaw bursts on odd hand timing | DEPLOY shield | kin head |
| residual infeasible mode that slipped admission | attenuated flutter | DEPLOY shield (+ refresh, below) | kin/afford heads |
| dynamically-fine but task-wrong output | feasible-looking pinch in a chopstick context | NOT catchable at deploy from 7-DoF — must be prevented at the data level | (division of labor) |

Three facts anchor this:
1. **The shield is not made redundant by perfect data curation**: our measured
   whipsaw failure came from a policy trained on ENTIRELY feasible data — it is an
   OOD-conditional generation failure, which no training-data filter can prevent.
2. **Diffusion policies preserve modes** (their defining property vs MSE-BC): a
   policy trained on flutter/decorrelation signatures reproduces them in samples,
   so trace-visible infeasible modes remain critic-visible at deploy. Caveat:
   few-step DDIM + small data can ATTENUATE high-frequency signatures →
   **E4-0 (added below)** evaluates the critic on policy-GENERATED chunks, not
   only data chunks; if separation degrades, one **adversarial-refresh round**
   (add generated negatives, retrain critic once) closes the gap.
3. **No arms race by construction**: the policy receives no gradient from the
   critic (admission selects supervision, it does not optimize the policy to score
   well; the shield is inference-only), so nothing pushes the policy toward
   critic-fooling outputs. Exception: the GUIDANCE variant actively climbs the
   critic's score and can exploit its flaws — bound the guidance weight and gate
   it on ensemble agreement.

Remaining known gap, accepted for v1: EEF-only state hides joint configuration
(near singularities the EEF-velocity limit is configuration-dependent; the same
pose can be fine elbow-up and near-singular elbow-down). Our tabletop workspace
stays in one IK branch far from singularities, so the approximation holds here;
V2 = append q ∈ R^6 to the conditioning (state 7→13) for generality. Images/SAM3
masks (object context) remain a V2+ extension via the repo's intentionally-stubbed
`ImageHumanRobotClassifier`.

---

## 2. The paired feasible/infeasible collection (data spec)

Rig: unchanged (two calibrated Orbbecs, HaMeR + SAM3, side-locked two-hand
pipeline, `--resample-hz 15` timestamp-uniform export).

**Feasible set (~40 ep)**: hand-1 pick/place/push/handover of tabletop objects;
hand-2 gives the *request* gesture (open-palm reach) or is absent. Largely covered
by the existing 29-07 pool (136 episodes) — top up with object variety only.

**Infeasible set (~40-60 ep)** — selection rule: *retargets to IN-LIMITS EEF motion
that a 2F-85 parallel gripper cannot execute* (out-of-limits motion is caught free
by the analytic screen; do not spend collection time on it):

| # | task | signature in 7-DoF trace | visible to 7-DoF critic? |
|---|---|---|---|
| 1 | in-hand pen spin | wrist quasi-static, euler churns (pos–euler decorrelation) | YES |
| 2 | cap-twist with regrasp cycles | grasp oscillation while "holding" + rotation (needs the 24-step window) | YES |
| 3 | cloth fold with fingers | micro-regrasp flutter, gathering micro-motions | YES |
| 4 | card slide-off-edge + flip | near-zero net wrist motion + sudden flip euler | PARTIAL |
| 5 | chopstick pickup | trace ≈ feasible pinch-and-lift | **NO — finger branch required** |
| 6 | multi-key chord press (2-3 keys, fingers spread) | trace ≈ feasible hover+poke | **NO — finger branch required** |

Tasks 5–6 are kept deliberately: they are the registered-prediction cells for the
§1.4 dual-input ablation (7-DoF critic should fail there; finger-branch critic
should recover them). Exclusions: single-button poke with a closed empty gripper
is FEASIBLE — not collected; "press while holding an object" is task-semantic
infeasibility — collected but analyzed separately, outside headline metrics.

Partner conditioning: each infeasible episode's hand-2 performs the matching
request gesture (point at pen / rotate-fist "open it" / extend cloth), so
feasibility is *interaction-conditioned* — at deploy the same policy conditioned on
open-palm produces shield-passing chunks; conditioned on point-at-pen produces
chunks the shield vetoes. The pairing gives the causal contrast for the robot demo.

Hygiene (each item exists because a shortcut died by it before):
- Same subjects, table region, objects VISIBLE on the table in both sets; same
  session lighting; counterbalanced recording order.
- Grasp channel for infeasible episodes: keep the RAW pinch-threshold signal (no
  object-contact cleaning) — the messy grasp IS the signature; document this
  asymmetry.
- Start-pose matching between sets (start-pose probe in E2 must be at chance).
- Per-episode `feas_label` attr + per-frame `feas_mask` (lead-in/out frames marked
  feasible-still) in a new exporter `export_xdiffusion_h5_feas.py` (thin derivative
  of the two-human exporter; per-task SAM3 concept prompts: "pen", "bottle",
  "chopsticks", "card", "cloth", "keyboard").

Volume: 40 ep × ~35 s × 15 Hz ≈ 21k frames ≈ 10k windows (stride 2) per class —
ample for a ~20M-param trunk that already trains on 50k+ windows.

## 3. Code changes (file-level)

1. `dataset_utils/corruptions.py` (NEW, ~150 lines): pure functions
   `(chunk_raw, rng, magnitude) → chunk_raw`, family registry, `CorruptionConfig`
   (families, p_corrupt, magnitude ranges, curriculum flag). Unit test: each family
   changes only its intended channels; magnitudes reproducible under seed.
2. `dataset_utils/h5_dataset.py`: hook corruption sampling immediately BEFORE
   `normalize_data` in `__getitem__`; emit `(sample, head_labels, head_mask)`;
   per-head balanced sampler.
3. `models/xdiffusion/hr_classifier.py`: `FeasibilityCritic(HumanRobotClassifier)`
   with `Linear(1024, 3)`, `loss_multitask()` (masked BCE), backward-compatible
   single-head checkpoint migration (key remap for the old `classifier.2` weight).
4. `scripts/ik_screen.py` (NEW): v1 = EEF-space screen (workspace box incl. table
   clearance, per-step velocity/accel vs our deploy limits, orientation
   reachability heuristic); v1.1 = closed-form UR10e IK, joint-branch continuity +
   joint velocity limits + wrist/elbow singularity margins. Writes per-window
   `ik_feas` h5 key (label source for `kin_feas`, baseline for E1/E2).
5. `scripts/train_critic.py` (extends `train_classifier.py`): three-pool assembly,
   per-head wandb metrics, **per-epoch AUROC@k∈{0,10,20,40} on val** — reuse the
   existing `noise_sweep` harness already in the classifier run dirs.
6. `scripts/calibrate_critic.py` (NEW, ~80 lines): temperature fit per head per
   k-grid point; writes `calibration.json`; plots reliability diagrams.
7. `scripts/critic_label_pass.py` (NEW): offline labeler — slide windows over any
   h5 pool, write `crit_*` keys + per-demo profile PNGs.
8. Deploy (`deploy_xdiffusion_ur10e.py` worker): `--shield-run-dir --shield-mode
   {off,select,veto,both} --shield-thr --shield-draws` per proposal §2 — consumes
   the critic run dir (ckpt + stats + calibration).

---

## 4. Evaluation: how we establish that the critic can tell feasible from infeasible

Design principle: every metric must separate **"learned the concept of
infeasibility"** from **"memorized the negative-generation operator"**. Four
layers, cheap→expensive.

### E1 — kinematic head, synthetic ground truth

- **E1a, in-distribution**: clean robot val vs each trained corruption family,
  AUROC at k∈{0,5,10,20,40}. Bar: ≥0.95 at k≤10 (vs today's 0.49–0.55 — the
  before/after table is the paper's first result).
- **E1b, leave-one-family-out (LOFO)** — the generalization test: retrain with one
  family held out entirely; test on it. Full 7×7 matrix (train-set × test-family).
  Transfer happens iff the net learned shared statistics (velocity/accel/continuity)
  rather than per-operator signatures. Bar: mean held-out AUROC ≥0.85. Precedent
  for this eval style: corruption-robustness benchmarks (Hendrycks & Dietterich,
  ICLR 2019) — same logic, our domain.
- **E1c, magnitude sweep / boundary location**: AUROC and P_feas vs corruption
  magnitude (speed 1.1×→6× etc.). Overlay the analytic limit (UR10e velocity/accel
  envelope): does the critic's 50% crossing sit at the physical boundary? If yes —
  the money figure: *the learned boundary coincides with the physical one without
  ever being told the limits*.
- **E1d, still control**: frozen chunks must score FEASIBLE (they are). Guards
  against "low activity = infeasible" collapse.
- Statistics: windows cluster by episode → cluster bootstrap CIs (1000 resamples,
  episode-level), n ≥ 300 windows/cell.

### E2 — affordance head, collected ground truth (the novel claim's test)

Splits, strictly episode-disjoint, three nested generalization levels:
1. **held-out episodes** (same tasks) — sanity, expect high;
2. **leave-one-task-out (LOTO)**: train afford on {pen-spin, cap-twist, cloth,
   chords}, test on {chopsticks}; rotate. Tests whether "gripper-infeasible" is one
   learnable concept across dexterous behaviors. Bar: window-level ≥0.80,
   segment-level (mean-pool over episode segment) ≥0.90.
3. **held-out subject** (if ≥2 demonstrators): person-invariance.

Confound controls (each kills a specific rival explanation):
- **Start-pose probe**: logistic regression on `s` alone → must be ≈0.5 AUROC
  (else the sets are separable by *where*, not *what*).
- **Moving-windows-only AUROC**: excludes still windows from both classes → the
  discrimination is not "stillness detection".
- **Speed-matched AUROC**: bin windows by mean |Δp|, compare within bins
  (propensity-matched) → the discrimination is not "dexterous stuff moves
  slower/faster". Bar: ≥0.75 survives matching.
- **Human agreement**: 200 random windows, 2 annotators watching the source video
  ("could the 2F-85 execute this snippet's retargeted motion and achieve its
  purpose?"); report critic–human Cohen's κ vs human–human κ. Grounds the
  construct; expected κ_hh itself < 1 (the boundary is genuinely fuzzy — report,
  don't hide).

Baselines the critic must beat (reviewers will demand all four):
1. **Stock identity classifier** (= our pilot numbers: ~0.5 dynamics, 0.67/0.77
   flutter/spin) — the "before".
2. **Analytic IK screen** — perfect on out-of-limits, ~0.5 on affordance (by
   construction) — shows why learning is needed at all.
3. **Hand-crafted features + logistic**: {mean/max |Δp|, jerk energy,
   grasp-transition count, euler spectral entropy, pos–euler correlation} → if 5
   features match the deep critic, the claim weakens; expect the critic to win on
   LOFO/LOTO transfer and speed-matched cells specifically.
4. **Unsupervised OOD detector** (kNN distance in trunk feature space fit on
   feasible-only data, Sun et al. ICML 2022): does supervised contrast beat pure
   OOD? Expected: OOD flags affordance negatives partially but also flags
   novel-yet-feasible motion (poor precision) — this ablation is what justifies
   collecting negatives at all.

Plus the **dual-input ablation** (registered predictions from §1.4): per-task
AUROC for the 7-DoF-only critic vs the finger-branch critic. Expected pattern:
tasks 1–3 high for both; tasks 5–6 ≈ chance for 7-DoF-only, recovered by the
finger branch. This table *measures which infeasibilities survive retargeting* —
the quantity no prior human→robot work reports.

### E3 — labeler validation on data with KNOWN defects (free, already owned)

Run the critic over the raw pre-denoising HaMeR tracks (track1_raw backups) and
the 07-29 pre-resample exports. We possess frame-level ground truth for both
defect populations from our own forensics (Hampel spike lists from
`filter_hand_tracks.py`; the 27–32% stall-teleport transitions from the time-base
audit). Metric: precision/recall of `crit_kin < τ` against those lists. Bar:
P ≥ 0.8 at R ≥ 0.8. This is real-world uncurated-data validation with zero new
collection — and directly rehearses X-Diffusion's own stated future work
(uncurated internet video curation).

### E4 — deployment shield (ties to proposal §2)

- **E4-0, generated-chunk separation (the "does generation launder the
  signatures?" test)**: train a deliberately NAIVE policy on the mixed pool
  (infeasible data admitted everywhere), sample chunks in infeasible-context
  conditions, score them vs feasible-context samples and vs the underlying data
  windows. Measures signature attenuation through the sampler. If AUROC drops
  materially vs the data-level number → one adversarial-refresh round
  (generated negatives added to critic training) and re-measure.
- **Offline replay**: logged 08-05 OOD sessions (mid-task hand withdrawal →
  40–54 mm/cycle whipsaw bursts) vs verified-good sessions: veto rate on burst
  cycles ≥90% at ≤1% veto on good cycles (threshold from the calibrated score).
- **Robot A/B**: unified handover with scripted OOD trigger; shield off/select/
  veto; metrics: burst events, intervention rate, task success preserved.
- **Robot demo (the paper's demo)**: partner open-palm → executes handover;
  partner points at pen / rotate-gesture → policy output scores infeasible →
  shield holds pose (declines). Same policy, same shield, opposite outcomes,
  chosen by interaction context.

### The k-profile figure (mechanism validation)

For every class, AUROC(k) must decay to 0.5 as k→101 (distributions provably
merge). The decay rate orders classes by distance-from-feasible-manifold:
teleport should stay separable to higher k than mild jitter. Plot the "feasibility
persistence spectrum" — AUROC vs k per class — and per-sample
`k*_feas` histograms per task. This validates the paper's Fig. 3 narrative with
ground-truth labels instead of eyeballed side-grasp examples, and is the
quantitative object that did not exist before.

---

## 5. Reference papers

Method core:
- Pace et al., **X-Diffusion**, arXiv 2511.04671, ICRA 2026 — base framework (Eq. 2 classifier, Eq. 3 k*, Eq. 4 gated loss).
- Daras et al., **Ambient Diffusion**, NeurIPS 2023, arXiv 2305.19256 (+ **Ambient Diffusion Omni**, arXiv 2506.10038; + **Ambient Diffusion Policy**, arXiv 2606.12365) — noise-level-gated learning from corrupted data; the classifier lineage.
- Kim et al., **Refining Generative Process with Discriminator Guidance**, ICML 2023, arXiv 2211.17091 — the sampling-time guidance term we transplant.
- Xu et al., **DWBC**, ICML 2022, arXiv 2207.10050 — discriminator-weighted BC (mechanism precedent for soft weights; same-embodiment quality, not feasibility).
- Guo et al., **On Calibration of Modern Neural Networks**, ICML 2017, arXiv 1706.04599 — temperature scaling.
- Ilse et al., **Attention-based Deep MIL**, ICML 2018, arXiv 1802.04712 — upgrade path for weak segment labels.

Evaluation methodology:
- Hendrycks & Dietterich, ICLR 2019, arXiv 1903.12261 — held-out-corruption evaluation logic.
- Sun et al., **kNN-OOD**, ICML 2022, arXiv 2204.06507 — unsupervised OOD baseline.

Positioning / baselines:
- Agia et al., **Sentinel/STAC**, CoRL 2024, arXiv 2410.04640 — runtime consistency monitor (deployment baseline).
- **FAIL-Detect**, RSS 2025, arXiv 2503.08558 — failure detection without failure data (baseline).
- Xiao et al., **SafeDiffuser**, ICLR 2025, arXiv 2306.00148 — hand-crafted CBFs inside denoising (contrast: ours is learned).
- **Learned Viability Filters**, arXiv 2502.19564 — nearest best-of-N runtime filter (animation, RL value semantics).
- **Phantom**, CoRL 2025, arXiv 2503.00779 — restricts collection to pinch grasps (the collection-protocol foil); similarly Mirage (RSS 2024, 2402.19249), EgoMimic (ICRA 2025, 2410.24221), MT-π (2501.06994), Point Policy (2502.20391).
- Hejna et al., **DemInf**, arXiv 2502.08623 — data-quality curation comparison.

## 6. Order of execution & compute

| step | what | duration | depends on |
|---|---|---|---|
| 1 | corruptions.py + critic heads + kin-head training on existing pools | 1–2 d | — |
| 2 | E1a/b/c/d full battery (LOFO = 7 retrains ≈ 30 min each) | 0.5 d | 1 |
| 3 | ik_screen v1 + labels + calibration | 1 d | 1 |
| 4 | infeasible-set collection + export + QC | 2 half-days | — (parallel) |
| 5 | afford head retrain + E2 battery + human κ (200 windows) | 1.5 d | 3,4 |
| 6 | E3 forensic validation on raw tracks | 0.5 d | 3 |
| 7 | shield in worker + E4 offline replay | 1 d | 5 |
| 8 | robot A/B + demo | 1–2 d | 7 |

Every training ≤ ~30 min on the current GPU (classifier-sized net, existing
loaders). Total ≈ 2 weeks including robot time, matching proposal §8.
