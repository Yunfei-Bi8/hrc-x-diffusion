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

### 1.4a CRITICAL BUG FOUND (2026-08-07): the classifier's `HalfUnet1D` never
saw `state_cond` — `cond *= 0` in `models/policy_nets/unet.py` (upstream
"Public release" commit, confirmed via `git log -L`) zeroed the conditioning
tensor before concatenation, for every classifier built on `HalfUnet1D`
(the policy's `ConditionalUnet1D` does not have this line and is unaffected).
Empirically verified: with RNG frozen so only `state_cond` varies, logits
were bit-identical (max abs diff 0.0) before the fix, and diverge (max abs
diff 7.0 on a synthetic probe) after removing the line. **Practical
consequence**: `c_θ(k, A_t^k, s_t)` in the paper's own Eq. 2 was actually
`c_θ(k, A_t^k)` in this codebase the whole time — every classifier we've
trained (including `classifier_handover_uni`, used in the §0 pilot) never
used proprioception. This explains why signal on teleport/whipsaw/jitter was
possible at all despite the bug (those corruptions are visible through the
action tensor `A` itself, not through `s`) but forecloses any
state-vs-action relationship check (e.g. "does this predicted chunk start
near the arm's actual current pose", workspace-boundary proximity) until
fixed. **Fixed**: the `cond *= 0` line is removed. **Breaking**: any
classifier checkpoint trained before the fix must be RETRAINED, not just
reloaded — its FiLM layers' weights for the state-derived channels of
`global_feature` were only ever trained against an all-zero input, so
unzeroing at inference injects effectively untrained noise into those
channels.

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

**Status (2026-08-07): items 0 and 3(partial) below are IMPLEMENTED and
smoke-tested** (variable window length T∈{8,24}, with/without finger stream,
forward + masked multitask loss, deploy-mode no-finger path — all pass).
Not yet wired: corruption loader hook, IK screen, calibration, deploy
integration, and no retraining has happened yet.

0. `models/policy_nets/unet.py` — **DONE**: removed the `cond *= 0` line in
   `HalfUnet1D.forward` (see §1.4a). `ConditionalUnet1D` (the policy) was
   unaffected and untouched.
1. `dataset_utils/corruptions.py` (NEW, ~150 lines): pure functions
   `(chunk_raw, rng, magnitude) → chunk_raw`, family registry, `CorruptionConfig`
   (families, p_corrupt, magnitude ranges, curriculum flag). Unit test: each family
   changes only its intended channels; magnitudes reproducible under seed.
2. `dataset_utils/h5_dataset.py`: hook corruption sampling immediately BEFORE
   `normalize_data` in `__getitem__`; emit `(sample, head_labels, head_mask)`;
   per-head balanced sampler.
3. `models/xdiffusion/hr_classifier.py` — **DONE (implementation differs
   slightly from the original sketch, in a better direction)**: added
   `DualPool1D`, `FingerBranchEncoder`, `FeasibilityCriticConfig`,
   `FeasibilityCritic` as new ADDITIVE classes (composition over the
   existing `HalfUnet1D`, not subclassing `HumanRobotClassifier` — cleaner
   than the originally-sketched checkpoint-migration approach; old
   identity-only checkpoints keep loading unchanged into `HumanRobotClassifier`).
   `FeasibilityCritic.heads = Linear(fused_dim, 3)` where `fused_dim` =
   dual-pooled 7-DoF trunk (§1.4 Layer 2 item 1, folded in directly) +
   optional dual-pooled finger branch with a learned `no_finger_token` for
   modality dropout (§1.4 Layer 3) — both upgrades implemented in the same
   pass rather than as separate later edits. `forward()` is agnostic to `T`
   (fully convolutional + pooled), so the §1.4 Layer 2 window-decoupling
   item (24-step training / 16+8 deploy) needs NO further model code —
   confirmed by smoke-testing T=8 and T=24 through the same module.
   `loss_multitask()` does per-head masked BCE per §1.2.
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

## 4.9 EXECUTED — synthetic-negative validation results (2026-08-07)

`scripts/feascritic_synth_exp.py`, data = handover_bowl_uni (the h5 exports of
29-07-two-human + 07-29-rf/hf-handover; 10,976 train / 453 val 24-step windows,
episode-disjoint). Measured envelope: |Δp|/step p99 = 9.9 mm, euler-rate p99 =
0.128 rad/step. 7 kin families (0% envelope-pass = guaranteed violations) + 3
afford families (100% envelope-pass after gen-filter = kinematically clean) +
synth-transport positive control sharing the synthesis machinery. 5 runs × 3000
steps: main / blind (state zeroed) / channel (state ⊕ input channels) / LOFO
(teleport+whipsaw held out) / LOTO (cap_twist held out).

**Affordance head — validated, strongly.** AUROC@k=0: pen_spin 0.968, cap_twist
1.000, gather 0.979. **LOTO: cap_twist NEVER TRAINED → 0.998** — the concept
transfers across generator families. Dissociation clean both ways: afford head
on kin corruptions ≤0.51 (no false firing); kin head on afford negatives
0.001–0.27 (scores them kinematically "hyper-feasible", correct — they are
quasi-static and in-envelope). Anti-shortcut controls hold: synth_transport
scores feasible on both heads (kin P 0.86, aff P 0.94); still-control passes
(0.89/0.94); real euler spans (6.28/2.9/6.24 rad) cover synthetic amplitudes.

**Kinematic head — 4/7 strong, 3 weak with understood causes.** AUROC@k=0
(main→channel): speed 0.949→0.966, whipsaw 0.981→0.987, euler_rate 0.998,
flutter_fast 1.000 — vs the identity classifier's 0.49–0.55 on the same axes
(08-06 pilot): the missing-supervision diagnosis confirmed. Weak: teleport
0.696→0.748, jitter 0.593→0.643, start_jump 0.735→**0.949 (channel)**.

Three root causes, each evidenced:
1. **FiLM cannot do state–action comparison**: blind ≈ main on start_jump
   (0.723 vs 0.735) — even unblocked, the FiLM path goes unused; concatenating
   the state as extra INPUT CHANNELS (action_dim 7→14, `--state-channel`) is
   what operationalizes it (0.949). → ADOPTED as the §1.4 model modification.
   (Also the honest refinement of §1.4a: the cond fix is necessary but NOT
   sufficient.)
2. **iid jitter is confounded with the forward-diffusion noise itself** — a
   noise-conditioned critic is trained to look THROUGH additive iid Gaussian
   perturbations; violations that mimic the noising distribution are its
   structural blind spot. Partial fix: low-k importance sampling (at k=0 the
   model knows noise≈0, so observed jitter must be data). Deploy-relevant
   bursts are STRUCTURED (whipsaw 0.98+), so the practical impact is bounded.
3. **Teleport signal exists only at low k** (a 40 mm jump drowns in ~100 mm/dim
   DDPM noise by k≈20-40); uniform-k training dilutes the pressure. Same low-k
   sampling fix.

**LOFO caveat**: held-out whipsaw transfers poorly (0.981 trained → 0.595
held-out) — the kin families as currently designed do not yet induce one
general "continuity violation" concept; broaden family diversity + magnitude
curriculum before claiming operator-independence.

Identity head sanity: preserved (mean P clean_robot 0.87 vs clean_human 0.02).
k-profiles decay as the ambient premise predicts (cap_twist holds 1.000 to
k=40; whipsaw 0.98→0.68).

**Gate verdict: GO.** Both target discriminations demonstrated on the real
handover data + synthetic negatives; the state-channel variant is the adopted
architecture; jitter/teleport weaknesses have understood causes and a concrete
training-recipe fix; the real-collection E2 remains the true affordance test.

## 4.10 EXECUTED — deployment-domain validation (2026-08-07, round 2)

Answers three challenges raised against §4.9: (a) is noise-step training even
right for a deployment-only shield? (b) policy-GENERATED chunks differ from
data windows — does discrimination survive? (c) more structure changes needed?

**E-A, k-sampling ablation** (train-k ∈ {uniform, zero, low}, all
state-channel): uniform-k training was exactly the weakness — at the shield's
operating point (k=0), k=0-only / low-k training lifts teleport 0.748→0.979 /
0.970 and jitter 0.643→0.994/0.990, everything ≥0.97 (afford ≥0.999).
**Verdict: for the deployment shield, train at k≈0; the noise axis is only
needed when the same critic also serves training-time admission.** Two-regime
answer; low-k (50% k=0 + 50% U[0,21)) is the adopted default (`klow`).

**E-B, real-policy tests** (uni_v3, the actually-deployed policy;
[16 executed ‖ 8 generated] composite windows = the deploy scoring shape):
- *Cell 1*: generated vs real-continuation composites are INDISTINGUISHABLE
  to the critic (texture-AUROC 0.498-0.501); good outputs pass 94-96% at the
  95%-TPR-on-real threshold. No generation-texture penalty exists.
- *Cell 3 vs 4 (attribution)*: corruption-detection drop on generated chunks
  (0.97-0.99 → 0.61-0.88) is reproduced EXACTLY on real windows when the
  corruption is confined to the last-8 slice → cause = slice-confinement
  (training only used full-window corruptions), NOT policy texture. Fix =
  `--corrupt-span mixed` (50% full / 25% suffix-8 / 25% random suffix):
  `kmix` recovers generated-domain detection to 0.86-0.99 (teleport 0.91,
  jitter 0.87, euler 0.96, flutter 0.99, whipsaw 0.98, seam_jump 0.95) while
  keeping the full-window battery ≥0.97. Both fixes are TRAINING-RECIPE, not
  architecture: no further model change needed beyond §4.9's state-channel.
- *Cells 2/5 (honest negative)*: the 08-05 whipsaw does NOT reproduce
  offline — open-loop burst rate 0.3%, and even 20-cycle closed-loop
  anchor-following rollouts drift smoothly (jump p95 4.2 mm). The live
  failure evidently required real-world condition flicker; robot A/B remains
  the test for that scenario. Directional sensitivity is present:
  corr(peak |Δp|, P_kin) = −0.354 across rollouts.

**E-C, laundering test**: a policy trained PURELY on synthetic
affordance-infeasible episodes (28 pen-spin + 16 cap-twist), then sampled:
its own generated chunks score P_aff 0.244 (data windows 0.311; uni_v3
feasible-policy outputs 0.902) — **signatures are NOT laundered by
diffusion-policy learning + DDIM sampling** (euler churn even amplified,
0.0085→0.0235 rad/step; grasp cycles slightly attenuated 0.44→0.31
transitions/chunk but present). Single-chunk veto at P_aff<0.5: 76.4%
(contaminated) vs ~0% (feasible policy); M-of-N cycle logic pushes sustained
behavior to ~certain veto. P_kin stays 0.995 on both — the dissociation
holds in the deployment domain.

Artifacts: `runs/feascritic_synth/{kzero,klow,kmix}.json`, `eb_deploy.json`,
`ec_contam.json`; contaminated policy `runs/policy_feas_contam`.
**Adopted recipe: state-channel + low-k training + mixed-span corruptions
(= the `kmix` configuration).**

## 4.11 EXECUTED — jar-lid-opening zero-shot probe (2026-08-10)

Proposed first real collection: hand-1 unscrews a jar lid (retargeted action),
hand-2 holds the jar (condition-only; invisible to the critic). Probe: four
synthetic proxies of the REAL motion, deliberately different from the trained
cap_twist family (sawtooth work/return strokes, open-on-return, roll/pitch
wobble, progressive z-rise, finger-walk periods, final lift-off), scored by
the FROZEN kmix critic — no retraining:

| style | AUROC (relevant head) | either-head veto |
|---|---|---|
| tight-lid twist-regrasp loop | afford **1.000** (kin silent — correct) | 100% |
| + HaMeR-level tracking noise (3 mm / 0.03 rad) | afford **1.000** | 100% |
| loose-lid finger-walk (period 6-9) | afford **1.000** | 100% |
| fast fingertip spin (period 2-4) | kin 0.706 + afford 0.773 (bandwidth band → kin territory, as designed) | 96.4% |
| final lid-LIFT phase (executable) | passes both | 3.6% (≈ the 5% design FPR) |

The heads PARTITION the style space as designed, and the critic is
phase-precise within the task: unscrew segments flagged, the genuinely
executable lift-off passes — the §3 segment-masked-co-training story
demonstrated zero-shot. Jar-opening is confirmed as the #1 collection
candidate: (i) it sits in the strongest-validated signature class
(grasp-oscillation-while-holding + rotation phase-lock); (ii) the
infeasibility argument is airtight from trace semantics alone —
**release-while-must-hold**: fingers keep the lid controlled during a
regrasp, the retargeted binary gripper "open" drops it — effectiveness
failure, no kinematic-limit hand-waving needed; (iii) trace-VISIBLE, so no
finger branch required for v1; (iv) natural HRC framing (partner provides
force closure = genuine collaboration; deploy demo = offer-jar → veto/decline
vs open-palm → handover); (v) built-in feasible phase (lift) for the
segment story. Collection notes: jar held between the two cameras (occlusion
hygiene); instruct NATURAL unscrewing (fast fingertip spins land in the kin
band instead — still vetoed, but attribution muddies); matched feasible set
on the same scene (pick/place/hand-over the same jar); everything else per §2.

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
