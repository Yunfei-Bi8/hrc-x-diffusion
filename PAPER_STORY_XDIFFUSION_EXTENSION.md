# Runtime Admission Control for Cross-Embodiment Diffusion Policies
## The Full Paper Story (v2, 2026-08-24)

*This version supersedes the 2026-08-20 memo (v1 remains in git history). It integrates the
2026-08-24 four-thread literature research: verified lineage (Ambient Omni → Ambient
Diffusion Policy → X-Diffusion), corrected citations, scoped novelty claims, the final
narrative skeleton, and the venue plan.*

Companion documents:
- `SAFETY_LAYER_MATH_FULL.md` — rigorous notation + full proofs (P1, L1–L4, T1–T5, A1).
  Theorem numbers cited below refer to that file.
- `README_TUM_DTE_FEASIBILITY.md` — deployed-system mechanics and deploy command.
- `README_TUM_FEASIBILITY_SHIELD.md`, `README_TUM_JAR_UNIFIED.md` — hardware session records.

---

## 0. The story in one paragraph

A whole lineage of diffusion-model training methods — Ambient Diffusion Omni (NeurIPS 2025),
Ambient Diffusion Policy (Tedrake lab, 2026), and X-Diffusion (Pace et al., ICRA 2026) — rests
on a single quantity: the minimal forward-diffusion noise level at which a data point becomes
statistically indistinguishable from the target distribution. Each of these papers computes
this quantity with a noise-conditional classifier, uses it to decide **which data may enter
training**, and then discards the classifier; Ambient Diffusion Policy's project page states
it outright: *"Inference: no changes."* We name this quantity the **admission time**, and we
show that the criterion these methods enforce at the entrance of the policy's lifecycle is
still needed at its exit: a trained cross-embodiment policy remains a stochastic sampler that
**re-emits embodiment-infeasible actions at execution** (measured: 12/12 unshielded rollouts),
and the training-time alternative — deleting the infeasible demonstrations — produces not
safety but undefined behavior (measured: 25% physical penetration). We therefore extend
admission control from training time to execution time: a **one-class diffusion time
estimation (DTE) network** estimates each emitted action chunk's admission time without any
negative data; the discarded classifier serves **exactly once more**, as a primary standard
whose scale is distilled into the deployable estimator; and a **split-conformal layer** turns
scores into vetoes with a finite-sample false-veto guarantee whose only free parameter is a
user-chosen risk budget. Verified live on a UR10e in both modes of a human-robot jar-opening
task, with preliminary transfer to a handover task by re-zeroing the calibration alone.

**The one-sentence sell:** *we do not bolt a new safety monitor onto X-Diffusion; we complete
its own admission-time criterion across the policy's lifecycle — the same test that decided
which human actions could train the policy now decides which policy actions may reach the
robot.*

---

## 1. The lineage and the gap (why this is an extension, not a combination)

### 1.1 Three papers, one object, all training-time-only (verified 08-24)

| Work | The object | Where it acts | At inference |
|---|---|---|---|
| Ambient Diffusion Omni — Daras et al., NeurIPS 2025 Spotlight, arXiv:2506.10038 | t_min = inf{t : time-conditional two-class classifier output crosses 0.5−eps} per sample; Thm 4.2 proves d_TV(P∗N(0,σ²I), Q∗N(0,σ²I)) ≤ d_TV(P,Q)·D/(2σ) | admits low-quality images into training only above t_min | nothing |
| Ambient Diffusion Policy — Wei, Pfaff, Cohn, Dayı, Daskalakis, Daras, Tedrake, arXiv:2606.12365 | same t_min ("minimum amount of noise required to confuse the classifier") for suboptimal robot demos | admits suboptimal demos into IL training above t_min | project page, verbatim: **"Inference ✓ No changes"** |
| X-Diffusion — Pace, Dan, Ning, Bhardwaj, Du, Duan, Ma, Kedia (Cornell), ICRA 2026, arXiv:2511.04671 | k*(A) = min{k : c_θ(k, A^k, s) ≥ 0.5}, "the earliest noise level where human actions are indistinguishable from robot actions" | admits retargeted human actions into the denoising loss only at k ≥ k* (their Eq. 3–4) | bare policy; classifier discarded |

Three consequences for the narrative:

1. **The object is the lineage's own.** We are not importing a foreign criterion; we are
   naming one that appears in three papers ("admission time" as a *term* is our coinage —
   none of the three names it; X-Diffusion's phrase is "the earliest noise level where …
   indistinguishable"). Naming and completing a predecessor's implicit object is a standard,
   legitimate extension-paper move.
2. **The gap is explicit in their own words.** The strongest possible framing of our
   contribution is handed to us by Ambient Diffusion Policy's "Inference: no changes" — the
   entire lineage enforces admission at the entrance and leaves the exit unguarded.
3. **Novelty must be scoped accordingly** (see §8): the admission-time object is prior art;
   our claims are the execution-time enforcement, the one-class estimator + equivalence
   theory, the anchoring distillation, and the conformal veto guarantee — plus hardware.

### 1.2 The reuse move has respectable ancestry

- **Discriminator Rejection Sampling** (Azadi et al., ICLR 2019): the GAN discriminator — a
  training-only apparatus — is reused at *sampling time* to correct the generator's
  distribution, with an idealized exactness theorem followed by an honest practical
  relaxation. Structurally our exact story shape.
- **Valvano et al. 2021** (arXiv:2108.12280), verbatim: *"should we discard the
  discriminator? … the life cycle of adversarial discriminators should not end after
  training."* The lifecycle rhetoric has precedent.
- **"Your Diffusion Model is Secretly a Zero-Shot Classifier"** (Li et al., ICCV 2023):
  revealing a discriminative capability latent in an already-trained generative object is a
  recognized story family.

---

## 2. Why execution-time enforcement is necessary (the hook — measured, not argued)

Per control cycle, for a stochastic policy π and a monitor M:

    P(infeasible action is executed)
        = P_π(infeasible action is emitted) × P_M(missed | emitted)
          \_____ training-time gate _____/    \__ execution-time gate __/
                 shapes this factor                bounds this factor

Two measured facts close the escape routes a reviewer would try:

1. **Training-time gating cannot zero the first factor.** The X-Diffusion-style co-trained
   policy (`policy_jar_uni_v10`), trained *with* admission gating, still emits the
   embodiment-infeasible lid rotation in **12/12 unshielded rollouts** (median first-emission
   cycle 3 under start perturbation) — a stochastic sampler places nonzero mass on what it
   has seen, however down-weighted (`scripts/necessity_emission_probe.py`).
2. **The obvious alternative is worse, not safer.** Hard-filtering the infeasible
   demonstrations out of training (`policy_jar_ab_nofeas`) removes the *detectable* rotation
   intent (0/12 emission) but replaces it with **undefined behavior: 25% of rollouts
   physically penetrate the lid cylinder** with close-intent on the lid. Deleting data does
   not delete the state region; it deletes the supervision there.

So the two gates are **multiplicative complements on the same criterion**, not alternatives:
training-time admission shapes the distribution; execution-time admission bounds every
sample. This risk decomposition is the paper's necessity argument and belongs in the intro.

---

## 3. The criterion, formalized: one latent quantity, two estimators

All symbols and full proofs: `SAFETY_LAYER_MATH_FULL.md`. Summary here.

### 3.1 The shared object

Action chunks are feature vectors x in R^32 (8 steps × 4 chunk-relative channels:
euler + grip; position-blind by design — in free-space tabletop manipulation positional
motion is task, orientation/gripper content is embodiment). For the feasible (executed)
distribution p_R, the forward process of the policy's own scheduler (cosine, K=100,
a_k = alphas_cumprod) defines the noised family

    p_R^(k)(x) = ∫ p_R(x0) · N(x; sqrt(a_k)·x0, (1−a_k)·I) dx0 ,   k = 0..K,

a Gaussian scale-space of the feasible action distribution. **Both estimators below are
functionals of this same family, in the same units (the schedule's k-axis).**

### 3.2 Estimator 1 — the lineage's classifier readout (relative; needs a foil)

The population minimizer of the noise-conditional BCE is the Bayes posterior
c*(k,x) = p_R^(k)(x) / (p_R^(k)(x) + p_C^(k)(x)) against a contrast class p_C (human data in
X-Diffusion; mined rotation counterfactuals in our task-adapted teacher). The admission time
k*(x) = min{k : c(k, x^(k)) ≥ 1/2} is the first noise level at which the smoothed feasible
density overtakes the smoothed contrast density at x.

**Theorem T1** (isotropic Gaussian idealization): log LR_k(x) = −(1/2)·g(k)·Delta(x) with
g(k) = sqrt(a_k)/v_k strictly decreasing, Delta(x) the excess squared distance to the
feasible mean — hence admission is monotone in k and the (eps-banded) crossing time is
**strictly increasing in Delta(x)**. External anchors: Ambient Omni Thm 4.2 (TV of the
noised pair decays ~ D/(2σ)); randomized smoothing (Cohen et al., ICML 2019: a noise scale
converts classifier confidence into a certified distance).

### 3.3 Estimator 2 — the one-class readout (intrinsic; foil-free)

Train a (K+1)-way classifier on (x^(k), k) pairs drawn from the feasible data alone
(self-supervised: the label k is set by us when we add the noise). The population minimizer
is q(k|x) = p_R^(k)(x) / Σ_j p_R^(j)(x), and the score is the posterior mean
t_hat(x) = E[k|x]. No contrast class appears anywhere — the noise ladder is its own foil.

**Theorem T2** (Gaussian idealization): q(k|·) has the monotone-likelihood-ratio property in
distance, so by stochastic dominance t_hat is **strictly increasing in the distance from the
feasible distribution**. External anchors, verified 08-24:
- **Livernoche et al., ICLR 2024 Spotlight (arXiv:2305.18593)** — the DTE origin — prove the
  posterior over the noise variance is **Inverse-Gamma(d/2−1, ‖x−x0‖²/2)** with x0 the
  nearest data point: t_hat is an *explicit monotone function of squared distance to the
  data*, with anomaly rankings provably identical to kNN. This covers the general
  (non-Gaussian, manifold) case.
- Abuduweili et al., TMLR 2025 (arXiv:2412.05488), verbatim: "the noise level … approximates
  their distance to the underlying manifold"; Permenter & Yuan, ICML 2024: denoising ≈
  gradient descent on the distance-to-manifold function with σ_t as the running distance
  estimate.

### 3.4 The equivalence (the mathematical heart of "extension, not combination")

T1 and T2 say: **k* and t_hat are two strictly-increasing readouts of the same latent
quantity — distance to the feasible action manifold — measured in the same noise-schedule
units.** Three independent formal bridges:

- **I-MMSE** (Guo–Shamai–Verdú, IEEE TIT 2005): dI/dSNR = MMSE/2 — the classifier-side
  object (information/distinguishability along the noise axis) and the one-class-side object
  (denoising error along the noise axis) are derivatives of one function.
- **Classification Diffusion Models** (Yadin et al., NeurIPS 2024, arXiv:2402.10095): the
  cross-entropy-optimal *noise-level classifier* and the MSE-optimal *denoiser* are
  analytically equivalent — the precise sense in which the one-class task estimates the same
  object as the density family.
- **NCE/TRE lineage** (Gutmann & Hyvärinen 2010; Rhodes et al., NeurIPS 2020; DRE-∞, Choi et
  al., AISTATS 2022): X-Diffusion's classifier contrasts p_R^(k) against p_C^(k) at each
  rung (human foil); DTE contrasts the rungs against each other (the ladder is the foil).
  Same estimated family, different contrast scheme.

**Measured anchor (jar task, 1,649 real chunks):** Spearman rho(k*, t_hat) = **0.765**
overall, **0.881** within the discriminative range; both floor at 0 on approach/executed
pools; both elevate on rotation (k* med 44; plain t_hat med 7.1). One quantity, two
estimators.

### 3.5 Why the one-class estimator is the deployable one

The two-class design has a generalization dead-end: its negative class requires task
knowledge (one must know *a priori* that "the infeasible thing here is lid rotation" to
build counterfactuals). The one-class estimator removes that dependency — "infeasible"
always means "far from this task's executed actions" — so the criterion becomes portable to
any task that has executed data. This is what makes execution-time admission *deployable*
where the lineage's own classifier is not.

---

## 4. The classifier's one last duty: k*-anchored calibration transfer

### 4.1 The identifiability problem distillation solves

The self-supervised DTE objective constrains the *ranking* of scores but under-determines
their *absolute scale* (plain student: rotation t_hat med 7.1 where the teacher reads 44 —
correct order, wrong units). The teacher's scale is not arbitrary: k* units are the
operational units of the lineage's own training rule (X-Diffusion Eq. 4 consumes them
literally). Aligning the student to them is a **calibration transfer**, in the metrology
sense (primary standard → working instrument; Workman, *Applied Spectroscopy* 2018).

### 4.2 The mechanism (implemented, measured)

    L(phi) = CE(self-supervised noise-step prediction; executed data)
           + lambda · mean_x[ (t_hat_phi(x)/100 − k*_teacher(x)/100)² ]

Teacher = the frozen task-adapted discriminator (`kstar_cls.pth`), 6 noise draws per k.
Pool = **5,449 unlabeled source-task chunks** (2,500 executed + 2,500 human-quiet + 449
human-demo rotation windows — all data that exists anyway; **no feasibility labels
anywhere**). lambda = 1.0 saved as `dte_distilled.pth` (`scripts/dte_distillation.py`).
At a *new* task the teacher is absent by design: the student adapts self-supervised and is
re-zeroed conformally (§6).

### 4.3 What it provably cannot change, and what it measurably did

**Theorem T3** (rank invariance): every rank-based decision — the conformal p-value, the
per-chunk veto — is invariant under strictly monotone rescaling of the score. So
distillation *cannot* move the per-chunk operating point, and indeed did not (P2 below).
What it can move is *magnitudes*: margins, and anything magnitude-sensitive (the sequential
e-detector's growth rate). Measured outcomes (offline replay, within-run baselines,
2026-08-20):

- **Axis calibration — CONFIRMED (the core goal).** Rotation t_hat med 7.1 → **49.4**
  (teacher k* med 44); feasible calibration scores stay low (q99 0.26 → 0.76); absolute
  separation margin widens **~2.4×**. The student now reads in k* units.
- **Sequential layer — improved.** e-detector feasible-stream FA 12/40 → **7/40**;
  slow-rotation miss **17% → 1%**, delay 5 → **2 cycles** (teacher: 2 cycles / 10% miss).
- **Per-chunk veto power — unchanged, as T3 requires.** 86.4% @ 1% FA before and after
  (teacher 91.5%).
- **Rank correlation — unchanged (0.763 → 0.766), as T3 predicts**; the original P3
  metric was ill-chosen (Spearman is scale-invariant by construction).
- The non-monotone residual of the transfer is **concentrated at the decision boundary**
  (2.1% rank reordering; |score shift| grows monotonically with the teacher's k* bin,
  0.04 → 3.02): the teacher injects its foil-induced invariances exactly where decisions
  are made — which is why the magnitude-sensitive sequential layer improved while
  rank-based per-chunk metrics stayed fixed. This rank-invariant/magnitude-sensitive
  dichotomy has pieces in the conformal literature but no complete statement; we state it
  as a small proposition.
- **Cross-task acceptance preserved:** jar-distilled student still accepts handover
  executed chunks at 1.3% ≈ alpha.

Precedents to cite: teacher-student anomaly detection (Bergmann et al., CVPR 2020; Deng &
Li, CVPR 2022; Tien et al., CVPR 2023); calibration transfer in metrology (Workman 2018).
**No prior work found (08-24 search) that distills a two-class noise-conditional teacher
into a one-class diffusion-time student to transfer its decision scale — claim "to our
knowledge" novel.**

Honest deployment note: the robot-verified shield runs `dte_oneclass.pth` (plain student +
conformal); `dte_distilled.pth` is the measured interaction mechanism, not yet the deployed
weights. State this plainly in the paper (the distilled student's per-chunk behavior is
provably identical; what hardware would gain is the sequential layer, which is future work).

---

## 5. From scores to vetoes: conformal admission with a risk budget

### 5.1 The guarantee

Split conformal on n held-out **executed** (hence feasible) chunks: s_alpha = the
ceil((n+1)(1−alpha))-th smallest calibration score. **Theorem T4:** for a new feasible
chunk exchangeable with calibration, P(s(x) > s_alpha) ≤ alpha — finite-sample,
distribution-free. This is verbatim the conformal outlier p-value guarantee of **Bates,
Candès, Lei, Romano, Sesia (Ann. Statist. 2023, arXiv:2104.08279)** (marginal validity;
decisions across chunks are dependent through the shared calibration set — say so).

**The only free parameter is alpha — an interpretable risk budget ("what fraction of
feasible actions I tolerate pausing"), not a score-unit magic number.** This replaces both
the lineage's fixed 0.5 crossing and our own earlier hand-tuned tau = 17.

### 5.2 Calibrate the statistic you actually deploy (Bonferroni, with an admissibility card)

The deployed vote is pessimistic over n_d = 3 policy draws per cycle (flag on the worst
draw). Per-draw calibration under pessimistic voting inflates the per-cycle false-alarm
rate ~n_d× — this was the *measured root cause of the 2026-08-19 false vetoes on hardware*
(a paper-worthy lesson in itself). **Theorem T5:** calibrating at alpha/n_d restores the
per-cycle bound (union bound). Deployed: n_cal = 1,778, alpha = 0.01/3 ≈ 0.0033,
s_alpha = 0.53, 2-of-3 voting + latch.

Reviewer-proofing (verified 08-24): **Vovk–Wang–Wang (Ann. Statist. 2022)** prove that
Bonferroni/min-p merging is not only valid under *arbitrary* dependence but **admissible** —
it cannot be uniformly improved without dependence assumptions (Simes needs PRDS). Our
alpha/n_d is the assumption-free admissible choice, not a hack.

### 5.3 Placement in the calibration literature

- Our rule is the **0–1-loss special case of Conformal Risk Control** (Angelopoulos et al.,
  ICLR 2024) — the door to non-binary risks (cost-weighted halting) is open and free.
- The one strictly-more-elegant alternative that exists: **conformal e-values** (the mean of
  e-values is an e-value → the 3-draw vote merges with *no* Bonferroni penalty, and the
  per-chunk and sequential layers share one anytime-valid currency). Cost: per-chunk power
  and reviewer familiarity. Verdict: keep p-based per-chunk + e-based sequential; one
  related-work sentence acknowledging the e-value unification.
- Every alternative threshold philosophy surveyed (entropy/MSP, energy, ODIN,
  likelihood-ratio, PAC-OOD, selective prediction) either leaves the threshold as an
  unguaranteed hand-set artifact, needs negative/failure data, or carries heavier
  assumptions — the systematic argument for conformal in the one-class safety setting.

---

## 6. Cross-task generalization: the metrology loop (the answer to "one classifier per task?")

The question the narrative must answer: *does a new task need a new classifier?* **No** —
and the three-part argument keeps each claim at exactly the strength we can prove:

1. **Units transfer by construction.** Both estimators are indexed on the policy lineage's
   shared noise schedule; the k-axis is task-invariant by definition.
2. **Geometry transfers empirically, under a stated assumption (A1: shared feasible core
   in chunk-relative, position-blind coordinates).** Zero-shot probes, jar-trained student
   on handover data: acceptance of executed chunks 1.3% ≈ alpha (unit consistency:
   graded synthetic rotations read 10.9 vs 9.3 and 42.6 vs 36.4 at matched intensities;
   detection 98–100%). Scoped to free-space tabletop manipulation; stated as an empirical
   regularity, not a theorem.
3. **The decision guarantee transfers unconditionally.** T4 holds verbatim on any task
   after re-zeroing: collect ~10 executed runs, recompute the quantile. **No negatives, no
   teacher, no retraining required** (measured degradation with single-run calibration:
   7.6–13.6% vs 1.3% — say so, and recommend ≥10 runs).

The metrology summary (one figure in the paper): the two-class discriminator is the
**primary standard** — it exists once, at the source task, where the lineage already trained
it; the DTE network is the **portable working instrument**; distillation is the **one-time
calibration transfer**; per-task conformal re-zeroing is **field use**. The classifier's
per-task cost is exactly zero because it is never needed again.

---

## 7. Hardware evidence inventory (honest attribution)

| Evidence | Shield version | Numbers |
|---|---|---|
| Live veto demo, jar task, mode A (robot must NOT rotate lid) | two-class k* shield (08-13, robot-verified, user-declared perfect) | veto 100%, feasible-phase FPR 5.4%, onset detection 87.5% |
| Live deployment, both collaboration modes, full task cycles incl. release | **DTE one-class shield** (08-18/19 sessions, after calibration fix) | modes A+B verified; false-veto bug root-caused and fixed (per-draw→per-statistic calibration); rehearsal: feasible FA 0%, rotation veto power 82% @ deployed threshold |
| Unshielded emission (necessity) | none | 12/12 rollouts emit rotation |
| Hard-filter ablation (necessity) | none | 0/12 emission but 25% lid-cylinder penetration |
| Offline replay suite | plain vs distilled DTE, teacher reference | §4.3 numbers; teacher 91.5% veto @1%FA / seqFA 1/34 |
| Cross-task (handover) | jar-trained DTE, re-zeroed | 1.3% ≈ alpha acceptance; small-N (2 runs) — preliminary, must be labeled as such |

Honest limits (state in the paper, every time): one platform (UR10e); two tasks; the
sequential e-detector layer is offline-validated but not yet on hardware; theory exact in a
Gaussian idealization with the manifold case covered by citation (Livernoche); conformal
guarantee is marginal and per-chunk, session-level control only in expectation; A1 is an
empirical regularity scoped to free-space tabletop tasks; the deployed weights are the plain
(not distilled) student.

---

## 8. Novelty scope and positioning (each claim against its nearest prior art)

| Our claim | Nearest prior art | The differentiating fact |
|---|---|---|
| (i) Execution-time enforcement of the admission criterion (veto, not alarm) | Ambient Omni / Ambient Diffusion Policy / X-Diffusion — all training-time-only ("Inference: no changes") | first to run the lineage's own criterion at the exit of the lifecycle |
| (ii) One-class estimation of admission time + equivalence theory | Livernoche et al. ICLR 2024 (DTE, general AD); Diff-DAgger (test-time denoising loss, but gates expert queries in DAgger, no deployment shield) | first use of predicted diffusion time as a *vetoing* runtime shield for robot actions; T1/T2 equivalence on one schedule is new |
| (iii) k*-anchored distillation (two-class noise-conditional teacher → one-class student, transferring decision scale) | teacher-student AD (Bergmann 2020; Deng & Li 2022) — teachers are pretrained feature nets, not decision-scale calibrators | no precedent found (08-24); "to our knowledge" |
| (iv) Conformal false-veto guarantee, Bonferroni-corrected for the deployed voting statistic | FIPER (NeurIPS 2025), FAIL-Detect (RSS 2025) — conformal-calibrated runtime failure *alarms* | we differentiate on the score (the lineage's own criterion vs RND/entropy/density) and on the action (hard veto vs alarm); Vovk–Wang–Wang admissibility for the multi-draw correction |

Positioning vs the runtime-monitor family (Sentinel CoRL 2024; FIPER; FAIL-Detect; SAFE
NeurIPS 2025): those detect "the policy is failing" — OOD observations, erratic/inconsistent
actions, high entropy. Our target failure mode is their blind spot: **the policy confidently
executing something embodiment-infeasible** — in-distribution, low-entropy, temporally
consistent, and wrong. Complementary, not competing; one sentence in related work.

Positioning vs model-based safety filters (SafeDiffuser ICLR 2025; path-consistent filtering
ICRA 2026; CBF/shielding/RTA lineage per Hsu–Hu–Fisac 2024): those need explicit constraint
or dynamics models; our score is learned from executed data alone and targets feasibility
constraints nobody wrote down. (Note: the ICRA 2026 filtering paper and FIPER share a first
author in our own institute — coordinate, and cite generously.)

---

## 9. The paper skeleton (Skeleton A — primary)

**Title:** *Runtime Admission Control for Cross-Embodiment Diffusion Policies*
(optional subtitle: *Finite-Sample Feasibility Vetoes from the Policy's Own Training
Criterion*)

**Abstract draft:**

> X-Diffusion trains diffusion policies on human demonstrations by admitting each
> retargeted human action into the loss only above its *admission time* — the earliest
> noise level at which a human-vs-robot classifier can no longer distinguish it from robot
> actions. We show this criterion is enforced at training yet still needed at execution:
> the classifier is discarded, the deployed policy remains stochastic, and on a physical
> UR10e it attempted an embodiment-infeasible lid rotation in 12 of 12 unshielded
> jar-opening rollouts, while hard-filtering the offending demonstrations instead produced
> undefined behavior with physical penetration events in 25% of rollouts. We therefore
> extend admission control to execution time using the same criterion: a one-class
> diffusion time estimation network predicts, per candidate action chunk, the noise level
> it appears drawn from — a quantity that, like the classifier's admission time, is
> provably strictly increasing in the distance to the feasible-action manifold. The
> discarded classifier serves exactly once more, as a primary standard: a one-shot
> distillation aligns the estimator's scale to the training-time admission axis, after
> which no negative data are ever required. Split-conformal calibration on executed robot
> actions turns scores into vetoes, so the only free parameter is a user-chosen risk
> budget bounding the false-veto rate in finite samples, Bonferroni-corrected for
> pessimistic voting over three action draws. On the UR10e the gate vetoes infeasible lid
> rotations live in both modes of a human-robot jar-opening task, and a preliminary study
> shows transfer to a handover task after re-zeroing the calibration alone.

**Sections:**
1. Introduction: Admission Is a Lifecycle Property, Not a Training Trick
2. Background: the admission-time lineage (Ambient Omni → Ambient Diffusion Policy →
   X-Diffusion) and the unguarded exit
3. What Survives the Training Gate: infeasible emission (12/12) and the filtering dilemma
   (25%) — the risk decomposition
4. Estimating Admission Without Negatives: one-class DTE, monotonicity theorems (T1/T2),
   equivalence, and one-shot classifier distillation (T3 + measured margins)
5. From Scores to Vetoes: split-conformal admission with a risk budget (T4/T5,
   admissibility)
6. Hardware Study: jar opening in both collaboration modes; preliminary handover transfer;
   limitations

**Key sentences:**
- "We do not bolt a new safety monitor onto X-Diffusion; we complete its own
  admission-time criterion across the policy's lifecycle."
- "One criterion — the diffusion admission time; two gates — into training, into
  execution; one axis — the policy's own forward process."
- "Where the lineage contrasts human against robot at every noise level, we let the noise
  ladder itself supply the contrast — the criterion survives when no foil exists."
- "The classifier the lineage discards after training is precisely the primary standard
  that calibrates the deployable monitor."
- "Training-time admission shapes the distribution; only execution-time admission bounds
  every sample."

**Alternatives held in reserve:**
- **Skeleton B — "One Criterion, Two Estimators"** (metrology-forward; strongest conceptual
  framing; best for CoRL 2027 if RA-L reviewers ask for more conceptual novelty).
- **Skeleton C — "Admitted in Training, Vetoed in Execution: Closing the Feasibility
  Loop"** (hazard-first; 12/12 and 25% in the second sentence; best for ICRA 2027).

---

## 10. Venue strategy

- **Primary: RA-L** (6 pages + 2 at charge; ICRA/IROS presentation option; 30-day R&R).
  The letter format matches our evidence shape exactly: one sharp claim + finite-sample
  guarantee + single-platform hardware validation (peer scale: Sentinel 1–2 real tasks;
  FIPER 1 real task, 10 calibration rollouts; Lindemann RA-L 2023 is the in-venue template
  for "the only knob is a user-defined risk").
- **If conference speed is wanted: ICRA 2027, deadline 2026-09-15** (~3 weeks) with
  Skeleton C. X-Diffusion itself is ICRA 2026 — same audience.
- CoRL 2026 has passed; CoRL 2027 (~May) is the fallback with Skeleton B.
- **Speed matters:** the space is converging fast (FIPER NeurIPS 2025 + ICRA 2026 safety
  filtering from the same institute; Ambient Diffusion Policy June 2026). T-RO is the
  *extended* follow-up (second platform / third task), not the first submission.

Pre-submission work items (from the 08-22 assessment, unchanged): (a) compile the hardware
metrics table from deploy logs; (b) replay FIPER-style (entropy/RND) and Sentinel-style
(consistency) baselines on our recorded streams to populate the comparison table; (c) add
handover deployment runs to lift the transfer study above small-N.

---

## 11. Verified citation set (all arXiv IDs fetched 08-24)

**Lineage (the object):** Pace et al., X-Diffusion, ICRA 2026, arXiv:2511.04671 · Daras et
al., Ambient Diffusion Omni, NeurIPS 2025 Spotlight, arXiv:2506.10038 (t_min, Thm 4.2) ·
Wei et al., Ambient Diffusion Policy, arXiv:2606.12365 ("Inference: no changes") · Daras et
al., Ambient Diffusion, NeurIPS 2023.

**One-class score + monotonicity:** Livernoche et al., ICLR 2024 Spotlight, arXiv:2305.18593
(Inverse-Gamma posterior; kNN equivalence) · Abuduweili et al., TMLR 2025, arXiv:2412.05488 ·
Permenter & Yuan, ICML 2024, arXiv:2306.04848 · Graham et al., CVPR-W 2023, arXiv:2211.07740 ·
Mahmood et al., ICLR 2021, arXiv:2010.13132.

**Equivalence machinery:** Guo, Shamai, Verdú, IEEE TIT 2005 (I-MMSE) · Yadin et al.,
NeurIPS 2024, arXiv:2402.10095 (CDM) · Choi et al., AISTATS 2022, arXiv:2111.11010 (DRE-∞) ·
Rhodes et al., NeurIPS 2020, arXiv:2006.12204 (TRE) · Gutmann & Hyvärinen, AISTATS 2010
(NCE) · Cohen et al., ICML 2019, arXiv:1902.02918 (randomized smoothing).

**Decision layer:** Bates et al., Ann. Statist. 2023, arXiv:2104.08279 (conformal outlier
p-values) · Vovk, Wang, Wang, Ann. Statist. 2022 (admissible p-merging → Bonferroni) ·
Angelopoulos et al., ICLR 2024 (Conformal Risk Control) · Tibshirani et al., NeurIPS 2019
(weighted conformal; covariate-shift transfer) · Gibbs & Candès, NeurIPS 2021 (ACI; drift
discussion) · Shin, Ramdas, Rinaldo, arXiv:2203.03532 (e-detectors) · Volkhonskiy et al.,
COPA 2017 (inductive conformal martingales) · Vovk et al., ICML 2003 (testing
exchangeability online) · Bashari et al., NeurIPS 2023, arXiv:2302.07294 (conformal
e-values).

**Runtime monitors / robotics:** Agia et al., Sentinel, CoRL 2024, arXiv:2410.04640 · Römer
et al., FIPER, NeurIPS 2025, arXiv:2510.09459 · Xu et al., FAIL-Detect, RSS 2025,
arXiv:2503.08558 · Gu et al., SAFE, NeurIPS 2025, arXiv:2506.09937 · Ren et al., KnowNo,
CoRL 2023, arXiv:2307.01928 · Lindemann et al., RA-L 2023, arXiv:2210.10254 · Diff-DAgger,
arXiv:2410.14868 · SafeDiffuser, ICLR 2025, arXiv:2306.00148 · Römer et al., path-consistent
safety filtering, ICRA 2026, arXiv:2511.06385 · Alshiekh et al., AAAI 2018 (shielding) ·
Hsu, Hu, Fisac, Annu. Rev. Control Robot. Auton. Syst. 2024 (safety-filter survey).

**Reuse-and-distillation precedents:** Azadi et al., ICLR 2019, arXiv:1810.06758
(Discriminator Rejection Sampling) · Valvano et al., 2021, arXiv:2108.12280 (lifecycle
rhetoric) · Li et al., ICCV 2023, arXiv:2303.16203 · Bergmann et al., CVPR 2020 (Uninformed
Students) · Deng & Li, CVPR 2022 (Reverse Distillation) · Tien et al., CVPR 2023 · Workman,
Applied Spectroscopy 2018 (calibration transfer / metrology).
