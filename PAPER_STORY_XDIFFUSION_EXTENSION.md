# Positioning: From Data Admission to Action Admission —
# Extending X-Diffusion's Feasibility Criterion from Learning Time to Execution Time

*Story memo, 2026-08-20. Purpose: sell the DTE feasibility shield as a genuine
EXTENSION of X-Diffusion, not a combination of two methods.*

---

## 1. The unifying mathematical object (why this is one method, not two)

X-Diffusion's entire mechanism lives on one axis: the **forward-diffusion noise ladder**
of the action space. Define the noise-indexed likelihood family ("diffusion scale-space")
of the robot action distribution:

    L_x(k) = p_R^(k)(x) = (p_R * N(0, sigma_k^2))(x),   k = 0..K

Both feasibility estimators are **readouts of this same family**:

| | X-Diffusion classifier (Eq. 2-3) | our DTE network |
|---|---|---|
| object read | p_R^(k) vs p_H^(k) (relative) | p_R^(k) alone (intrinsic) |
| readout | k* = min{k : c_theta(k, A^k) >= 0.5} — the crossing point of the two smoothed densities | t-hat = posterior mean over k of p_R^(k)(x) — the noise level that best explains x |
| foil | human dataset D_H | **the noise ladder itself** |
| semantics | "minimum indistinguishability step" | "estimated diffusion time" |

Both define **feasibility as a diffusion admission time**: how much forward noise until
this action is accounted for by the robot's distribution. The difference is only *what
supplies the contrast*: X-Diffusion contrasts against the human dataset at each k; DTE
contrasts the noise levels of the robot's own data against each other (a 101-way
noise-level classification — contrastive estimation along the diffusion axis, in the
NCE/TRE lineage; X-Diffusion's classifier is the same telescoping-bridge construction
with the human data as the foil).

**Empirical anchor (measured, jar task):** Spearman rank correlation between k* and
t-hat across 1,649 real chunks: **rho = 0.765 overall, 0.881 within the discriminative
range** (both floor at 0 on approach/executed pools; both elevate on rotation: k* med 44,
t-hat med 7.1). Two estimators, one quantity. (`figures/e1_same_quantity.npz`)

## 1.5 Mathematical foundations

Throughout, actions are chunk features $x \in \mathbb{R}^d$ (here $d=32$: 8 steps of
chunk-relative euler + grip), and the forward process is the variance-preserving (VP)
diffusion shared with the policy, $q(x_k \mid x_0) = \mathcal{N}\big(\sqrt{\bar\alpha_k}\,x_0,\ (1-\bar\alpha_k) I\big)$,
with $\bar\alpha_0 \approx 1 > \bar\alpha_1 > \dots > \bar\alpha_K \approx 0$ (cosine schedule, $K=100$).

### 1.5.1 The noise-indexed family is a Gaussian scale-space

For any distribution $p$ over actions, define its noised marginal at step $k$:

$$p^{(k)}(x) \;=\; \int p(x_0)\, \mathcal{N}\!\big(x;\ \sqrt{\bar\alpha_k}\,x_0,\ (1-\bar\alpha_k) I\big)\, dx_0 .$$

Writing $D_a$ for the dilation $ (D_a p)(x) = a^{-d} p(x/a)$, this is exactly

$$p^{(k)} \;=\; \big(D_{\sqrt{\bar\alpha_k}}\, p\big) \,*\, \mathcal{N}\big(0,(1-\bar\alpha_k)I\big),$$

i.e. **a Gaussian scale-space of the action distribution** (up to the VP shrinkage
$\sqrt{\bar\alpha_k}$): increasing $k$ blurs $p$ with a Gaussian of growing bandwidth
$\sigma_k^2 = 1-\bar\alpha_k$ while contracting its support toward the origin. Both
estimators below are functionals of the same family $\{p_R^{(k)}\}_{k=0}^{K}$, where
$p_R$ is the distribution of feasible (robot-executed) chunks.

### 1.5.2 The X-Diffusion readout: smoothed likelihood-ratio crossing

With balanced class sampling, the population minimizer of X-Diffusion's BCE (their Eq. 2)
at noise level $k$ is the Bayes posterior

$$c^\star(k, x) \;=\; \frac{p_R^{(k)}(x)}{p_R^{(k)}(x) + p_H^{(k)}(x)},
\qquad\text{so}\qquad
c^\star(k,x) \ge \tfrac12 \iff \Lambda_k(x) := \frac{p_R^{(k)}(x)}{p_H^{(k)}(x)} \ge 1 .$$

Hence their Eq. 3, $k^\ast(x) = \min\{k : c_\theta(k, x^{(k)}) \ge \tfrac12\}$, is (at the
population level) **the first noise scale at which the smoothed robot density overtakes the
smoothed human density at $x$** — a *relative* readout of the family, requiring the human
foil $\{p_H^{(k)}\}$.

**Proposition A (monotone admission and crossing–distance relation; isotropic model).**
Let $p_R = \mathcal{N}(\mu_R, s^2 I)$, $p_H = \mathcal{N}(\mu_H, s^2 I)$ and define the
excess squared distance $\Delta(x) = \lVert x-\mu_R\rVert^2 - \lVert x-\mu_H\rVert^2$.
Then with $v_k = \bar\alpha_k s^2 + (1-\bar\alpha_k)$,

$$\log \Lambda_k(x) \;=\; -\,\frac{\bar\alpha_k}{2 v_k}\,\Delta(x)\;(1 + o(1)),$$

and $\bar\alpha_k / v_k$ is strictly decreasing in $k$ whenever $s^2 < 1$ (normalized data).
Consequently (i) $|\log\Lambda_k(x)|$ shrinks monotonically to $0$ — admission is monotone
in $k$, so the first-crossing time is well defined (this is the population fact our
monotone regularizer enforces in the finite-sample classifier); and (ii) for any decision
band $\varepsilon>0$, the crossing time $k^\ast_\varepsilon(x) = \min\{k: |\log\Lambda_k(x)|\le\varepsilon\}$
is an **increasing function of $\Delta(x)$**: actions farther (in excess) from the robot
class are admitted only at proportionally larger noise scales. *Proof sketch:* expand both
smoothed log-densities at equal variance $v_k$; the quadratic difference gives the display;
monotonicity of $\bar\alpha_k/v_k = \big(s^2 + (1/\bar\alpha_k - 1)\big)^{-1}$ is immediate;
solve $\bar\alpha_k/v_k = 2\varepsilon/\Delta$ and invert the decreasing map. $\square$

### 1.5.3 The DTE readout: posterior over the noise level itself

Train a $(K{+}1)$-way classifier on pairs $(x^{(k)}, k)$ with $x^{(k)} \sim p_R^{(k)}$ and
$k \sim \mathrm{Unif}\{0..K\}$. The population minimizer of the cross-entropy is the Bayes
posterior **over the noise ladder of the feasible family alone**:

$$q(k \mid x) \;=\; \frac{p_R^{(k)}(x)}{\sum_{j=0}^{K} p_R^{(j)}(x)},
\qquad
\hat t(x) \;=\; \mathbb{E}[k \mid x] \;=\; \sum_k k\, q(k\mid x).$$

This is an *intrinsic* readout of the same family — no foil distribution appears anywhere.

**Proposition B (distance monotonicity of $\hat t$; Gaussian case).** Let
$p_R = \mathcal{N}(\mu, s^2 I)$ with $s^2 < 1$, and $r = \lVert x - \sqrt{\bar\alpha_k}\,\mu\rVert$
(for clarity take $\mu=0$, $r=\lVert x\rVert$). Then $p_R^{(k)} = \mathcal{N}(0, v_k I)$ with
$v_k$ strictly increasing in $k$, and for $r_1 < r_2$ the ratio
$q(k\mid r_2)/q(k\mid r_1) \propto \exp\!\big(-(r_2^2-r_1^2)/(2v_k)\big)$ is strictly
increasing in $k$ (monotone likelihood ratio). Hence $q(\cdot\mid r_2)$ stochastically
dominates $q(\cdot\mid r_1)$ and $\hat t(x)$ is **strictly increasing in the distance $r$
from the feasible distribution**. $\square$
For non-Gaussian, manifold-supported $p_R$, Livernoche et al. (ICLR 2024) derive the
analytic posterior and show it depends on $x$ through its distance to the data — the same
monotone-distance semantics in the general case.

Propositions A and B are the formal content of "two readouts of one quantity": both
$k^\ast$ and $\hat t$ are increasing functions of how far $x$ sits from the feasible
family, measured **in units of the same forward-diffusion noise schedule**. Empirically, on
1,649 real chunks Spearman $\rho(k^\ast, \hat t) = 0.765$ (0.881 within the discriminative
range).

### 1.5.4 One family, two contrast schemes (the NCE/TRE view)

Both classifiers are noise-contrastive estimators on the same ladder:

- X-Diffusion, at each rung $k$, contrasts $p_R^{(k)}$ **against $p_H^{(k)}$** — a
  telescoping-bridge family with the human dataset as the foil (cross-embodiment
  contrast);
- DTE contrasts the rungs $\{p_R^{(j)}\}_j$ **against each other** — the optimal logits of
  the $(K{+}1)$-way CE recover $\log p_R^{(k)}(x)$ up to an $x$-dependent constant, i.e.
  the ladder itself supplies the negatives (cross-noise-level contrast).

Removing the human foil therefore does not change the estimated object — only the contrast
used to estimate it. This is the precise sense in which the extension is foil-free rather
than method-new.

### 1.5.5 Decision layer: distribution-free calibration of the readout

Let $s(x)$ be either readout ($\hat t$ in the deployed system) and
$S = \{s(x_1),\dots,s(x_n)\}$ scores of held-out **executed** (hence feasible) chunks.

**Proposition C (conformal false-veto guarantee).** If a new feasible chunk $x_{n+1}$ is
exchangeable with the calibration chunks, then with
$\hat s_\alpha = S_{(\lceil (n+1)(1-\alpha)\rceil)}$ (order statistic),

$$\mathbb{P}\big(s(x_{n+1}) > \hat s_\alpha\big) \;\le\; \alpha,$$

finite-sample and distribution-free (split conformal). The only free parameter is the risk
budget $\alpha$; no score-unit threshold is tuned.

**Proposition D (deployed statistic; Bonferroni).** The deployed vote is pessimistic over
$m$ policy draws per cycle ($m=3$): flag if $\max_{j\le m} s(x^{(j)}) > \hat s_{\alpha/m}$.
If each draw is marginally exchangeable with calibration when the policy is in a feasible
mode, the union bound gives per-cycle false-flag probability $\le \alpha$. (This is why the
deployed level is $\alpha/m = 0.01/3 \approx 0.0033$: **calibrate the statistic actually
deployed** — per-draw calibration under pessimistic voting was the root cause of the
08-19 false vetoes.)

The M-of-N vote and latch sit above these marginal guarantees as deterministic logic; the
sequential (e-process) layer with anytime-valid guarantees is future work (known one-class
calibration-drift gap, Sec. 8 of the DTE readme).

### 1.5.6 Lifecycle risk decomposition (why both gates are needed)

For a stochastic policy $\pi$ and monitor $M$, per cycle

$$\mathbb{P}(\text{infeasible executed}) \;=\; \underbrace{\mathbb{P}_\pi(\text{infeasible emitted})}_{\text{training-time gate shrinks}} \times \underbrace{\mathbb{P}_M(\text{miss} \mid \text{emitted})}_{\text{execution-time gate bounds}} .$$

Training-time admission (X-Diffusion) shapes the *distribution* — it cannot zero the first
factor for a sampler (measured: 12/12 unshielded rollouts emit rotation), and deleting the
infeasible data instead replaces detectable intent with undefined behavior (25 % lid-cylinder
penetration in the hard-filter ablation). The execution-time gate controls the second factor
with the conformal guarantee on false vetoes and empirically measured miss rates. The two
gates are multiplicative complements on the same criterion, not alternatives.

## 2. The story arc: one criterion, two gates, one axis

> X-Diffusion asks: *how much noise until a human action may TEACH the robot?*
> We ask the dual: *how much noise until the robot's own action is EXPLAINED by what it
> has executed?* Same criterion, extended from learning to acting.

- **Step 0 (X-Diffusion, prior work).** Admission time k* gates which human actions
  enter the denoising loss — a **training-time data gate**. The classifier is discarded
  after training; deployment is the bare policy.
- **Step 1 (ours, robot-verified).** The same admission time, enforced at **execution
  time**: score every policy-emitted chunk, veto when the admission time exceeds the
  feasible range. *Necessity is measured, not assumed*: the stochastic policy emits
  rotation in 12/12 unshielded rollouts (median first emission cycle 3 under start
  perturbation), and the training-time alternative (hard-filtering infeasible data)
  does not produce safety but **undefined behavior** — 25% of rollouts physically
  penetrate the lid cylinder, with close-intent on the lid. Training-time gating shapes
  the mean; only execution-time gating bounds every sample.
- **Step 2 (ours).** Foil-free estimation of the same admission time: replace the human
  foil with the noise ladder itself (DTE). This removes the task-specific negative
  requirement (the human/counterfactual dataset) — the criterion becomes portable to any
  task that has executed data. Zero-shot cross-task evidence: jar-trained DTE accepts
  handover executed chunks at 1.3% ≈ alpha.
- **Step 3 (ours).** Calibration: the fixed 0.5 crossing / hand threshold is replaced by
  a conformal quantile of executed-chunk scores with a user-interpretable risk budget
  alpha (finite-sample guarantee; Bonferroni-corrected for the pessimistic multi-draw
  vote).

## 3. The concrete interaction: k*-anchored distillation (proposed, testable)

The user-visible question "do the two networks interact?" has a clean affirmative
mechanism — **the pre-trained X-Diffusion discriminator calibrates the one-class
network's time axis**:

    L(phi) = L_DTE(phi; executed data)                    # self-supervised, foil-free
           + lambda * E_x [ (t_hat_phi(x) - k*_theta(x))^2 ]   # teacher anchoring, source task

- The teacher is **free**: every X-Diffusion policy training already produces the
  discriminator; instead of discarding it (as the original paper does), we distill its
  noise-scale semantics into the deployable one-class monitor.
- What distillation transfers is exactly what pure one-class training failed to acquire
  (measured): the **foil-induced invariances** (texture equalization; ours showed
  sequential-layer drift 8/34 stream FA and a null positive-augmentation ablation).
  Precedent for teacher-student one-class monitoring: reverse distillation
  (CVPR 2022, 2023).
- At a **new task** the teacher is absent by design — DTE adapts self-supervised on that
  task's executed data while keeping the inherited invariances/axis.
- Bonus: t-hat lands on the same 0-100 k-axis as X-Diffusion's k*, so all of the
  original paper's semantics (Fig. 3-style visualizations, "indistinguishability step")
  carry over verbatim to deployment plots.

Testable predictions (offline replay): (P1) distilled DTE closes the sequential
stability gap toward the discriminator's 1/34; (P2) per-chunk veto power closes
88.4% -> ~91.5% @ 1% FA; (P3) k*-vs-t-hat correlation rises from 0.88 toward ~1 on the
source task without hurting cross-task acceptance.

Secondary interactions (mention, not headline): the discriminator as **curator** of the
student's feasible corpus (k* <= tau filter replacing the heuristic euler filter);
agreement/disagreement between the two readouts as a runtime self-diagnosis signal.

## 4. Paper skeleton

1. Background: X-Diffusion; feasibility = minimum indistinguishability step (their
   Eq. 2-3); classifier used only to admit data into training.
2. **Claim: admission belongs at both ends of the lifecycle.** Risk decomposition
   P(execute infeasible) = P(emit) x P(monitor miss); training-time gating cannot zero
   the first factor for a stochastic sampler (12/12 emission; hard-filter ablation).
3. Same axis, execution gate: the k* shield (robot-verified demo), all veto machinery.
4. Foil-free admission time: DTE as the intrinsic readout (rho=0.88 equivalence), zero
   negatives, cross-task recipe + zero-shot handover evidence.
5. k*-anchored distillation: the pretrained discriminator calibrates the deployable
   monitor's axis and invariances (P1-P3).
6. Conformal calibration: risk-budget alpha, deployed-statistic correction (alpha/n_draws).
7. Hardware: unified Nutella task, both modes, banner demo; offline replay suite.

## 5. Key sentences (for abstract/intro)

- "X-Diffusion decides which human actions may teach a robot; we extend the same
  criterion to decide which of its own actions a robot may execute."
- "One criterion — the diffusion admission time; two gates — into training, into
  execution; one axis — the policy's own forward process."
- "Where X-Diffusion contrasts human against robot at every noise level, we let the
  noise ladder itself supply the contrast — the criterion survives when no human foil
  exists."
- "The discriminator that X-Diffusion discards after training is precisely the teacher
  that calibrates the deployable monitor."

## References (beyond the X-Diffusion paper)

- Livernoche et al. On Diffusion Modeling for Anomaly Detection (DTE). ICLR 2024.
- Rhodes, Xu, Gutmann. Telescoping Density-Ratio Estimation. NeurIPS 2020.
- Gutmann & Hyvarinen. Noise-Contrastive Estimation. AISTATS 2010.
- Deng & Li. Anomaly Detection via Reverse Distillation from One-Class Embedding. CVPR 2022;
  Tien et al., Revisiting Reverse Distillation. CVPR 2023.
- Angelopoulos & Bates et al. Learn then Test / conformal risk control. AOAS 2025.
- Daras et al. Ambient Diffusion (Omni) — the data-quality classifier lineage X-Diffusion builds on.
