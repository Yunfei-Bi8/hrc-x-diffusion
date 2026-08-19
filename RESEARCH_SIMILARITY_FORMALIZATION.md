# Formalizing "Similarity to Feasible Data" for the Inference-Time Feasibility Shield

*Research memo, 2026-08-18. Status: direction survey + formalization; no training started.*

## 0. The question

At inference we want to decide, per policy action chunk `x`, whether it is "similar to the
feasible (robot-executed) data" used to build the discriminator — **without a hand-tuned,
score-unit threshold** (the current `tau=17`), and with "similarity" defined in a general,
principled way.

This decomposes into two (optionally three) independent layers, each a mature field:

| layer | question | field |
|---|---|---|
| score | s(x) = how far is x from the feasible distribution p_R? | OOD / one-class detection |
| decision | turn s(x) into execute/veto without magic numbers | conformal calibration (p-values, risk level α) |
| sequential (opt.) | accumulate weak evidence over cycles | e-processes / conformal test martingales |

## 1. Why the discriminator output alone is NOT similarity — and why k\* already is

A binary classifier estimates the **ratio** r(x) = p_R(x)/p_alt(x). "Similar to feasible"
asks about **p_R(x) itself** (or distance to supp p_R). These diverge wherever both densities
are small — the **density chasm** problem (Rhodes et al., *Telescoping Density-Ratio
Estimation*, NeurIPS 2020). Our wobble-band false vetoes were an empirical instance: chunks
unlike BOTH classes get arbitrary ratios.

Two literature connections that legitimize the existing design:

1. **The noise-conditional discriminator IS a telescoping density-ratio estimator**: forward
   diffusion creates bridge distributions; classifying at each noise level is exactly TRE's
   bridge-classification solution to the chasm.
2. **The k\* readout ≈ Diffusion Time Estimation** (Livernoche et al., ICLR 2024): using the
   diffusion time needed for a sample to blend into the distribution as the anomaly score.

**Formalization of k\*** (proposed theory box):

    k*(x) = min{ k : (p_R * N(0, σ_k²))(x) ≥ (p_alt * N(0, σ_k²))(x) }

i.e. the smoothing scale at which the feasible density dominates at x. Under mild conditions
σ_{k*} is monotone in the Euclidean distance from x to the feasible data manifold — **k\* is a
distance-to-feasible-support measured in units of the diffusion noise schedule.** That is
already a valid similarity measure; what needs replacing is the raw threshold on it.

## 2. Score families (direction 1)

| family | formal score | fit for 32-d chunks | refs |
|---|---|---|---|
| distance | kNN distance to executed-chunk buffer | excellent; training-free; interpretable ("nearest feasible chunk differs by X rad") | Sun et al. ICML 2022 (KNN-OOD); Laxhammar's SNN-CAD used kNN nonconformity |
| one-class density | −ELBO or DTE posterior E[t|x] from a small diffusion/flow trained on feasible chunks ONLY | elegant; **removes rotation counterfactuals entirely** (no f₁ anywhere) | DTE ICLR 2024; Graham et al. DDPM-OOD |
| reconstruction | noise-to-k then denoise, multi-k error profile | same family as above | Graham et al. |
| smoothed ratio (current) | k\* as above; soft variant = area ∫(0.5 − P_k)₊ dk | deployed & robot-verified; weak residual dependence on the alternative class | TRE + DTE connections |

Caveats: likelihood-based OOD has known pathologies in high dim (typicality; Nalisnick et
al.) — mild here (32-d). Distance family is the simplest robust default.

## 3. Decision layer without magic numbers (direction 2): conformal p-values

Choose any score s. With a held-out calibration set {x_1..x_n} of feasible chunks:

    p(x) = (1 + #{ i : s(x_i) ≥ s(x) }) / (n + 1)

Then p(x) is (finite-sample, distribution-free) uniform for feasible-like chunks. The
decision `veto if p(x) < α` replaces `k* > 17` — **α is a universal risk budget ("I accept
1% of feasible chunks being paused"), not a score-unit constant.** Precedents: conformal
anomaly detection for trajectories (Laxhammar & Falkman, PAMI 2014); conformal runtime
monitors in robotics (Lindemann et al.; FIPER's band calibration).

## 4. Sequential layer (direction 3): e-processes — the non-hacky CUSUM replacement

The user's objection to CUSUM is fair: it needs a parametric alternative f₁ (counterfactual
LLR) and a threshold h. **Conformal test martingales / e-processes need neither.** Feed the
per-chunk p-values into a betting function, e.g. g(p) = κ p^{κ−1} with κ∈(0,1) (or mixture
over κ):

    E_t = Π_{i≤t} g(p_i)

If the stream is exchangeable with the feasible calibration data, E_t is a supermartingale;
**Ville's inequality** gives P(∃t: E_t ≥ 1/α) ≤ α — a **lifetime false-alarm guarantee**
where the alarm line 1/α is implied by the guarantee, not tuned. Weak evidence compounds
multiplicatively → slow-rotation evasion is still caught (CUSUM's virtue, kept), with inputs
only {feasible data, α}. Refs: Vovk (conformal test martingales, COPA 2021; inductive
conformal martingales for change-point detection, COPA 2017); Ramdas & Wang, *Hypothesis
Testing with E-values* (2025); Ramdas–Grünwald–Vovk–Shafer, game-theoretic statistics.

## 5. Recommended stack ("conformal feasibility monitor")

    score:      k* (formalized as smoothing-distance)  |  kNN  |  one-class DTE   ← ablatable
    calibrate:  conformal p on held-out executed chunks
    decide:     per-chunk p<α   +   e-process wealth ≥ 1/α (sequential)
    only knob:  α  (interpretable risk budget, finite-sample guarantee)

Contributions: (i) theory box: k\* = distance-to-feasible-support via smoothing (TRE/DTE
links); (ii) score-agnostic conformal + anytime-valid sequential monitor that subsumes the
five hand-tuned deploy mechanisms (τ, M-of-N, latch, yaw-trip, evidence verdict); (iii)
score-family ablation under an identical decision layer. Positioning vs FIPER/Sentinel
unchanged: they detect "policy about to fail"; we detect "action infeasible for the
embodiment", now with anytime-valid guarantees.

All validation is offline replay (existing 40+ deploy logs + both datasets). The only
optional training is the one-class score for the ablation.

## References

- Rhodes, Xu, Gutmann. Telescoping Density-Ratio Estimation. NeurIPS 2020.
- Livernoche et al. On Diffusion Modeling for Anomaly Detection (DTE). ICLR 2024.
- Graham et al. Denoising Diffusion Models for Out-of-Distribution Detection. CVPR-W 2023.
- Sun, Ming, Zhu, Li. Out-of-Distribution Detection with Deep Nearest Neighbors. ICML 2022.
- Laxhammar, Falkman. Online Learning and Sequential Anomaly Detection in Trajectories. TPAMI 2014.
- Vovk. Retrain or not retrain: conformal test martingales for change-point detection. COPA 2021.
- Volkhonskiy et al. Inductive Conformal Martingales for Change-Point Detection. COPA 2017.
- Ramdas, Wang. Hypothesis Testing with E-values. Foundations & Trends in Statistics, 2025.
- Ramdas, Grünwald, Vovk, Shafer. Game-Theoretic Statistics and Safe Anytime-Valid Inference. Statistical Science 2023.
- Lindemann et al. Safe Planning in Dynamic Environments using Conformal Prediction. RA-L 2023.
- Römer, Kobras, Worbis, Schoellig. FIPER: Failure Prediction at Runtime for Generative Robot Policies. NeurIPS 2025.
- Agia et al. Unpacking Failure Modes of Generative Policies (Sentinel). CoRL 2024.
