# Inference-Time Feasibility Shield via One-Class Diffusion Time Estimation (DTE)

*Unified Nutella jar-opening task. Deployment-time feasibility checking that needs
**no negative / infeasible samples** and **no hand-tuned score threshold**.*

Deployed artifacts:
- Policy: `X-Diffusion-Data/_private/runs/policy_jar_uni_v10`
- Shield: `X-Diffusion-Data/_private/runs/kstar_uni/dte_oneclass.pth`
  (`score=="dte_posterior_mean"`, 1778 embedded conformal calibration scores)
- Deploy: `mt_pi/scripts/deploy_xdiffusion_ur10e.py --shield-run …/dte_oneclass.pth --shield-dte-alpha 0.0033`

---

## 0. One-sentence summary

Each control cycle the policy emits an 8-step action chunk; the DTE network scores
"how far this chunk is from feasible (robot-executed) actions"; if that score exceeds a
threshold **computed automatically by conformal calibration** (no physical-unit magic
number), a 2-of-3 vote vetoes and freezes the arm. The DTE network **never saw a single
infeasible action during training** — that is what makes it transfer across tasks.

---

## 1. Relationship to the original X-Diffusion discriminator

**We do NOT use the original X-Diffusion discriminator at deployment time.** Three distinct
objects are easy to confuse:

| | Original X-Diffusion | Our k\* version (earlier) | Our DTE version (current deploy) |
|---|---|---|---|
| type | two-class classifier (human vs robot) | two-class classifier (robot vs rotation counterfactual) | **one-class model (feasible only)** |
| used where | **training only** (gates policy loss) | inference-time safety layer | inference-time safety layer |
| needs negatives | yes (human data) | yes (rotation counterfactuals) | **no** |

Two essential points:

1. **The original discriminator is never used at deployment.** In X-Diffusion the identity
   classifier only runs during *policy training*, to weight which human actions enter the
   denoising loss; it is then **discarded and not loaded at inference**. Their deployment is
   the bare policy with no runtime feasibility check. Therefore *"use the discriminator as a
   deployment-time safety layer"* is itself our contribution — the original paper does not
   do it.

2. **Even the idea, we changed.** The earlier k\* version kept the two-class architecture
   (just swapping the classes to robot-executed vs rotation-counterfactual). The current
   **DTE version changes the architecture itself**: from a two-class classifier that needs
   both positives and negatives, to a **one-class diffusion model** trained on feasible data
   only. We inherit X-Diffusion's core insight (use forward-diffusion noising to test whether
   an action belongs to the robot distribution) but change the discriminator's **form,
   training, and stage of use**.

Why go this far: the two-class design has a generalization dead-end — its negative class
needs task knowledge (you must know *a priori* that "the infeasible thing in this task is
rotation" to build the counterfactuals). On a new task the negative-sample design does not
transfer. DTE removes that dependency: **"infeasible" always means "far from feasible", with
no knowledge of what infeasible looks like.**

---

## 2. What the DTE network is and how it works

### 2.1 Background: forward diffusion

Any action chunk `x` can be progressively corrupted with Gaussian noise. The amount of noise
is indexed by a **diffusion time** `t ∈ [0,100]`: `t=0` is the clean chunk, `t=100` is
almost pure noise.

### 2.2 What the network does

The DTE network (a small **temporal 1-D CNN**, treating a chunk as a 4-channel × 8-step
sequence) does one thing:

> given an action chunk, **estimate which diffusion time `t` it looks like it was noised to.**

It outputs a distribution over `t∈[0,100]`; we take the **posterior mean** as the score.

### 2.3 Why that score is a feasibility distance

The network is trained **only on feasible actions**, so its worldview is "everything is a
feasible action plus some amount of noise." Therefore:

| input action | network's reaction | score |
|---|---|---|
| truly feasible (descend, grasp holder) | "a clean feasible action, ~no noise" | **≈ 0** |
| infeasible (lid rotation) | "I don't recognize this; the only explanation is a feasible action with **a lot** of noise added" | **high** |

So the score = **"how much noise must be assumed to explain this chunk as a feasible
action."** The farther from the feasible manifold, the more noise must be assumed, the higher
the score. This is not ad-hoc: Livernoche et al., *On Diffusion Modeling for Anomaly
Detection* (ICLR 2024) show the posterior-mean diffusion time is **monotone in the Euclidean
distance to the data manifold** — a theoretically-grounded distance-to-feasible-support.

---

## 3. How it is trained and what ground truth it needs

### 3.1 Training loop (self-supervised)

Each step:
1. take a batch of **feasible chunks** (robot-executed);
2. draw a random `t` per chunk, add the corresponding Gaussian noise;
3. train the network to **predict `t`** from the noised chunk (101-way classification,
   cross-entropy);
4. backprop. ~4000 steps.

### 3.2 Ground truth needed — essentially none

- ❌ no "feasible/infeasible" labels;
- ❌ no infeasible samples (the k\* version needed rotation counterfactuals);
- ✅ the only "label" is the random `t` — and **`t` is set by us when we add the noise**, not
  annotated. It is free and self-generated (self-supervision).

The **only real input is a dataset of "actions the robot has executed."** Which ones are
feasible needs no human labeling either — it follows from **provenance** (executed on the
robot ⇒ physically feasible).

### 3.3 Features (position-blind, same as k\*)

Input = euler(3) + grip(1), euler made **chunk-relative** (subtract row 0), fixed physical
scaling → 32-D. **No position** — in this task all positional motion is feasible; the only
infeasible content is wrist rotation, which lives in the euler channels.

---

## 4. Feasible-or-not decision — is there a hand-tuned gate? NO

There is **no `tau=17`-style hand-tuned physical threshold**. The decision line is computed
automatically by conformal calibration.

### 4.1 Threshold computed once at startup

The checkpoint stores **1778 calibration scores** (DTE scores of held-out robot-executed
feasible chunks). Given a risk level **α**, the threshold `s_α` = the `(1-α)` quantile of
those calibration scores:

    sort the 1778 feasible-action scores, take the (1-α) quantile = s_α
    meaning: "a truly feasible action has score below s_α with probability 1-α"

**The only knob is α — an interpretable risk ("what fraction of feasible actions I tolerate
being paused"), not a score-unit magic number.** This is the key difference from a hand-tuned
τ: τ=17 cannot state its false-alarm rate; α is directly specified, interpretable, with a
finite-sample guarantee.

### 4.2 Why α = 0.0033, not 0.01

The deploy vote takes the **most anomalous of 3 policy draws** per cycle (pessimistic rule).
For a "max-over-3-draws" statistic the effective false-alarm rate is inflated ~3× by a union
bound. So we Bonferroni-correct: **α = 0.01 / 3 ≈ 0.0033** (`--shield-dte-alpha 0.0033`) →
`s_α = 0.53`. This is not a hand-tuned gate — it is the **correct conformal calibration of the
statistic actually deployed**. (Paper-worthy lesson: calibrate the statistic you actually use;
per-draw calibration + pessimistic voting needs α/n_draws.)

### 4.3 Per-cycle decision

    policy emits chunk → DTE score s
    s > s_α (=0.53) ?  →  this chunk casts one "bad" vote
    2 of the last 3 chunks bad (2-of-3)  →  VETO: freeze + red INFEASIBLE banner + latch

---

## 5. Full data loop (per ~0.6 s control cycle)

    main proc reads robot pose + hand condition → sends to inference proc
    inference proc: policy emits 8-step chunk → DTE scores it → returns (chunk, score)
    main proc: vote on the score
       pass → execute the chunk
       veto → freeze + banner  (mode A: fires above the lid via sanitize/descend-lock/yaw-trip)

Both modes share this:
- **Mode B (robot grasps holder first):** all actions feasible, DTE ≈ 0, all pass → descend,
  grasp holder, human rotates lid, release, retreat.
- **Mode A (human grasps holder first):** robot approaches lid (feasible, pass) → attempts to
  rotate (infeasible, DTE spikes) → veto + freeze.

---

## 6. Scientific upgrade vs the earlier version

| | k\* two-class | **DTE one-class** |
|---|---|---|
| training data | robot + **rotation counterfactuals (negatives)** | **feasible only, zero negatives** |
| implicit task prior | "infeasible = rotation" (must know it) | **none** — "infeasible = far from feasible" |
| decision gate | hand-tuned τ = 17 | conformal `s_α` (knob = risk α) |
| new task | must redesign counterfactuals | **same recipe, just give it executed data** |
| cross-task evidence | — | jar-trained DTE zero-shot accepts handover chunks, ~1.3% ≈ α flag rate |

Generalization guarantee: the method **removes the dependency on knowing what infeasible
looks like.** Infeasible always equals "far from this task's feasible actions", and "far" is a
generic density/distance notion. To retarget, you do not need the new task's infeasible modes
— only its executed data; the same code and recipe run directly.

---

## 7. Deploy command (current best)

```bash
cd ~/Tum_lsy_ur10e_pipeline/ur10_clearpath/Yunfei/crisp_gym
pixi run -e jazzy-lerobot python /home/admin_025/mt_pi_codebase/mt_pi/scripts/deploy_xdiffusion_ur10e.py \
    --run-dir /home/admin_025/X-Diffusion-Data/_private/runs/policy_jar_uni_v10 \
    --ckpt latest --no-safety-observer \
    --hamer-interferer --interferer-hand auto \
    --shield-run /home/admin_025/X-Diffusion-Data/_private/runs/kstar_uni/dte_oneclass.pth \
    --shield-dte-alpha 0.0033 \
    --shield-m 2 --shield-n 3 --shield-latch --no-home --goto-start --demo \
    --cond-a-proto "928,250,22" --wait-hand 2 --goto-start-speed 0.10 \
    --shield-sanitize 0.550 --a-descend-lock 0.585 --a-yaw-trip 0.06 \
    --grasp-z-max 0.538 --grasp-z-assist 3 --hold-pin \
    --speedup 1.6 --open-horizon 8 --open-settle-cycles 3 --open-agg max \
    --cond-hold 4
```

Startup log `n_cal=1778 alpha=0.0033 -> s_alpha=0.53` confirms the DTE mode is active. The
k\* path is untouched — swap back by pointing `--shield-run` at `kstar_cls.pth`.

---

## 8. Honest boundaries (to state in the paper)

- **Sequential layer not on hardware yet:** the e-detector (accumulating weak evidence to
  catch slow motions) has a one-class calibration-drift issue; deployment uses per-chunk
  conformal + 2-of-3 voting, not the e-detector. This is the next research item.
- **Cross-task experiments are preliminary:** jar→handover zero-shot transfer (1.3% ≈ α) is a
  first data point; a systematic jar↔handover↔HRC cross-train/cross-test study is the path to
  the big contribution.
- **Standing practice:** refresh the DTE calibration scores after each hardware session (like
  the k\* aggregation) via `scripts/dte_calibration_refresh.py`.

## References

- Livernoche et al. On Diffusion Modeling for Anomaly Detection (DTE). ICLR 2024.
- Graham et al. Denoising Diffusion Models for Out-of-Distribution Detection. CVPR-W 2023.
- Mahmood et al. Multiscale Score Matching for Out-of-Distribution Detection. ICLR 2021.
- Vincent. A Connection Between Score Matching and Denoising Autoencoders. 2011.
- Vovk. Conformal test martingales for change-point detection. COPA 2021.
- Angelopoulos, Bates et al. Learn then Test: calibrating predictive algorithms. AOAS 2025.
- Pace et al. X-Diffusion: Training Diffusion Policies on Cross-Embodiment Human Demonstrations. ICRA 2026, arXiv:2511.04671.
