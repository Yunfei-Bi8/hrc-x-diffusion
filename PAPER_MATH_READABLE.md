# Mathematical Foundations — Plain-Notation Version

Readable companion to `PAPER_STORY_XDIFFUSION_EXTENSION.md` §1.5. Same content, ASCII math
(no LaTeX markup). One criterion — the *diffusion admission time* — read two ways, and
how the X-Diffusion discriminator and the DTE network relate on it.

---

## Notation

    x           an action chunk as a feature vector (here 32 numbers = 8 steps x 4
                channels of chunk-relative euler + gripper)
    p_R         distribution of FEASIBLE actions (robot-executed chunks)
    p_H         distribution of HUMAN actions
    k           diffusion noise step, from 0 (clean) up to K = 100 (almost pure noise)
    a_k         "noise-retention" coefficient (alpha_bar_k); a_0 ~ 1, a_K ~ 0, decreasing in k
    Normal(mean=m, cov=C)   a Gaussian density with that mean and covariance
    I           the identity matrix
    ||v||       Euclidean length of vector v

Forward noising (how a clean chunk x0 becomes a noisy chunk x_k):

    x_k  =  sqrt(a_k) * x0  +  sqrt(1 - a_k) * noise ,     noise ~ Normal(0, I)

So step k keeps a fraction sqrt(a_k) of the signal and adds Gaussian noise of variance
(1 - a_k). Small k = barely touched; large k = mostly noise.

---

## 1. The shared object: the noised family (a "blur ladder")

For ANY action distribution p, its noised version at step k is:

    p_k(x)  =  INTEGRAL over x0 of   p(x0) * Normal( x | mean = sqrt(a_k)*x0, cov = (1-a_k)*I )  dx0

In words: **take p, shrink it toward the origin by sqrt(a_k), then blur it with a Gaussian
of width sqrt(1 - a_k).** As k grows, p gets blurrier. The whole ladder

    p_0 ,  p_1 ,  p_2 ,  ... ,  p_K        (a Gaussian scale-space)

is the single object both networks read. Apply it to the feasible distribution p_R and you
get the feasible ladder  p_R,0, p_R,1, ..., p_R,K.

---

## 2. Reading #1 — X-Diffusion's k* (needs a human foil)

Train a classifier to tell, at each noise level k, whether a noised action came from the
robot or the human. The best possible such classifier is the Bayes posterior:

    c(k, x)  =  p_R,k(x)  /  ( p_R,k(x) + p_H,k(x) )

and

    c(k, x) >= 0.5     is the same as     ratio(k, x) := p_R,k(x) / p_H,k(x)  >=  1 .

X-Diffusion's admission time is the first noise level where the (blurred) robot density
overtakes the (blurred) human density at x:

    k*(x)  =  smallest k such that  c(k, x) >= 0.5 .

This is a RELATIVE reading — it requires the human distribution p_H as a foil.

### Proposition A (why k* is a monotone distance, Gaussian case)

Let p_R = Normal(mu_R, s^2 * I) and p_H = Normal(mu_H, s^2 * I). Define how much closer x
sits to the human mean than the robot mean:

    Delta(x)  =  ||x - mu_R||^2  -  ||x - mu_H||^2 .

Let the per-step blurred variance be  v_k = a_k * s^2 + (1 - a_k).  Then:

    log ratio(k, x)  =  -( a_k / (2 * v_k) ) * Delta(x)     (approximately)

and the factor  a_k / v_k  strictly DECREASES as k grows (whenever s^2 < 1, i.e. normalized
data). Two consequences:

    (i)  |log ratio(k, x)| shrinks smoothly to 0 as k grows.
         => admission is monotone in k, so "first crossing" is well defined.
         (This is the population fact our monotone regularizer enforces in practice.)

    (ii) the crossing time is an INCREASING function of Delta(x).
         => actions farther (in excess) from the robot class are admitted only at
            proportionally larger noise levels.

---

## 3. Reading #2 — DTE's t_hat (no foil)

Now train a classifier to guess the NOISE LEVEL k itself, given a noised FEASIBLE chunk
(101-way: "which rung of the ladder does this look like?"). The best possible such
classifier is the posterior over the noise ladder of the feasible family ALONE:

    q(k | x)  =  p_R,k(x)  /  ( SUM over j=0..K of  p_R,j(x) )

and the score is the expected noise level:

    t_hat(x)  =  E[ k | x ]  =  SUM over k of  k * q(k | x) .

This is an INTRINSIC reading — no p_H appears anywhere. The ladder itself supplies the
contrast.

### Proposition B (why t_hat is a monotone distance, Gaussian case)

Let p_R = Normal(0, s^2 * I) with s^2 < 1, and let r = ||x|| be the distance from the
feasible mean. Then p_R,k = Normal(0, v_k * I) with v_k increasing in k, and for any two
distances r1 < r2:

    q(k | r2) / q(k | r1)   is proportional to   exp( -(r2^2 - r1^2) / (2 * v_k) ),

which INCREASES with k. So a farther point puts more posterior mass on higher noise levels,
and therefore:

    t_hat(x)  strictly increases with the distance r from the feasible distribution.

(For non-Gaussian, manifold-supported p_R, Livernoche et al. ICLR 2024 derive the analytic
posterior and show it depends on x through its distance to the data — same behavior.)

### The equivalence (the crux of "extension, not combination")

Propositions A and B say the same thing about both scores:

    k*(x)     increases with distance-to-feasible
    t_hat(x)  increases with distance-to-feasible

both measured IN THE SAME NOISE-SCHEDULE UNITS. They are two readouts of one quantity.
Measured on 1,649 real chunks: Spearman rank correlation between k* and t_hat = 0.765
overall, 0.881 within the discriminative range.

---

## 4. One family, two contrast schemes (the NCE / TRE view)

Both are noise-contrastive estimators on the same ladder:

    X-Diffusion:  at each rung k, contrast   p_R,k   against   p_H,k      (human is the foil)
    DTE:          contrast the rungs   p_R,0, p_R,1, ... , p_R,K   against each other
                  (the ladder is its own foil)

The optimal DTE logits recover  log p_R,k(x)  up to a constant that depends only on x. So
dropping the human foil does NOT change WHAT is estimated (the feasible ladder) — only HOW
it is estimated. That is the exact sense in which the DTE monitor is *foil-free*, not
*method-new*.

---

## 5. Turning a score into a decision (no hand-tuned gate)

Let s(x) be either score (t_hat in the deployed system). Take a calibration set of
held-out EXECUTED (hence feasible) chunks and their scores:

    S = { s(x_1), s(x_2), ... , s(x_n) } .

### Proposition C (conformal false-veto guarantee)

Sort S and set the threshold to the  ceil( (n+1)*(1 - alpha) )-th smallest value:

    s_alpha  =  the (1 - alpha) quantile of S .

Then for a new feasible chunk x_new (exchangeable with the calibration chunks):

    P( s(x_new) > s_alpha )  <=  alpha .

This is finite-sample and assumption-free. The ONLY free parameter is alpha — an
interpretable risk budget ("fraction of feasible actions I tolerate pausing"), NOT a
score-unit magic number. (This replaces X-Diffusion's fixed 0.5 crossing / our old tau=17.)

### Proposition D (calibrate the statistic you actually deploy — Bonferroni)

The deployed vote is PESSIMISTIC over m = 3 policy draws per cycle:

    flag this cycle  if   max over the m draws of s(x^draw)  >  s_{alpha/m} .

By a union bound the per-cycle false-flag probability is <= alpha. This is why the deployed
level is  alpha / m = 0.01 / 3 ~= 0.0033  (not 0.01). Using the single-draw alpha with a
worst-of-3 vote was the exact cause of the 2026-08-19 false vetoes.

---

## 6. How the two networks interact: k*-anchored distillation

The discriminator (teacher, FREE — X-Diffusion training already produces it, then discards
it) supervises the DTE student's noise axis. Train the student on UNLABELED source-task
actions (feasible chunks + human demos; no feasibility labels) with:

    L(phi)  =  CE( predict the noise level ;  executed data, positive-only augmentation )
             +  lambda * average over x of  ( t_hat(x)/100  -  k*_teacher(x)/100 )^2

    term 1 = ordinary foil-free DTE training (needs feasible data only)
    term 2 = pulls the student's diffusion-time output onto the teacher's calibrated k* axis

At a NEW task the teacher is absent by design: the student adapts self-supervised on that
task's executed data while keeping the axis and invariances it inherited. The discriminator
X-Diffusion throws away is exactly the teacher that calibrates the deployable monitor.

---

## 7. Why both gates are needed (lifecycle risk)

Per control cycle, for a stochastic policy pi and a monitor M:

    P( execute an infeasible action )
        =  P_pi( emit an infeasible action )    x    P_M( miss it | emitted )
           \_______ training-time gate ______/       \___ execution-time gate ___/
                    shrinks this factor                     bounds this factor

Training-time admission (X-Diffusion) shapes the DISTRIBUTION — it cannot drive the first
factor to zero for a sampler (measured: 12 of 12 unshielded rollouts emit rotation), and
deleting the infeasible data instead replaces detectable intent with undefined behavior
(25% of rollouts physically penetrate the lid cylinder in the hard-filter ablation). The
execution-time gate controls the second factor with the conformal guarantee above. The two
gates MULTIPLY — they are complements on one criterion, not alternatives.
