# The Inference-Time Feasibility Layer: Complete, Self-Contained Mathematical Exposition

*Every symbol is defined in Section 0 and used with exactly one meaning throughout.
Every numbered statement (P/L/T) is proved where stated, and every use of a prior
result cites its number. ASCII math only: `sqrt(.)`, `||v||` (Euclidean norm),
`N(m, C)` (Gaussian with mean m, covariance C), `E[.]` (expectation), `<u, w>`
(inner product). A dependency map is given in Section 11.*

---

## 0. Notation table (single source of truth)

| symbol | type | defined in | meaning |
|---|---|---|---|
| `d` | integer | §1 | feature dimension; here d = 32 |
| `x, x_0` | vector in R^d | §1 | an action chunk in feature coordinates; `x_0` emphasizes "clean" (not noised) |
| `x_k` | vector in R^d | §2, (E2) | the chunk after k noising steps |
| `k` | integer in {0,...,K} | §2 | diffusion (noise) step index |
| `K` | integer | §2 | maximum step; K = 100, so there are K+1 = 101 steps |
| `beta_i` | scalar in (0,1) | §2, (E1) | per-step noise increment of the schedule |
| `a_k` | scalar in (0,1) | §2, (E1) | cumulative signal-retention coefficient (the quantity usually written "alpha-bar"); strictly decreasing in k (P1) |
| `SNR(k)` | scalar | §2 | signal-to-noise ratio a_k / (1 - a_k) |
| `p_R` | distribution on R^d | §1 | distribution of FEASIBLE chunks (robot-executed) |
| `p_C` | distribution on R^d | §4 | distribution of the CONTRAST class used to train the discriminator (in X-Diffusion: the human dataset; in our task-adapted variant: mined rotation windows + synthetic rotation counterfactuals; the mathematics below never uses which) |
| `p^(k)` | distribution on R^d | §3, (E3) | the k-step noised marginal of a distribution p; applied as `p_R^(k)`, `p_C^(k)` |
| `m_R, m_C` | vectors in R^d | §4/§5 | class means in the isotropic Gaussian model |
| `s^2` | scalar in (0,1) | §4/§5 | common per-coordinate variance of the Gaussian model classes (data is normalized so s^2 < 1) |
| `v_k` | scalar | L1, (E4) | per-coordinate variance of a noised Gaussian class at step k; v_k = 1 - a_k (1 - s^2); strictly increasing in k (L2) |
| `b_k` | scalar | L1 | shrinkage factor sqrt(a_k) (shorthand used inside proofs) |
| `c(k, x)` | function -> [0,1] | §4 | the discriminator's output ("probability this noised chunk is feasible-class") |
| `LR_k(x)` | scalar > 0 | §4, (E6) | likelihood ratio p_R^(k)(x) / p_C^(k)(x) |
| `k_star(x)` | integer | §4, (E7) | admission time: first k with c(k, x_k) >= 1/2 (X-Diffusion Eq. 3) |
| `eps` | scalar > 0 | §4 | decision half-band of a real (finite-capacity) classifier around the 1/2 output |
| `k_star_eps(x)` | integer | §4, (E10) | eps-banded admission time (the idealized-model version of k_star) |
| `g(k)` | scalar > 0 | §4, (E9) | sqrt(a_k) / v_k; strictly decreasing in k (proved inside T1) |
| `r_R(x), r_C(x)` | scalars >= 0 | §4 | Euclidean distances ||x - m_R||, ||x - m_C|| to the two class means |
| `Delta(x)` | scalar | §4, (E8) | excess squared distance r_R(x)^2 - r_C(x)^2 |
| `q(k \| x)` | prob. vector over {0..K} | §5, (E11) | the DTE network's posterior over the noise step |
| `t_hat(x)` | scalar in [0, K] | §5, (E12) | DTE score: posterior mean of k |
| `r` | scalar >= 0 | §5 (T2 proof) | shorthand for ||x - m_R|| after centering at m_R |
| `F_1, F_2` | strictly increasing scalar functions | §6 | the two instruments' readout maps from distance to score |
| `D(x)` | scalar >= 0 | §6, §9 | Euclidean distance from x to the feasible manifold (general, non-Gaussian setting) |
| `M_A, M_B` | subsets of R^d | §9 | feasible manifolds of two tasks A (source) and B (target) |
| `phi` | parameters | §7 | DTE network parameters (student) |
| `theta` | parameters | §7 | discriminator parameters (teacher, frozen) |
| `lambda` | scalar > 0 | §7, (E13) | anchor weight in the calibration loss |
| `alpha` | scalar in (0,1) | §8 | user-chosen risk budget (tolerated false-veto rate on feasible chunks) |
| `n` | integer | §8 | number of calibration chunks |
| `S_1..S_n, S_new` | scalars | §8 | scores of calibration chunks and of a new test chunk |
| `S_(j)` | scalar | §8 | j-th smallest calibration score (order statistic) |
| `j_alpha` | integer | §8, (E14) | ceil((n+1)(1-alpha)); index of the conformal threshold order statistic |
| `s_alpha` | scalar | §8 | the conformal threshold S_(j_alpha) |
| `n_d` | integer | §8 | number of policy draws scored per control cycle (deployed: n_d = 3) |
| `T(.)` | strictly increasing scalar map | §7 (T3) | an arbitrary monotone rescaling of scores |

Conventions: "strictly increasing/decreasing" always refers to the stated argument
with all else fixed. All Gaussian-model statements (L1, L2, T1, T2) are exact in the
isotropic model defined where they are stated; §6 and §10 say precisely what is and
is not claimed beyond that model.

---

## 1. Features and problem setup

An action chunk is the policy's atomic output: 8 consecutive end-effector steps.
Its feature vector is

    x in R^d,  d = 32  =  8 steps x (3 chunk-relative euler + 1 gripper),

where "chunk-relative" means the first row's euler values are subtracted from all
rows, and a fixed physical normalization is applied (euler / 0.5 rad; gripper
{0,1} -> {-1,+1}). Two deliberate design choices:

1. **Relative orientation**: the features encode how orientation CHANGES within
   the chunk, not the absolute pose.
2. **No position channels**: in free-space tabletop manipulation, position paths
   are executable by any arm; the human-vs-gripper embodiment gap concentrates in
   orientation/finger activity. (Empirically, in our deployment history every
   position-driven veto was a false veto; every true infeasibility was carried by
   the euler channels.) This choice also drives the cross-task argument of §9.

`p_R` denotes the distribution of feasible chunks, defined **by provenance**:
chunks the robot has actually executed (teleoperation + shield-approved deployment
chunks). No feasibility labels exist anywhere in the pipeline.

The safety-layer question: given a candidate chunk x emitted by the policy at
runtime, produce a score measuring "how far is x from feasible content", and a
veto decision with a controlled false-veto rate.

## 2. The noise schedule (the shared unit system)

All instruments live on one axis: the forward diffusion process of the policy
itself. It is generated by the cosine schedule:

    f(u)    = cos( ((u + 0.008)/1.008) * (pi/2) )^2          for u in [0,1]
    beta_i  = min( 1 - f((i+1)/(K+1)) / f(i/(K+1)),  0.999 ),  i = 0..K      (E1)
    a_k     = product_{i=0..k} (1 - beta_i)

with K = 100 (hence K+1 = 101 steps, k in {0,...,K}). The forward noising
operation at step k is

    x_k = sqrt(a_k) * x_0 + sqrt(1 - a_k) * eps_noise,   eps_noise ~ N(0, I)  (E2)

**Reading (E2)**: step k keeps a fraction sqrt(a_k) of the signal and adds
Gaussian noise of standard deviation sqrt(1 - a_k) per coordinate.

**P1 (schedule property).** Every beta_i in (E1) is strictly positive, therefore
`a_k` is strictly decreasing in k. Numerically: a_0 = 0.9994, a_10 = 0.967,
a_40 = 0.639, a_100 ~ 0. Consequently SNR(k) = a_k/(1 - a_k) falls strictly
(from ~1600 at k = 0 to 0 at k = K): **k is a monotone reparametrization of the
signal-to-noise ratio.**

Because the schedule (E1) is fixed and shared by the policy, the discriminator,
and the DTE network — and is the same for every task — "k units" have a single,
task-independent physical meaning (an SNR level). This fact is used in §9(a).

## 3. The noised family (the object both instruments read)

For any distribution p on R^d, define its k-step noised marginal: the law of
(E2) when x_0 ~ p, i.e.

    p^(k)(x) = Integral over x_0 of  p(x_0) * N( x ; sqrt(a_k) x_0, (1-a_k) I ) dx_0   (E3)

As k grows, p^(k) is a progressively blurred (and slightly shrunk) version of p.
Applied to the feasible distribution this yields the **feasible ladder**
`p_R^(0), ..., p_R^(K)` — the single object that both instruments below read.

**L1 (noised marginal of an isotropic Gaussian).** If p = N(m, s^2 I) then

    p^(k) = N( sqrt(a_k) * m ,  v_k * I ),     v_k := a_k s^2 + (1 - a_k)      (E4)

*Proof.* Write b_k = sqrt(a_k). If X_0 ~ N(m, s^2 I) then b_k X_0 ~
N(b_k m, b_k^2 s^2 I) = N(b_k m, a_k s^2 I). Adding the independent noise term
of covariance (1 - a_k) I in (E2) adds the covariances:
a_k s^2 I + (1 - a_k) I = v_k I. QED.

**L2 (monotone blur).** If 0 < s^2 < 1, then v_k = 1 - a_k (1 - s^2) is strictly
increasing in k, with v_0 ~ s^2 (since a_0 ~ 1) and v_K ~ 1 (since a_K ~ 0).

*Proof.* (1 - s^2) > 0 and a_k strictly decreasing (P1) make a_k (1 - s^2)
strictly decreasing, hence v_k strictly increasing. QED.

L1 and L2 are the only two facts about the schedule that the monotonicity
theorems T1 and T2 use.

## 4. Instrument 1: the discriminator and its admission time

**Structure (brief).** A small timestep-conditioned temporal network c(k, x):
the noised chunk enters a 1-D convolution over the 8 steps, the index k enters
through a timestep embedding, and the output is one logit -> sigmoid -> a value
in [0,1]. Training: balanced binary cross-entropy; class 1 = feasible chunks
noised to a uniformly random k via (E2); class 0 = contrast chunks noised
identically.

**L3 (what BCE training converges to).** With equal class priors, the population
minimizer of the BCE at a fixed pair (k, x) is

    c*(k, x) = p_R^(k)(x) / ( p_R^(k)(x) + p_C^(k)(x) )                        (E5)

*Proof.* At fixed (k, x) write A = p_R^(k)(x), B = p_C^(k)(x). The pointwise
objective is J(c) = -A log c - B log(1 - c). Then J'(c) = -A/c + B/(1-c), which
vanishes at c = A/(A+B); J''(c) = A/c^2 + B/(1-c)^2 > 0 confirms a minimum.
(This presumes the network can realize the pointwise optimum — the standard
infinite-capacity idealization, stated as such in §10.) QED.

Define the likelihood ratio

    LR_k(x) = p_R^(k)(x) / p_C^(k)(x)                                          (E6)

By (E5): c*(k,x) >= 1/2  if and only if  LR_k(x) >= 1. X-Diffusion's admission
time (their Eq. 3) is

    k_star(x) = min{ k in {0..K} :  c(k, x_k) >= 1/2 }                          (E7)

— at the population optimum, **the first blur level at which the feasible density
overtakes the contrast density at x**.

### T1: the admission time is monotone in distance (isotropic model)

**Model and coordinates.** Let p_R = N(m_R, s^2 I), p_C = N(m_C, s^2 I) with
m_R != m_C and 0 < s^2 < 1. Translate the coordinate origin to the midpoint
(m_R + m_C)/2. This is legitimate because translating all of x, m_R, m_C by the
same vector changes neither Euclidean distances nor the value of LR_k at the
corresponding point; it is a bookkeeping choice, used ONLY inside this proof.
After it, ||m_R|| = ||m_C||. Write u = m_R - m_C (nonzero). Define

    r_R(x) = ||x - m_R||,   r_C(x) = ||x - m_C||,
    Delta(x) = r_R(x)^2 - r_C(x)^2        (excess squared distance)            (E8)

Interpretation of Delta: how much farther (in squared distance) x sits from the
feasible mean than from the contrast mean. Delta > 0 = "on the contrast side".

**Claim (T1).** Define g(k) = sqrt(a_k) / v_k                                  (E9)
and, for a decision half-band eps > 0,

    k_star_eps(x) = min{ k :  |log LR_k(x)| <= eps }                           (E10)

Then (i) log LR_k(x) = -(1/2) g(k) Delta(x) exactly; (ii) g is strictly
decreasing in k with g(K) ~ 0; (iii) therefore |log LR_k(x)| decays strictly and
monotonically to 0 in k ("monotone admission"), the set in (E10) is nonempty,
and k_star_eps(x) is a nondecreasing step function of |Delta(x)| — strictly
increasing under the real-valued relaxation of k.

*Proof.*

Step 1 (exact form of the log-ratio). By L1, p_R^(k) = N(b_k m_R, v_k I) and
p_C^(k) = N(b_k m_C, v_k I) with the SAME variance v_k (equal s^2). The Gaussian
normalizing constants therefore cancel in the log-ratio:

    log LR_k(x) = [ ||x - b_k m_C||^2 - ||x - b_k m_R||^2 ] / (2 v_k)

Expand the numerator, using ||m_R|| = ||m_C|| (midpoint origin):

    ||x - b_k m_C||^2 - ||x - b_k m_R||^2
      = ( ||x||^2 - 2 b_k <x, m_C> + b_k^2 ||m_C||^2 )
      - ( ||x||^2 - 2 b_k <x, m_R> + b_k^2 ||m_R||^2 )
      = 2 b_k <x, m_R - m_C>  =  2 b_k <x, u>

Meanwhile, expanding (E8) with the same origin:

    Delta(x) = ( ||x||^2 - 2<x, m_R> + ||m_R||^2 ) - ( ||x||^2 - 2<x, m_C> + ||m_C||^2 )
             = -2 <x, m_R - m_C>  =  -2 <x, u>

So the numerator equals 2 b_k <x, u> = -b_k Delta(x), giving exactly

    log LR_k(x) = - ( sqrt(a_k) / (2 v_k) ) * Delta(x) = -(1/2) g(k) Delta(x)   (claim i)

Step 2 (g strictly decreasing). Define phi(a) = sqrt(a) / (1 - a(1 - s^2)) on
a in (0,1), so that g(k) = phi(a_k). Its derivative has numerator

    (1/(2 sqrt(a))) * (1 - a(1-s^2))  +  sqrt(a) * (1 - s^2)

The first term is positive because 1 - a(1-s^2) = v > 0; the second is positive
because s^2 < 1. Hence phi'(a) > 0: phi is strictly increasing in a. Since a_k
is strictly decreasing in k (P1), g(k) = phi(a_k) is strictly decreasing in k.
Boundary values: g(0) = sqrt(a_0)/v_0 ~ 1/s^2; g(K) = sqrt(a_K)/v_K ~ 0
(a_K ~ 0, v_K ~ 1). (claim ii)

Step 3 (monotone admission and the crossing time). By (i),
|log LR_k(x)| = (1/2) g(k) |Delta(x)|, which by (ii) strictly decreases in k and
tends to (1/2) g(K) |Delta| ~ 0. Hence for any eps > 0 the condition
|log LR_k| <= eps, equivalently

    g(k) <= 2 eps / |Delta(x)|,

fails for small k (when 2eps/|Delta| < g(0)) and holds from some step onward;
the set in (E10) is nonempty and is an "upper tail" {k >= threshold} because g
is decreasing. Therefore

    k_star_eps(x) = min{ k : g(k) <= 2 eps / |Delta(x)| }.

Now increase |Delta(x)|: the right-hand side 2eps/|Delta| strictly decreases;
because g is strictly decreasing, the smallest k satisfying the inequality can
only stay or move upward — and moves strictly upward under the real-valued
relaxation k in [0, K] (where k_star_eps = g^{-1}(2 eps/|Delta|) with g^{-1}
strictly decreasing, composed with the strictly decreasing map
|Delta| -> 2eps/|Delta|, i.e. strictly increasing overall). If
2eps/|Delta| >= g(0), then k_star_eps = 0 (immediately inside the band). QED.

**Why the eps-band is needed (honesty note).** In the equal-covariance ideal
model, sign(log LR_k) = -sign(Delta) for ALL k: the ratio approaches 1 from one
side and c crosses 1/2 only in the limit k -> K. A finite crossing (E7) arises
in reality from finite capacity/data (an effective decision band |logit| <= eps)
and from unequal class covariances. (E10) captures this inside the ideal model;
it is the idealization's device, not a property of real data.

**Corollary T1-c (link to Euclidean distance).** k_star_eps is stated as
monotone in the EXCESS squared distance Delta(x) = r_R^2 - r_C^2. If x varies so
that its distance to the contrast mean r_C stays fixed (e.g., x moves away from
m_R along a sphere centered at m_C), then Delta = r_R^2 - const, and k_star_eps
is strictly increasing in the Euclidean distance r_R to the feasible mean.
**Note what this reveals**: the discriminator's admission time inherently
measures distance-to-feasible RELATIVE to the contrast — it needs p_C to be
defined. This motivates the second instrument.

## 5. Instrument 2: the DTE network and its diffusion-time estimate

**Structure (brief).** A temporal 1-D CNN over the (4-channel x 8-step) chunk
(3 conv layers, 128 channels), mean+max pooled over time, then a 2-layer MLP
head producing K+1 = 101 logits — one per candidate noise step; ~0.19 M
parameters. Unlike the discriminator, it receives NO k input: its job is to
PREDICT k. Softmax of the logits is the estimated posterior q(k | x).

**Training (self-supervised, feasible data only).** Sample x_0 ~ p_R, sample
k ~ Uniform{0..K}, form x_k by (E2), train with cross-entropy to recover k from
x_k. The only "label" is the k we ourselves drew when adding noise: no human
annotation, no negative class, no contrast data.

**L4 (what CE training converges to).** The training pairs (k, x_k) have joint
density P(k) p_R^(k)(x) with P(k) = 1/(K+1). The population CE minimizer is the
true posterior

    q(k | x) = p_R^(k)(x) / ( sum over j=0..K of p_R^(j)(x) )                  (E11)

*Proof.* For a fixed input x, choosing the predictive vector q(.) minimizes
- sum_k P(k|x) log q(k) = H(P(.|x)) + KL(P(.|x) || q). KL >= 0 with equality
iff q = P(.|x) (Gibbs' inequality). Under the infinite-capacity idealization
the minimization is pointwise in x, giving (E11); the uniform prior P(k) cancels
in the posterior. QED.

The score is the posterior mean

    t_hat(x) = sum over k of  k * q(k | x)                                      (E12)

At inference the CLEAN candidate chunk is fed. Intuition for what (E11)-(E12)
do: the network's entire worldview is "every input is a feasible chunk plus some
amount of noise"; content it cannot recognize as near-clean feasible content can
only be explained as heavily-noised feasible content, pushing the posterior —
and hence t_hat — toward large k.

### T2: the diffusion-time estimate is monotone in distance (isotropic model)

**Model and coordinates.** Let p_R = N(m_R, s^2 I) with 0 < s^2 < 1. Translate
the origin to m_R (again a bookkeeping choice, independent of T1's choice and
used only inside this proof), so p_R = N(0, s^2 I). Write r = ||x|| — the
Euclidean distance of the chunk from the feasible mean.

**Claim (T2).** t_hat depends on x only through r, and is strictly increasing
in r. No contrast distribution appears anywhere.

*Proof.*

Step 1 (posterior depends only on r). By L1 with m = 0:
p_R^(k) = N(0, v_k I), so

    p_R^(k)(x) = (2 pi v_k)^(-d/2) * exp( - r^2 / (2 v_k) ),

a function of r alone. By (E11), q(k | x) = q(k | r) with

    q(k | r) = C(r)^(-1) * v_k^(-d/2) * exp( - r^2 / (2 v_k) ),

C(r) the normalizer.

Step 2 (monotone likelihood ratio in k). Fix r_2 > r_1 >= 0 and define

    h(k) = q(k | r_2) / q(k | r_1)
         = [ C(r_1)/C(r_2) ] * exp( - (r_2^2 - r_1^2) / (2 v_k) ).

The bracket is a positive constant in k. The exponent is negative with magnitude
(r_2^2 - r_1^2)/(2 v_k), which strictly SHRINKS as v_k grows; by L2, v_k is
strictly increasing in k. Hence h(k) is strictly increasing in k.

Step 3 (single crossing). Both q(.|r_2) and q(.|r_1) sum to 1 over k. If
h(k) >= 1 for every k, then q(k|r_2) >= q(k|r_1) pointwise with equal sums,
forcing equality everywhere — contradicting strict increase of h. So h < 1
somewhere; symmetrically h > 1 somewhere. Since h is strictly increasing, the
set {k : h(k) >= 1} is an upper interval: define k_0 = min{k : h(k) >= 1}
(nonempty, and k_0 >= 1 because h(0) is the minimum of h and h < 1 somewhere
implies h(0) < 1). Then

    k <  k_0  =>  q(k | r_2) <  q(k | r_1)
    k >= k_0  =>  q(k | r_2) >= q(k | r_1)

Step 4 (stochastic dominance). Let F_i(j) = sum_{k <= j} q(k | r_i) be the two
cumulative distributions. For j < k_0, every summand of F_2(j) is strictly below
the corresponding summand of F_1(j), so F_2(j) < F_1(j). For j >= k_0, write
F_i(j) = 1 - sum_{k > j} q(k | r_i); every term of the tail sum for r_2
dominates the corresponding term for r_1 (those k are >= k_0), so
F_2(j) <= F_1(j). Hence F_2(j) <= F_1(j) for ALL j, with strict inequality for
j = k_0 - 1: the posterior at the larger distance first-order stochastically
dominates the one at the smaller distance.

Step 5 (posterior mean). For a random variable k on {0..K} with cdf F, the tail
formula gives E[k] = sum_{j=0..K-1} (1 - F(j)). Therefore

    t_hat(r_2) - t_hat(r_1) = sum_{j=0..K-1} [ F_1(j) - F_2(j) ]  >  0,

the sum being of nonnegative terms with at least one (j = k_0 - 1) strictly
positive. So t_hat is strictly increasing in r. QED.

**Contrast between T1 and T2.** T1's monotonicity is in the RELATIVE quantity
Delta(x) (needs the contrast class); T2's is directly in the distance r to the
feasible distribution (needs nothing else). This asymmetry is the mathematical
form of "the DTE is the foil-free instrument".

## 6. Equivalence and the identifiability gap

Write D(x) for the distance of x to the feasible content (in the Gaussian model,
D = r_R; for real data, the distance to the feasible manifold — see the honesty
note below). T1 and T2 say: on their common domain of validity,

    k_star_eps(x) = F_1( D(x) ),      t_hat(x) = F_2( D(x) )

with F_1, F_2 strictly increasing but otherwise DIFFERENT functions (F_1 also
carries the dependence on the contrast through Delta; along the slices of
Corollary T1-c it is a strictly increasing function of D alone). Two
consequences, both measured on real data:

1. **Order must agree.** Ideal instruments would have rank correlation 1;
   measured on 1,649 real chunks: Spearman 0.88 in the discriminative range
   (the gap to 1 is model/estimation error — the propositions are exact only in
   the isotropic model).
2. **Values need not agree.** Measured pre-calibration: the same rotation
   content reads k_star ~ 44 on the discriminator, t_hat ~ 7.1 on the raw DTE.

**The identifiability statement (why the scale of t_hat is arbitrary).** By
(E11), the DTE posterior — hence t_hat — is a functional of the feasible family
{p_R^(k)} alone; its numeric values depend on the SHAPE of p_R (the spread s
enters v_k in T2's proof and thereby sets how fast the posterior migrates with
distance). The self-supervised objective can verify nothing about those values
beyond ordering: any instrument of the form (strictly increasing map) applied to
D(x) is indistinguishable from the "correct" one by rank alone. **A one-class
self-supervised score has a well-defined ranking and an arbitrary scale.**

k_star is different in one respect that matters: its scale has an OPERATIONAL
definition inside the framework being extended. X-Diffusion's training rule
(their Eq. 4) supervises a human action A only at steps k >= k_star(A); hence
"k units" are the units in which the training-time admission rule is written,
and a chunk with admission time above some level tau is a chunk whose
fine-grained (low-noise) supervision was withheld during training. A runtime
gate that claims to enforce THE SAME rule must speak the same units. That is
the purpose of calibration.

**Honesty note (beyond the Gaussian model).** For real, manifold-supported p_R,
the exact statements T1/T2 become: the same computations hold leaf-wise under a
mixture-of-Gaussians approximation of p_R, and the rigorous manifold analysis of
the diffusion-time posterior (its dependence on x through the distance to the
data) is given in Livernoche et al., ICLR 2024. We treat the general case as:
exact theorem in the model + cited analysis + direct empirical validation (the
0.88 rank agreement above, and the graded-intensity probes of §9).

## 7. Calibration: what anchoring on the teacher does

**Procedure.** Collect an unlabeled pool of source-task chunks (feasible +
human content; no feasibility labels), stamp each pool chunk x with the frozen
teacher's k_star_theta(x), and train the student phi with

    Loss(phi) =  E over (x_0 ~ p_R, k ~ Unif) [ CE( f_phi(x_k), k ) ]
              +  lambda * E over pool [ ( t_hat_phi(x)/K - k_star_theta(x)/K )^2 ]   (E13)

The first term is the unchanged self-supervised DTE loss; the second pulls the
student's output toward the teacher's value on the pool. (Scores are divided by
K purely to normalize both to [0,1] in the loss.)

**First-order account (the scale).** Suppose (idealization) the teacher is an
exact function of distance, k_star = F_1(D(x)), and the student is constrained
to the family { G(D(x)) : G strictly increasing } that its own objective leaves
undetermined (§6). Then the anchor term is minimized by G = F_1 on the range of
D covered by the pool: the anchor composes the student's readout with the
monotone correction F_1 o F_2^{-1}. Prediction: values move onto the teacher's
axis; rankings do not. Measured: rotation content 7.1 -> 49.4 (teacher: 44);
the student's own before/after rank agreement is 0.979.

**T3 (rank-based decisions are invariant to monotone rescaling).** Let T be any
strictly increasing map. The conformal decision of §8 computed from scores
T(S_1..S_n), T(S_new) is IDENTICAL to the one computed from S_1..S_n, S_new.

*Proof.* A strictly increasing map preserves order, hence maps the j-th
smallest calibration score to the j-th smallest transformed score:
T(S)_(j) = T(S_(j)). The decision compares S_new with S_(j_alpha);
T(S_new) > T(S_(j_alpha)) iff S_new > S_(j_alpha). QED.

**Consequences of T3, and what the measurements then prove.** The per-chunk
conformal layer (§8) — and also any layer consuming only conformal p-values,
since p-values are rank statistics — CANNOT change under a pure monotone
recalibration. Measured: the per-chunk veto power was 86.4% before and after
calibration, exactly as T3 requires. But the sequential evidence-accumulation
layer improved substantially (slow-rotation miss 17% -> 1%, delay 5 -> 2 cycles;
feasible-stream false alarms 12/40 -> 7/40). By T3-logic this is possible ONLY
because the anchor's effect was NOT purely monotone: and indeed the direct
probe shows 2.1% of rank reorderings, concentrated where the teacher's k_star
is high (the mean |score shift| grows from 0.04 in the clearly-feasible bin to
3.02 in the near-rotation bin), and the feasible-vs-rotation margin widens by
about 2.4x. Interpretation: beyond the monotone scale correction (the semantic
purpose), the anchor injects the teacher's finer boundary resolution — acquired
from its contrast training and its invariances — exactly in the band where the
one-class geometry ranks poorly; the sequential layer, which compounds mid-band
evidence across cycles, amplifies precisely that band. **Scoped conclusion: the
teacher transfers its scale and its margins; not the global ranking, and not
the single-chunk operating point.**

## 8. From score to decision: finite-sample guarantees, no hand-tuned gate

Let s(.) be the deployed score (the calibrated t_hat). Take n held-out EXECUTED
(hence feasible) chunks with scores S_1..S_n, and a new chunk with score S_new.

**T4 (split-conformal false-veto guarantee).** Assume S_new is exchangeable
with S_1..S_n (the new chunk is genuinely feasible, from the same process), and
assume ties have probability zero (or are broken conservatively). Set

    j_alpha = ceil( (n+1)(1 - alpha) ),      s_alpha = S_(j_alpha)             (E14)

and veto iff S_new > s_alpha. Then P( false veto ) <= alpha.

*Proof.* Under exchangeability of the n+1 scores, the rank of S_new among them
is uniform on {1, ..., n+1}. The veto event S_new > S_(j_alpha) requires the
rank of S_new to be at least j_alpha + 1, an event of probability

    (n + 1 - j_alpha) / (n + 1)  <=  (n+1 - (n+1)(1-alpha)) / (n+1)  =  alpha.  QED.

The ONLY free parameter is alpha — an interpretable risk budget ("what fraction
of feasible actions I tolerate pausing") — not a threshold in score units. This
replaces both the fixed 1/2 crossing of (E7) and any hand-tuned score gate.

**T5 (deployed statistic; Bonferroni over draws).** Deployment scores n_d = 3
policy draws per cycle and votes on the WORST draw: flag the cycle iff
max over draws of s(x^(draw)) exceeds the threshold. If each draw is marginally
exchangeable with the calibration scores when the policy is in a feasible mode,
then thresholding at level alpha/n_d per draw, i.e. using s_(j_(alpha/n_d)),
gives per-cycle false-flag probability

    P( max of n_d draws exceeds s_(alpha/n_d) )
      <= sum over draws of P( draw exceeds )      (union bound)
      <= n_d * (alpha / n_d) = alpha.                                          QED.

Hence the deployed level alpha/n_d = 0.01/3 ~ 0.0033. (Calibrating at the
single-draw alpha while voting on the worst of three was the measured cause of
a false-veto incident; the general lesson: calibrate the statistic actually
deployed.)

## 9. Cross-task generalization: the precise sense of "universal"

The trained instrument is one fixed function s : R^d -> [0, K]. The universality
claim splits into three parts of deliberately different epistemic strength.

**(a) Units are task-invariant BY CONSTRUCTION.** The k-axis is defined by the
schedule (E1), which is the same for every task (same forward process under
every policy). "k = 40" denotes the same SNR everywhere. Nothing to test.

**(b) Geometry transfers UNDER A STATED, TESTED ASSUMPTION.** T2 makes s
monotone in the distance D_A(x) = dist(x, M_A) to the SOURCE task's feasible
manifold. For a target task B we need monotonicity in D_B(x) = dist(x, M_B).
The bridge is:

    Assumption A1 (shared feasible core): in chunk-relative orientation+grip
    coordinates, the feasible manifolds of tabletop end-effector tasks
    approximately coincide: M_A ~ M_B, hence D_A(x) ~ D_B(x).

Mechanism for A1 (this is §1's feature design paying off): what ANY arm
executes looks alike in these coordinates — near-quiet relative orientation
plus stereotyped grip transitions — while task identity lives largely in the
excluded position channels; embodiment-infeasible content is exactly the
coherent orientation/finger activity these coordinates isolate. Under A1 the
fixed instrument is automatically (approximately) monotone in D_B: transfer
follows from geometry with no retraining and no task-B contrast class. A1 is
TESTED, not assumed: (i) acceptance — target-task feasible chunks score low
(jar-trained instrument on handover: false-alarm ~ alpha under an exchangeable
split, 1.3%); (ii) unit consistency — identical synthetic rotation intensities
read nearly identical values on both tasks' content (rate 0.05 rad/step: 10.9
vs 9.3; rate 0.12: 42.6 vs 36.4), the measured form of "s is a function of the
distance alone, not of the task"; (iii) detection — 98-100% veto of graded
infeasible probes built on target-task content.

**(c) The decision guarantee transfers UNCONDITIONALLY, by re-zeroing.**
Whatever residual shift A1 leaves, T4 applies verbatim to a calibration set
drawn from the target task's own executed chunks (about ten runs, label-free,
no negatives): the false-veto guarantee is exact per task after this re-zeroing.
The asymmetry is the honest structure of the claim: **the safety (false-alarm)
guarantee holds exactly on every task after re-zeroing; the detection power
transfers to the degree A1 holds, which we validate empirically.**

Metrology summary of the whole architecture: the discriminator is the PRIMARY
STANDARD (exists once, where the framework trained; its contrast class gives
the unit its operational meaning via §6); the DTE is the WORKING INSTRUMENT
(foil-free, carried to any task); calibration (E13) transfers the unit once;
each new task performs only a conformal RE-ZEROING (T4) on its own executed
data. The standard is never rebuilt — answering "do we need a per-task
classifier" in the negative.

## 10. Assumptions and limits (the honesty box)

1. L3 and L4 use the infinite-capacity, population-limit idealization (networks
   realize the pointwise Bayes optimum).
2. T1 and T2 are exact in the isotropic Gaussian model; the equal-covariance
   case of T1 needs the eps-band device (E10) for a finite crossing, reflecting
   real classifiers' decision bands and unequal real covariances. The general
   manifold case is argued by mixture approximation, cited (Livernoche et al.,
   ICLR 2024), and validated empirically (0.88 rank agreement; graded probes).
3. T4 assumes exchangeability; within-episode correlation makes per-chunk
   validity approximately marginal (empirically nominal), and the per-task
   re-zeroing needs on the order of ten runs — with a single calibration run,
   run-level shift inflates false alarms (measured: 7.6-13.6% vs 1.3% under an
   exchangeable split).
4. A1 is an empirical regularity of free-space tabletop end-effector tasks with
   a mechanistic argument, not a theorem; contact-rich or position-constrained
   feasibility would require extending the feature space, after which A1 must
   be re-validated.
5. The calibration (E13) requires the contrast-trained teacher once, on one
   task; its measured benefit is confined to scale and margins (sequential
   layer). The per-chunk operating point is provably (T3) unaffected by the
   monotone part and measured unaffected overall.

## 11. Dependency map

    P1 (a_k strictly decreasing)  --------+
                                          |--> L2 (v_k strictly increasing) --+
    L1 (Gaussian noised marginal) --------+                                   |
                                                                              |
    L3 (BCE -> density ratio) --> (E7) k_star  --> T1 (monotone in Delta) <---+
    L4 (CE -> posterior)      --> (E12) t_hat  --> T2 (monotone in r)     <---+
                                                                              
    T1 + T2 --> §6 (two monotone readouts F_1, F_2 of one distance;
                    scale identifiability gap; measured rank agreement 0.88)
    §6 --> §7 (calibration selects F_1's units; T3 predicts per-chunk
              invariance; measured sequential gains = non-monotone residual)
    T4 (conformal) --> §8 decisions; T5 (Bonferroni) --> deployed statistic
    T2 + A1 --> §9(b) geometric transfer;  T4 --> §9(c) per-task guarantee
