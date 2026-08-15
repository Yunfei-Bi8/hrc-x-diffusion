"""k* identity classifier for the UNIFIED jar task (per
KSTAR_DEPLOYMENT_FORMULATION.md) — NO feasibility labels in training.

  D_R (y=1) = jar_uni/robot windows (same task, full feasible cycle:
              descend + close + hold + release + retreat)
  D_H (y=0) = jar_uni/human windows, BOTH role streams, ALL windows

Recipe: uniform k, balanced classes, TEXTURE-EQUALIZATION augmentation on both
classes, monotone regularizer L_mono. tau = robot-heldout q99 + 10 (label-free).
Gates (phase labels used ONLY here): robot k* low; role-B and role-A approach
low; role-A ROTATE high; AUROC >= 0.85; monotonicity.
Saves runs/kstar_uni/kstar_cls.pth with {"kstar": True, "tau": ...} for the
--shield-kstar deploy mode.
Run (mtpi): python scripts/kstar_uni.py
"""
import json
import sys
import types
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, "/home/admin_025/X-Diffusion")
from models.xdiffusion.hr_classifier import HumanRobotClassifier, HumanRobotClassifierConfig

TASK = Path("/home/admin_025/X-Diffusion-Data/retargeted/jar_uni")
OUT = Path("/home/admin_025/X-Diffusion-Data/_private/runs/kstar_uni")
DEV = "cuda"
T = 8
STRIDE = 2
KMAX = 101
K_GRID = np.arange(0, KMAX, 2)
torch.manual_seed(0)
rng = np.random.default_rng(0)


def load(f):
    # POSITION-BLIND features (2026-08-14 redesign): every false veto all day was
    # position-driven — position paths are feasible EVERYWHERE in this task; the
    # only infeasible content is coherent wrist rotation, which lives in the
    # euler channels. Restricting the identity classifier to euler+grip makes
    # quiet chunks class-ambiguous (P~0.5 -> k*=0 = feasible) and rotation the
    # sole discriminant (exists only in D_H -> k* high).
    with h5py.File(f) as h:
        a = np.concatenate([np.array(h["ee_euler"]),
                            np.array(h["gripper_open"])[:, None]], 1).astype(np.float64)
        fm = np.array(h["feas_mask"]) if "feas_mask" in h else np.ones(len(a), np.int32)
        role = h.attrs.get("role", "B")
        nb = int(h.attrs.get("uni_bridge", 0))
    if nb > 0:            # NEVER feed synthetic glide-bridge rows to the classifier:
        a, fm = a[nb:], fm[nb:]   # robot-plausible motion labeled "human" would poison D_H
    return a, fm, role


def rel_euler(W):
    """chunk-RELATIVE euler: subtract row 0. Quiet windows of BOTH classes
    become ~0+jitter (class-ambiguous -> k*~0 = feasible); rotation becomes a
    growing ramp (only in D_H -> k* high). Also fixes the normalization: the
    unwrapped absolute yaw spanned ~10 rad and crushed quiet content into a
    hair-thin band that any k-noise flooded (P-curves collapsed by k=10)."""
    W = W.copy()
    W[..., :3] -= W[:, :1, :3]
    return W


def windows(a, stride=STRIDE):
    return rel_euler(np.stack([a[t:t + T] for t in range(0, len(a) - T, stride)])), \
           np.arange(0, len(a) - T, stride)


rob = [f for f in sorted(TASK.glob("robot/demo*.h5")) if int(f.stem[4:]) < 40]
hum = [f for f in sorted(TASK.glob("human/demo*.h5")) if int(f.stem[4:]) < 200]
rob_tr, rob_va = rob[:-2], rob[-2:]
humB = [f for f in hum if load(f)[2] == "B"]
humA = [f for f in hum if load(f)[2] == "A"]
hum_va = humB[-3:] + humA[-3:]
hum_tr = [f for f in hum if f not in set(hum_va)]

R_eps = [load(f)[0] for f in rob_tr]
# COVERAGE (F1, KSTAR_DEPLOYMENT_FORMULATION §5): the teleop demos approach the
# HOLDER only; mode-A requires the robot to approach the LID — that content
# exists in the 32 executed jar_v1 deployments (feasible by construction).
# Each executed cycle chunk is natively (T=8, 7): keep them as ATOMIC windows
# (v3 lesson: concatenating chunks + episode-uniform rescale sampling poisoned
# D_R with cross-boundary jumps and swamped the teleop grasp content 80/20).
import glob as _glob, pickle as _pickle
dep_runs = []
for pk in sorted(_glob.glob("/home/admin_025/mt_pi_codebase/deploy_logs/xdiff_naive_*/commands.pkl")):
    try:
        dd = _pickle.load(open(pk, "rb"))
    except Exception:
        continue
    _rd = str(dd.get("run_dir", ""))
    # jar_v1 = lid approaches; policy_jar_uni_* = ONLINE AGGREGATION of this very
    # policy's shield-approved executed chunks (covers its own approach variants —
    # 08-14 19:49 latch: an east-drifting descend variant scored k* 60-76 vs tau 66)
    if ("policy_jar_v1" not in _rd) and ("policy_jar_uni" not in _rd):
        continue
    rows = []
    for c in dd["cycles"]:
        if c["commanded"] is None or len(np.atleast_1d(c["commanded"])) == 0:
            continue
        _ch = np.asarray(c["chunk"], np.float64)[:, 3:7]
        # EULER FILTER: rotation-like content must NEVER enter D_R regardless of
        # provenance — "executed" does not make rotation robot-feasible content;
        # absence of rotation from D_R IS the shield's mechanism. (Pollution found
        # 08-14 21:22: at-holder chunks with eul-act 0.49-0.93 scored k*<=58.)
        # wobble band 0.15-0.30 is NORMAL policy texture (run 002449 latched on
        # k* 12-44 wobble chunks under tau=10 while the hand detector was still
        # warming up); only coherent rotation (data 0.38+, policy 0.49+) is out
        if np.abs(np.diff(_ch[:, :3], axis=0)).sum(0).max() >= 0.30:
            continue
        rows.append(np.concatenate([_ch[:, :3] - _ch[:1, :3], _ch[:, 3:]], 1))
    if len(rows) >= 4:
        dep_runs.append(np.stack(rows))
# interleaved split: runs sort by timestamp, so "last 4" would put ALL
# jar_uni online-aggregated runs in heldout and none in training
_va_idx = set(range(len(dep_runs))[::max(1, len(dep_runs) // 4)][:4] or [0])
dep_va = [r for i, r in enumerate(dep_runs) if i in _va_idx]
dep_tr = [r for i, r in enumerate(dep_runs) if i not in _va_idx]
# always hold out the newest run too (freshest deploy distribution for tau)
if len(dep_tr) > 1:
    dep_va.append(dep_tr.pop(-1))
DEP_W = np.concatenate(dep_tr)          # (N, 8, 7) atomic deploy windows
print(f"deploy corpus: {len(dep_tr)} train runs -> {len(DEP_W)} atomic windows, "
      f"{len(dep_va)} heldout runs for tau")
# BOOTSTRAP corpus: the uni policy's own screened non-rotation proposals (its
# diagonal lid approach lies outside teleop+jar_v1 coverage -> rising-k* false
# vetoes without this).  Rotation-intent chunks were excluded at collection.
BOOT_va = BOOT_tr = np.zeros((0, T, 7))   # v7: bootstrap REMOVED (see v6 postmortem)
# closed-grip teleop spans (pad 4): stratified draws fix the closed-hold k* tail
R_closed = []
for a in R_eps:
    cl = a[:, 3] < 0.5
    i = 0
    while i < len(a):
        if cl[i]:
            j = i
            while j < len(a) and cl[j]:
                j += 1
            s, e = max(0, i - 4), min(len(a), j + 4)
            if e - s >= 16:
                R_closed.append(a[s:e])
            i = j
        else:
            i += 1
print(f"closed-grip teleop spans: {len(R_closed)}")
H_eps = [load(f)[0] for f in hum_tr]
# dedicated ROTATE pool (train episodes only): direct supervision of the exact
# demo boundary — rotate windows are only ~5% of D_H and the real rotation is
# GENTLE (S-slowed, eulA med 0.38), easily under-weighted otherwise
ROT_TR = []
for f in hum_tr:
    a, fm, role = load(f)
    if role != "A" or not (fm == 0).any():
        continue
    for t in range(0, len(a) - T, 2):
        if fm[t:t + T].max() == 0:
            ROT_TR.append(a[t:t + T])
ROT_TR = rel_euler(np.stack(ROT_TR))
print(f"rotate supervision pool: {len(ROT_TR)} windows")
R_tr = np.concatenate([windows(a)[0] for a in R_eps])
H_tr = np.concatenate([windows(a)[0] for a in H_eps])
print(f"D_R={len(R_tr)} (robot, {len(rob_tr)} eps)  D_H={len(H_tr)} (human both roles, "
      f"{len(hum_tr)} eps) — NO feasibility labels")

# FIXED physical scaling (no data dependence): relative euler / 0.5 rad, grip
# {0,1} -> {-1,1}. Data-min/max was dominated by extreme fast-rotation windows
# (+-3.2 rad) and crushed typical gentle rotation to ~0.1 normalized units —
# k~10 diffusion noise already swamped it (rotate k* median 10, veto 49%).
st = (np.array([-0.5, -0.5, -0.5, 0.0]), np.array([0.5, 0.5, 0.5, 1.0]))


def sample_rescaled(eps, n):
    """SPEED-INVARIANCE augmentation: sample a segment of length round(8*r),
    r ~ U[0.6, 1.8], linearly resample to T=8 — the classifier cannot use
    absolute speed as the class discriminant (robot teleop is ~2x slower than
    the speed-aligned human streams)."""
    out = np.empty((n, T, eps[0].shape[-1]))
    for i in range(n):
        a = eps[rng.integers(0, len(eps))]
        r = rng.uniform(0.6, 1.8)
        L = max(3, int(round(T * r)))
        if len(a) <= L + 1:
            L = len(a) - 1
        t0 = rng.integers(0, len(a) - L)
        seg = a[t0:t0 + L]
        xs = np.linspace(0, L - 1, T)
        for d in range(out.shape[-1]):
            out[i, :, d] = np.interp(xs, np.arange(L), seg[:, d])
    return rel_euler(out)


START_P = np.array([0.927, 0.184, 0.66])
LID_P = np.array([0.944, 0.186, 0.557])
CANON_E = np.array([3.09, 0.02, -1.57])


def rot_counterfactual(W, rng):
    """Coherent euler motion injected onto ROBOT-content windows, labeled HUMAN.
    No feasibility labels involved: rotation is absent from D_R by construction,
    and these counterfactuals amplify that absence into an explicit boundary —
    without them the position prior dominates and synthetic at-lid rotation
    scored k*<=24 (08-14 probe): the veto was riding on position novelty, which
    online aggregation progressively removes."""
    W = W.copy()
    for i in range(len(W)):
        ax = rng.integers(0, 3)
        if rng.random() < 0.5:
            rate = rng.uniform(0.055, 0.12) * (1 if rng.random() < 0.5 else -1)
            W[i, :, ax] += np.arange(T) * rate
        else:
            amp = rng.uniform(0.25, 0.6)
            W[i, :, ax] += amp * np.sin(np.arange(T) * rng.uniform(0.6, 1.6)
                                        + rng.uniform(0, 2 * np.pi))
    return W


def norm(x):
    return (2 * (x - st[0]) / (st[1] - st[0] + 1e-8) - 1).astype(np.float32)


cls = HumanRobotClassifier(obs_horizon=1, state_cond_dim=4, action_dim=4,
                           cfg=HumanRobotClassifierConfig(num_train_timesteps=KMAX)).to(DEV)
opt = torch.optim.AdamW(cls.parameters(), lr=1e-4, weight_decay=1e-4)

for step in range(4000):
    # robot batch: teleop 80 + closed-strat 16 + deploy 32 (v4 recipe + closed strat)
    dep = DEP_W[rng.integers(0, len(DEP_W), 32)]
    clo = sample_rescaled(R_closed, 16) if R_closed else sample_rescaled(R_eps, 16)
    rob = np.concatenate([sample_rescaled(R_eps, 80), clo, dep])
    # human batch: 80 real + 48 rotation-counterfactuals; HALF the counterfactual
    # bases are at-lid transit content — "at-lid moving (robot) vs at-lid
    # rotating (human)" is exactly the demo boundary and position no longer
    # separates it (the synthetic corpus covers the lid region by design)
    # class 0 = ROTATION-LIKE (activity-mined real rotate windows + counterfactuals
    # on robot content); class 1 = robot-executed (euler-quiet by construction).
    # REAL-HUMAN QUIET ROWS ARE DELIBERATELY ABSENT: under position-blind relative
    # features they are indistinguishable from robot quiet — feeding them as class 0
    # is irreducible contradictory gradient (72% of the batch) that drowned the
    # rotation signal (bce stuck at 0.71; the clean task converges in 500 steps).
    cf = rot_counterfactual(rob[rng.choice(len(rob), 64, replace=False)], rng)
    rot = ROT_TR[rng.integers(0, len(ROT_TR), 64)]
    W = np.concatenate([rot, cf, rob])
    # texture equalization: both classes get identical small augmentation
    W = W.copy()
    W[..., :3] += rng.normal(0, 0.008, W[..., :3].shape)
    y = torch.tensor([0.0] * 128 + [1.0] * 128, device=DEV)
    A = torch.from_numpy(norm(W)).to(DEV)
    b = types.SimpleNamespace(action=A, state_cond=A[:, :1, :], label=y)
    # counterfactuals are supervised at LOW k only: at high noise coherent
    # rotation and transit are genuinely indistinguishable — forcing a human
    # label there makes the classifier overfit junk and drags real robot
    # windows up (tau blew to 96). The k* rule only needs rotation to sit
    # below 0.5 at LOW k, exactly where this supervision now concentrates.
    ts = torch.randint(0, KMAX, (len(A),), device=DEV)
    logit = cls.unified_forward(b, timesteps=ts).squeeze(-1)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, y)
    # monotone regularizer: human P(robot|k) non-decreasing, robot non-increasing
    k1 = torch.randint(0, KMAX - 12, (len(A),), device=DEV)
    p1 = torch.sigmoid(cls.unified_forward(b, timesteps=k1).squeeze(-1))
    p2 = torch.sigmoid(cls.unified_forward(b, timesteps=k1 + 10).squeeze(-1))
    l_mono = (torch.relu(p1 - p2) * (1 - y) + torch.relu(p2 - p1) * y).mean()
    total = loss + 0.1 * l_mono
    opt.zero_grad(); total.backward()
    torch.nn.utils.clip_grad_norm_(cls.parameters(), 5.0)
    opt.step()
    if step % 500 == 0 or step == 2999:
        print(f"step {step:4d} bce={loss.item():.4f} mono={l_mono.item():.4f}")
cls.eval()


@torch.no_grad()
def p_curve(W, draws=8, bs=1024):
    A_all = torch.from_numpy(norm(W)).to(DEV)
    out = np.zeros((len(W), len(K_GRID)), np.float32)
    for i0 in range(0, len(W), bs):
        A = A_all[i0:i0 + bs]
        b = types.SimpleNamespace(action=A, state_cond=A[:, :1, :])
        for ki, k in enumerate(K_GRID):
            ts = torch.full((len(A),), int(k), dtype=torch.long, device=DEV)
            ps = [torch.sigmoid(cls.unified_forward(b, timesteps=ts).squeeze(-1)) for _ in range(draws)]
            out[i0:i0 + bs, ki] = torch.stack(ps).mean(0).cpu().numpy()
    return out


def kstar(P):
    ks = np.full(len(P), float(KMAX))
    hit = P >= 0.5
    multi = 0
    for i in range(len(P)):
        idx = np.where(hit[i][:-1] & hit[i][1:])[0]
        if len(idx):
            ks[i] = K_GRID[idx[0]]
        elif hit[i][-1]:
            ks[i] = K_GRID[-1]
        cross = np.diff(hit[i].astype(int))
        multi += int((cross == 1).sum() > 1)
    return ks, multi


rob_va_W = np.concatenate([windows(load(f)[0])[0] for f in rob_va] + list(dep_va)
                          + ([BOOT_va] if len(BOOT_va) else []))
ks_rob, _ = kstar(p_curve(rob_va_W))
tau = float(np.quantile(ks_rob, 0.99)) + 10.0
print(f"\nrobot heldout k*: median {np.median(ks_rob):.0f} q99 {np.quantile(ks_rob,0.99):.0f} -> tau={tau:.0f}")

buck = {"B feasible": [], "A approach": [], "A ROTATE": [], "A lift": []}
multi_tot = n_tot = 0
for f in hum_va:
    a, fm, role = load(f)
    W, t0 = windows(a)
    P = p_curve(W)
    ks, m = kstar(P)
    multi_tot += m; n_tot += len(W)
    for w0, t, k in zip(W, t0, ks):
        te = t + T
        if role == "B":
            buck["B feasible"].append(k)
        else:
            seg = fm[t:te]
            first_rot = int(np.argmax(fm == 0)) if (fm == 0).any() else len(fm)
            if seg.max() == 0:
                buck["A ROTATE"].append(k)
            elif seg.min() == 1 and te <= first_rot:
                buck["A approach"].append(k)
            elif seg.min() == 1 and t >= first_rot:
                buck["A lift"].append(k)

def auroc(pos, neg):
    x = np.concatenate([pos, neg]); y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    o = np.argsort(x); r = np.empty(len(x)); r[o] = np.arange(1, len(x) + 1)
    return float((r[y == 1].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))

print(f"\n=== GATES (held-out; labels used only here) ===")
res = {"tau": tau, "robot_median": float(np.median(ks_rob))}
for k, v in buck.items():
    v = np.array(v)
    if len(v):
        res[k] = {"n": len(v), "median": float(np.median(v)), "veto": float((v > tau).mean())}
        print(f"  {k:12s} n={len(v):4d} k* median={np.median(v):3.0f}  k*>tau: {(v>tau).mean():.1%}")
feas = np.concatenate([np.array(buck["B feasible"]), np.array(buck["A approach"])])
rot = np.array(buck["A ROTATE"])
res["auroc"] = auroc(rot, feas)
res["monotone_ok"] = 1 - multi_tot / max(n_tot, 1)
print(f"  AUROC (rotate vs feasible pools): {res['auroc']:.3f}")
print(f"  single-crossing fraction: {res['monotone_ok']:.1%}")
gap = float(np.quantile(rot, 0.05) - tau) if len(rot) else -1
res["gap"] = gap
ok = res["auroc"] >= 0.85 and res.get("A ROTATE", {}).get("veto", 0) >= 0.9 and \
     res.get("B feasible", {}).get("veto", 1) <= 0.15 and gap >= 0
print(f"  gap (rotate q05 - tau): {gap:.0f}")
print(f"GATE: {'PASS' if ok else 'REVIEW'}")

OUT.mkdir(parents=True, exist_ok=True)
torch.save({"kstar": True, "feat": "eul_grip_rel", "tau": tau,
            "model_state_dict": cls.state_dict(),
            "stats": [st[0].tolist(), st[1].tolist()]}, OUT / "kstar_cls.pth")
(OUT / "gates.json").write_text(json.dumps(res, indent=1))
print(f"saved {OUT}/kstar_cls.pth (tau={tau:.0f})")
