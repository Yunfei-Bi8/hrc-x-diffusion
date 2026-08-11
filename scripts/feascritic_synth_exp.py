"""Feasibility-critic synthetic-negative experiment (2026-08-07).

Question: can the (fixed) discriminator architecture separate
  (a) KINEMATIC infeasibility  -- envelope-violating corruptions of real windows
  (b) AFFORDANCE infeasibility -- synthetic trajectories that stay INSIDE the
      kinematic envelope but carry gripper-impossible dexterity structure
from robot-feasible motion, with the two heads DISSOCIATING (kin head ignores
afford negatives and vice versa)?

Data: h5 exports of 29-07-two-human-handover + 07-29-robot/human-first-handover
(X-Diffusion-Data/retargeted/handover_bowl_uni: 12 robot + 136 human demos).

Runs (see __main__): main | lofo (train w/o teleport+whipsaw) |
loto (train w/o cap_twist) | blind (state_cond zeroed = the pre-fix behavior).
"""
import argparse, json, sys, types
from pathlib import Path

import h5py
import numpy as np
import torch

sys.path.insert(0, "/home/admin_025/X-Diffusion")
from models.xdiffusion.hr_classifier import FeasibilityCritic, FeasibilityCriticConfig

DATA = Path("/home/admin_025/X-Diffusion-Data/retargeted/handover_bowl_uni")
OUT = Path("/home/admin_025/X-Diffusion-Data/_private/runs/feascritic_synth")
VAL_ROBOT = {"demo00005", "demo00011"}
VAL_HUMAN = {"demo00068", "demo00069", "demo00164", "demo00165"}
T = 24
STRIDE = 4
EUL_RATE_MAX = 0.22       # rad/step @15Hz ~ UR wrist joint velocity limit
GRASP_MIN_SPACING = 6     # steps ~ 0.4s, near 2F-85 stroke bandwidth
KIN_FAMS = ["speed", "teleport", "jitter", "whipsaw", "euler_rate", "flutter_fast", "start_jump"]
AFF_FAMS = ["pen_spin", "cap_twist", "gather"]
KS = [0, 5, 10, 20, 40]


def wrap_diff(a):
    d = np.diff(a, axis=1)
    return (d + np.pi) % (2 * np.pi) - np.pi


def load_pool():
    tr, va = [], []
    for sub, val_ids in (("robot", VAL_ROBOT), ("human", VAL_HUMAN)):
        for f in sorted((DATA / sub).glob("demo*.h5")):
            with h5py.File(f, "r") as h:
                a = np.concatenate([np.array(h["ee_pos"]), np.array(h["ee_euler"]),
                                    np.array(h["gripper_open"])[:, None]], 1).astype(np.float64)
            ws = [a[t:t + T] for t in range(0, len(a) - T, STRIDE)]
            if not ws:
                continue
            (va if f.stem in val_ids else tr).append((sub, np.stack(ws)))
    def cat(lst):
        return {"robot": np.concatenate([w for s, w in lst if s == "robot"]),
                "human": np.concatenate([w for s, w in lst if s == "human"])}
    return cat(tr), cat(va)


def measure_envelope(train):
    allw = np.concatenate([train["robot"], train["human"]])
    dstep = np.linalg.norm(np.diff(allw[..., :3], axis=1), axis=-1)   # (N,T-1) m/step
    e_rate = np.abs(wrap_diff(allw[..., 3:6])).max(-1)                # per-step max-axis rad
    env = {
        "p50": float(np.percentile(dstep, 50)), "p95": float(np.percentile(dstep, 95)),
        "p97": float(np.percentile(dstep, 97)), "p99": float(np.percentile(dstep, 99)),
        "e_p99": float(np.percentile(e_rate, 99)),
        "z_lo": float(np.percentile(allw[..., 2], 1)), "z_hi": float(np.percentile(allw[..., 2], 99)),
        "eul_span": [float(x) for x in (allw[..., 3:6].max((0, 1)) - allw[..., 3:6].min((0, 1)))],
        "res_pos": float(np.median(np.std(allw[..., :3] - 0.5 * (np.roll(allw[..., :3], 1, 1) + np.roll(allw[..., :3], -1, 1)), axis=1))),
    }
    return env


def envelope_check(W, env):
    """returns fraction of windows fully inside the kinematic envelope"""
    dstep = np.linalg.norm(np.diff(W[..., :3], axis=1), axis=-1)
    ok_pos = (dstep <= 1.15 * env["p99"]).all(1)
    ok_eul = (np.abs(wrap_diff(W[..., 3:6])) <= EUL_RATE_MAX).all(axis=(1, 2))
    g = (W[..., 6] > 0.5).astype(int)
    tr = np.abs(np.diff(g, axis=1))
    ok_g = np.array([np.all(np.diff(np.flatnonzero(r)) >= GRASP_MIN_SPACING) if r.sum() > 1 else True for r in tr])
    return ok_pos & ok_eul & ok_g


# ---------------- kinematic corruption families (row 0 untouched except start_jump) ----
def corrupt(W, fam, env, rng):
    W = W.copy()
    B = len(W)
    L = W.shape[1]
    p99 = env["p99"]
    if fam == "speed":
        cur = np.linalg.norm(np.diff(W[..., :3], axis=1), axis=-1).max(1, keepdims=True)  # (B,1)
        target = rng.uniform(1.5, 3.0, (B, 1)) * p99
        m = np.clip(target / np.maximum(cur, 1e-4), 1.2, 8.0)[..., None]
        W[..., :3] = W[:, :1, :3] + m * (W[..., :3] - W[:, :1, :3])
    elif fam == "teleport":
        idx = rng.integers(max(2, L // 4), L - 2, B)
        off = rng.normal(size=(B, 3)); off /= np.linalg.norm(off, axis=1, keepdims=True)
        off *= (rng.uniform(2.5, 6.0, (B, 1)) * p99)
        for i in range(B):
            W[i, idx[i]:, :3] += off[i]
    elif fam == "jitter":
        s = rng.uniform(0.8, 2.0, (B, 1, 1)) * p99
        n = rng.normal(0, 1, (B, L, 3)) * s
        n[:, 0] = 0
        W[..., :3] += n
    elif fam == "whipsaw":
        a = rng.uniform(1.5, 3.5, (B, 1)) * p99
        ax = rng.integers(0, 2, B)
        alt = ((-1) ** np.arange(L))[None, :] * a
        alt[:, 0] = 0
        for i in range(B):
            W[i, :, ax[i]] += alt[i]
    elif fam == "euler_rate":
        ax = rng.integers(0, 3, B)
        r = rng.uniform(0.3, 0.6, B) * rng.choice([-1, 1], B)
        ramp = np.arange(L)[None, :] * r[:, None]
        for i in range(B):
            W[i, :, 3 + ax[i]] += ramp[i]
    elif fam == "flutter_fast":
        per = rng.integers(1, 3, B)
        for i in range(B):
            g = ((np.arange(L) // per[i]) % 2).astype(float)
            if g[0] != W[i, 0, 6]:
                g = 1 - g
            W[i, 1:, 6] = g[1:]
    elif fam == "start_jump":
        off = rng.normal(size=(B, 3)); off /= np.linalg.norm(off, axis=1, keepdims=True)
        off *= (rng.uniform(3.0, 7.0, (B, 1)) * p99)
        W[..., :3] += off[:, None, :]      # whole window shifted; state_cond stays at true row0
    return W


# ---------------- affordance-infeasible generators (inside the envelope) --------------
def _ou(B, Trng, sigma, theta, rng):
    x = np.zeros((B, Trng, 3))
    for t in range(1, Trng):
        x[:, t] = x[:, t - 1] * (1 - theta) + rng.normal(0, sigma, (B, 3))
    return x


def _res_noise(B, env, rng, scale=1.0):
    n = rng.normal(0, env["res_pos"] * scale, (B, T, 3)); n[:, 0] = 0
    return n


def gen_afford(anchors, fam, env, rng):
    """anchors: (B,7) real rows (grasp==0 for pen_spin/cap_twist). Returns (B,T,7)."""
    B = len(anchors)
    W = np.repeat(anchors[:, None, :], T, axis=1).copy()
    W[..., :3] += _ou(B, T, 0.0015, 0.12, rng) + _res_noise(B, env, rng)
    if fam == "pen_spin":
        ax1 = rng.integers(0, 3, B)
        ax2 = (ax1 + rng.integers(1, 3, B)) % 3         # two DISTINCT euler axes churn
        for ax in (ax1, ax2):
            A = rng.uniform(0.15, 0.5, B)
            Pmin = np.maximum(2 * np.pi * A / 0.13, 10)
            P = np.maximum(rng.uniform(10, 26, B), Pmin)
            phi = rng.uniform(0, 2 * np.pi, B)
            t = np.arange(T)[None, :]
            wave = A[:, None] * (np.sin(2 * np.pi * t / P[:, None] + phi[:, None]) - np.sin(phi)[:, None])
            for i in range(B):
                W[i, :, 3 + ax[i]] += wave[i]
        W[..., 6] = 0.0                                 # holding throughout
    elif fam == "cap_twist":
        A = rng.uniform(0.3, 0.6, B) * rng.choice([-1, 1], B)
        r1 = rng.integers(8, 13, B)
        o = rng.integers(6, 10, B)
        for i in range(B):
            up = np.linspace(0, A[i], r1[i])
            down = np.linspace(A[i], 0, T - r1[i])
            W[i, :, 5] += np.concatenate([up, down])    # slow twist out & back (yaw)
            W[i, r1[i]:r1[i] + o[i], 6] = 1.0           # release during untwist, then regrasp
    elif fam == "gather":
        f1, f2 = rng.integers(1, 3, B), rng.integers(1, 3, B)
        rcap = 0.55 * env["p99"] * T / (2 * np.pi * np.maximum(f1, f2))
        r = np.minimum(rng.uniform(0.004, 0.010, B), rcap)
        ph1, ph2 = rng.uniform(0, 2 * np.pi, (2, B))
        t = np.arange(T)[None, :]
        x = r[:, None] * (np.sin(2 * np.pi * f1[:, None] * t / T + ph1[:, None]) - np.sin(ph1)[:, None])
        y = r[:, None] * (np.sin(2 * np.pi * f2[:, None] * t / T + ph2[:, None]) - np.sin(ph2)[:, None])
        W[..., 0] += x; W[..., 1] += y
        W[..., 2] += 0.003 * np.sin(2 * np.pi * t / T)
        W[..., 4] += 0.06 * np.sin(2 * np.pi * t / T + rng.uniform(0, 6.28, (B, 1)))
        t1 = rng.integers(7, 12, B)
        for i in range(B):
            W[i, t1[i]:, 6] = 1.0 - W[i, 0, 6]
            if rng.random() < 0.5 and t1[i] + 8 < T:
                W[i, t1[i] + 8:, 6] = W[i, 0, 6]
    return W


def gen_transport(anchors, env, rng):
    """synthetic FEASIBLE min-jerk moves — positive control sharing the synth machinery"""
    B = len(anchors)
    W = np.repeat(anchors[:, None, :], T, axis=1).copy()
    cap = 0.75 * env["p97"] * T / 1.875
    d = rng.normal(size=(B, 3)); d[:, 2] *= 0.5
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    d *= rng.uniform(0.04, max(0.05, cap), (B, 1))
    tz = np.clip(anchors[:, 2] + d[:, 2], env["z_lo"], env["z_hi"])
    d[:, 2] = tz - anchors[:, 2]
    s = np.arange(T) / (T - 1)
    mj = 10 * s**3 - 15 * s**4 + 6 * s**5
    W[..., :3] += d[:, None, :] * mj[None, :, None]
    W[..., :3] += _ou(B, T, 0.0015, 0.12, rng) + _res_noise(B, env, rng)
    A = rng.uniform(0, 0.1, (B, 1)) * rng.choice([-1, 1], (B, 1))
    W[..., 4] += A * np.sin(np.pi * s)[None, :]
    flip = rng.random(B) < 0.3
    tf = rng.integers(8, 16, B)
    for i in np.flatnonzero(flip):
        W[i, tf[i]:, 6] = 1.0 - W[i, 0, 6]
    return W


def gen_checked(genfn, pool, n, env, rng, **kw):
    """generate 2x, keep only envelope-passing windows, take n (afford negatives
    and synthetic positives MUST be kinematically clean — they carry kin=1)"""
    W = genfn(pool[rng.integers(0, len(pool), 2 * n)], env=env, rng=rng, **kw)
    ok = envelope_check(W, env)
    W = W[ok]
    if len(W) < n:
        W2 = genfn(pool[rng.integers(0, len(pool), 4 * n)], env=env, rng=rng, **kw)
        W = np.concatenate([W, W2[envelope_check(W2, env)]])
    return W[:n]


# ---------------------------------- experiment ----------------------------------------
def norm(x, st):
    return (2 * (x - st[0]) / (st[1] - st[0] + 1e-8) - 1).astype(np.float32)


def auroc(pos, neg):
    x = np.concatenate([pos, neg]); y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    o = np.argsort(x); r = np.empty(len(x)); r[o] = np.arange(1, len(x) + 1)
    return float((r[y == 1].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--skip-kin", default="")
    ap.add_argument("--skip-afford", default="")
    ap.add_argument("--state-blind", action="store_true")
    ap.add_argument("--state-channel", action="store_true",
                    help="also concat state (broadcast over T) as extra input channels: action_dim 7->14")
    ap.add_argument("--corrupt-span", choices=["full", "mixed"], default="full",
                    help="kin corruption span at TRAINING: full window (original) or mixed "
                         "(50%% full / 25%% suffix-8 [the deploy chunk slot] / 25%% random suffix)")
    ap.add_argument("--train-k", choices=["uniform", "zero", "low"], default="uniform",
                    help="noise-step sampling during TRAINING: uniform=U[0,101) (paper), "
                         "zero=k=0 always (clean-chunk classifier for shield-only use), "
                         "low=50%% k=0 + 50%% U[0,21)")
    ap.add_argument("--eval-draws", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    dev = "cuda"
    skip_kin = set(filter(None, args.skip_kin.split(",")))
    skip_aff = set(filter(None, args.skip_afford.split(",")))
    train_kin = [f for f in KIN_FAMS if f not in skip_kin]
    train_aff = [f for f in AFF_FAMS if f not in skip_aff]

    train, val = load_pool()
    env = measure_envelope(train)
    print(f"pool: robot {len(train['robot'])}/{len(val['robot'])} human {len(train['human'])}/{len(val['human'])} windows (train/val)")
    print(f"envelope: |dp|/step p50={env['p50']*1000:.1f} p95={env['p95']*1000:.1f} p99={env['p99']*1000:.1f} mm; "
          f"|de|/step p99={env['e_p99']:.3f} rad; eul spans={[round(x,2) for x in env['eul_span']]} rad; res={env['res_pos']*1000:.2f} mm")

    allw = np.concatenate([train["robot"], train["human"]])
    st = (allw.reshape(-1, 7).min(0), allw.reshape(-1, 7).max(0))

    anch_tr = allw[:, 0, :]
    anch_tr_closed = anch_tr[anch_tr[:, 6] < 0.5]
    valw = {"robot": val["robot"], "human": val["human"]}
    anch_va = np.concatenate([val["robot"], val["human"]])[:, 0, :]
    anch_va_closed = anch_va[anch_va[:, 6] < 0.5]
    print(f"anchors: train {len(anch_tr)} ({len(anch_tr_closed)} closed) val {len(anch_va)} ({len(anch_va_closed)} closed)")

    # --- construction validity table (on val-anchored samples) ---
    print("\n=== construction validity (n=400 each) ===")
    validity = {}
    vw = np.concatenate([valw["robot"], valw["human"]])
    vw_moving = vw[np.linalg.norm(np.diff(vw[..., :3], axis=1), axis=-1).max(1) >= 0.5 * env["p95"]]
    for fam in KIN_FAMS:
        basev = vw_moving if fam == "speed" else vw
        W = corrupt(basev[rng.integers(0, len(basev), 400)], fam, env, rng)
        frac = envelope_check(W, env).mean()
        pk = np.linalg.norm(np.diff(W[..., :3], axis=1), axis=-1).max(1).mean() * 1000
        validity[fam] = {"env_pass": float(frac), "peak_mm": float(pk)}
        note = "(state-relative violation; window itself in-envelope by design)" if fam == "start_jump" else "(want ~0%)"
        print(f"  kin {fam:13s}: envelope-pass {frac:5.1%} {note}  peak|dp| {pk:6.1f} mm")
    for fam in AFF_FAMS:
        pool = anch_va_closed if fam in ("pen_spin", "cap_twist") else anch_va
        W = gen_afford(pool[rng.integers(0, len(pool), 400)], fam, env, rng)
        frac = envelope_check(W, env).mean()
        pk = np.linalg.norm(np.diff(W[..., :3], axis=1), axis=-1).max(1).mean() * 1000
        validity[fam] = {"env_pass": float(frac), "peak_mm": float(pk)}
        print(f"  aff {fam:13s}: envelope-pass {frac:5.1%} (want ~100%) peak|dp| {pk:6.1f} mm")
    Wt = gen_transport(anch_va[rng.integers(0, len(anch_va), 400)], env, rng)
    print(f"  pos transport   : envelope-pass {envelope_check(Wt, env).mean():5.1%} (want ~100%)")

    # moving-window subsets (speed corruption is only meaningful on windows that move)
    def moving_mask(W):
        return np.linalg.norm(np.diff(W[..., :3], axis=1), axis=-1).max(1) >= 0.5 * env["p95"]
    base_tr = np.concatenate([train["robot"], train["human"]])
    base_tr_mov = base_tr[moving_mask(base_tr)]
    print(f"moving-window base: {len(base_tr_mov)}/{len(base_tr)} train")

    # --- model ---
    cfg = FeasibilityCriticConfig(num_train_timesteps=101, finger_dim=0,
                                  head_names=["identity", "kin_feas", "afford_feas"])
    crit = FeasibilityCritic(obs_horizon=1, state_cond_dim=7,
                             action_dim=14 if args.state_channel else 7, cfg=cfg).to(dev)
    opt = torch.optim.AdamW(crit.parameters(), lr=1e-4, weight_decay=1e-4)

    # state_cond is always the ORIGINAL row0 (the arm's true current pose) —
    # in particular for start_jump, where the corrupted window is shifted away
    # from it and the mismatch IS the violation.
    def make_batch2(B=256):
        n_rob, n_hum, n_tra, n_kin, n_aff = 32, 72, 38, 64, 50
        Ws, Ss, labs = [], [], []
        def add(W, S, lab, n):
            Ws.append(W); Ss.append(S); labs.extend([lab] * n)
        W = train["robot"][rng.integers(0, len(train["robot"]), n_rob)]
        add(W, W[:, :1].copy(), (1, 1, 1, 1), n_rob)
        W = train["human"][rng.integers(0, len(train["human"]), n_hum)]
        add(W, W[:, :1].copy(), (0, 1, 1, 1), n_hum)
        W = gen_checked(gen_transport, anch_tr, n_tra, env, rng)
        add(W, W[:, :1].copy(), (0, 0, 1, 1), len(W))
        per = np.bincount(rng.integers(0, len(train_kin), n_kin), minlength=len(train_kin))
        for fam, n in zip(train_kin, per):
            if n == 0: continue
            base = base_tr_mov if fam == "speed" else base_tr
            W0 = base[rng.integers(0, len(base), n)]
            Wc = W0.copy()
            if args.corrupt_span == "mixed":
                u = rng.random()
                s = 0 if u < 0.5 else (16 if u < 0.75 else int(rng.integers(4, 15)))
            else:
                s = 0
            Wc[:, s:] = corrupt(W0[:, s:], fam, env, rng)
            add(Wc, W0[:, :1].copy(), (0, 0, 0, 1), n)
        per = np.bincount(rng.integers(0, len(train_aff), n_aff), minlength=len(train_aff))
        for fam, n in zip(train_aff, per):
            if n == 0: continue
            pool = anch_tr_closed if fam in ("pen_spin", "cap_twist") else anch_tr
            W = gen_checked(gen_afford, pool, n, env, rng, fam=fam)
            add(W, W[:, :1].copy(), (0, 0, 1, 0), len(W))
        W = np.concatenate(Ws); S = np.concatenate(Ss); L = np.array(labs, np.float32)
        return W, S, L

    def to_batch(W, S, L=None):
        A = torch.from_numpy(norm(W, st)).to(dev)
        Sc = torch.from_numpy(norm(S, st)).to(dev)
        if args.state_blind:
            Sc = Sc * 0
        if args.state_channel:
            A = torch.cat([A, Sc.expand(-1, A.shape[1], -1)], dim=-1)
        b = types.SimpleNamespace(action=A, state_cond=Sc)
        if L is not None:
            b.labels = {"identity": torch.from_numpy(L[:, 0]).to(dev),
                        "kin_feas": torch.from_numpy(L[:, 2]).to(dev),
                        "afford_feas": torch.from_numpy(L[:, 3]).to(dev)}
            b.label_mask = {"identity": torch.from_numpy(L[:, 1] > 0.5).to(dev)}
        return b

    def sample_train_k(n):
        if args.train_k == "uniform":
            return None
        if args.train_k == "zero":
            return torch.zeros(n, dtype=torch.long, device=dev)
        ks = torch.randint(0, 21, (n,), device=dev)
        ks[torch.rand(n, device=dev) < 0.5] = 0
        return ks

    crit.train()
    for step in range(args.steps):
        W, S, L = make_batch2()
        b = to_batch(W, S, L)
        losses = crit.loss_multitask(b, timesteps=sample_train_k(len(b.action)))
        opt.zero_grad(); losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(crit.parameters(), 5.0)
        opt.step()
        if step % 500 == 0 or step == args.steps - 1:
            print(f"step {step:5d}  " + "  ".join(f"{k}={v.item():.4f}" for k, v in losses.items()))

    # ---------------- gate battery ----------------
    crit.eval()

    @torch.no_grad()
    def score(W, S, k):
        b = to_batch(W, S)
        ts = torch.full((len(W),), k, dtype=torch.long, device=dev)
        ps = []
        for _ in range(args.eval_draws):
            ps.append(torch.sigmoid(crit(b, timesteps=ts)))
        return torch.stack(ps).mean(0).cpu().numpy()   # (N,3)

    evalsets = {}
    vr, vh = valw["robot"], valw["human"]
    evalsets["clean_robot"] = (vr, vr[:, :1].copy())
    evalsets["clean_human"] = (vh, vh[:, :1].copy())
    Wt = gen_checked(gen_transport, anch_va, 500, env, rng)
    evalsets["synth_transport"] = (Wt, Wt[:, :1].copy())
    vw_all = np.concatenate([vr, vh])
    vw_mov = vw_all[moving_mask(vw_all)]
    for fam in KIN_FAMS:
        base = vw_mov if fam == "speed" else vw_all
        W0 = base[rng.integers(0, len(base), 600)]
        evalsets[f"kin:{fam}"] = (corrupt(W0, fam, env, rng), W0[:, :1].copy())
    for fam in AFF_FAMS:
        pool = anch_va_closed if fam in ("pen_spin", "cap_twist") else anch_va
        W = gen_checked(gen_afford, pool, 600, env, rng, fam=fam)
        evalsets[f"aff:{fam}"] = (W, W[:, :1].copy())
    Wst = np.repeat(vw_all[rng.integers(0, len(vw_all), 400)][:, :1, :], T, axis=1)
    evalsets["still"] = (Wst, Wst[:, :1].copy())

    P = {k: {name: score(W, S, k) for name, (W, S) in evalsets.items()} for k in KS}
    feas_ref = {k: np.concatenate([P[k]["clean_robot"], P[k]["clean_human"], P[k]["synth_transport"]]) for k in KS}

    res = {"env": env, "validity": validity, "auroc": {}, "meanP": {}}
    print(f"\n=== AUROC (feasible-pool vs family) — run {args.run_name} ===")
    print(f"{'family':22s}" + "".join(f"  kin@k={k:<3d}" for k in KS) + " |" + "".join(f"  aff@k={k:<3d}" for k in KS))
    for name in [f"kin:{f}" for f in KIN_FAMS] + [f"aff:{f}" for f in AFF_FAMS]:
        row = f"{name:22s}"
        res["auroc"][name] = {}
        for hi, hname in ((1, "kin"), (2, "aff")):
            for k in KS:
                a = auroc(feas_ref[k][:, hi], P[k][name][:, hi])
                res["auroc"][name][f"{hname}@{k}"] = a
            row += "".join(f"  {res['auroc'][name][f'{hname}@{k}']:.3f}   " for k in KS)
            if hi == 1:
                row += " |"
        print(row)
    print("\n=== mean P at k=0 (head: identity/kin/afford) ===")
    for name, (W, S) in evalsets.items():
        m = P[0][name].mean(0)
        res["meanP"][name] = [float(x) for x in m]
        print(f"  {name:22s} id={m[0]:.3f} kin={m[1]:.3f} aff={m[2]:.3f}")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{args.run_name}.json").write_text(json.dumps(
        {"args": vars(args), **res}, indent=1))
    torch.save({"model_state_dict": crit.state_dict(), "stats": [st[0].tolist(), st[1].tolist()]},
               OUT / f"{args.run_name}.pth")
    print(f"\nsaved {OUT}/{args.run_name}.json")


if __name__ == "__main__":
    main()
