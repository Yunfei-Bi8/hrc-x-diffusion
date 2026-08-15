"""k* (minimum indistinguishability step) a la the ORIGINAL X-Diffusion — NO
feasibility labels anywhere in training.

Classifier: the paper's recipe exactly (Eq. 2): binary human-vs-robot identity
on (k, A^k, s) with k ~ U[0, K), noise on action AND state, balanced classes.
  human class = 10-08 rotating-jar windows (ALL of them — no phase labels)
  robot class = handover robot demos (the rig's feasible motions)
k*(A) = min{ k : c_theta(k, A^k, s) >= 0.5 }   (Eq. 3, P averaged over draws)

Evaluation (labels used ONLY here, never in training): per-window k* profiles
along HELD-OUT jar episodes with the rotate phase shaded; k* distributions
approach-vs-rotate; AUROC of k* as a feasibility score; deploy-threshold sweep
(k* > K_thr -> "infeasible, hold").
Run (mtpi): python scripts/kstar_identity_jar.py
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

JAR = Path("/home/admin_025/X-Diffusion-Data/retargeted/jar_infeasible/human")
ROB = Path("/home/admin_025/X-Diffusion-Data/retargeted/handover_bowl_uni/robot")
OUT = Path("/home/admin_025/X-Diffusion-Data/_private/runs/kstar_jar" + (("_aug" if "--aug-robot" in __import__("sys").argv else "") + ("_t24tex" if "--t24" in __import__("sys").argv else "") + ("_deploy" if "--robot-deploy" in __import__("sys").argv else "")))
DEV = "cuda"
T = 24 if "--t24" in __import__("sys").argv else 8   # 24: rotation-signature context (critic-work lesson)
STRIDE = 2
KMAX = 101
N_HELD = 8                 # last 8 jar episodes held out (same split as the critic work)
ROB_VAL = {"demo00005", "demo00011"}

torch.manual_seed(0)
rng = np.random.default_rng(0)


def load_rows(f):
    with h5py.File(f) as h:
        a = np.concatenate([np.array(h["ee_pos"]), np.array(h["ee_euler"]),
                            np.array(h["gripper_open"])[:, None]], 1).astype(np.float64)
        fm = np.array(h["feas_mask"]) if "feas_mask" in h else np.ones(len(a), np.int32)
    return a, fm


def windows(a, stride=STRIDE):
    return np.stack([a[t:t + T] for t in range(0, len(a) - T, stride)]), \
           np.arange(0, len(a) - T, stride)


jar_files = sorted(JAR.glob("demo*.h5"))
rob_files = sorted(ROB.glob("demo*.h5"))
jar_tr, jar_va = jar_files[:-N_HELD], jar_files[-N_HELD:]
rob_tr = [f for f in rob_files if f.stem not in ROB_VAL]
rob_va = [f for f in rob_files if f.stem in ROB_VAL]

H_tr = np.concatenate([windows(load_rows(f)[0])[0] for f in jar_tr])

if "--robot-deploy" in sys.argv:
    # D_R = the robot's OWN executed trajectories from the jar deployments
    # (chunk rows of cycles that actually ran, 15Hz, 7-dim) — SAME task, SAME
    # workspace, real robot dynamics, feasible BY CONSTRUCTION (everything the
    # shield allowed to execute). Proxy for the upcoming teleop demos.
    import glob as _glob
    import pickle as _pickle
    _rows_per_run = []
    for pk in sorted(_glob.glob("/home/admin_025/mt_pi_codebase/deploy_logs/xdiff_naive_*/commands.pkl")):
        try:
            dd = _pickle.load(open(pk, "rb"))
        except Exception:
            continue
        if "policy_jar_v1" not in str(dd.get("run_dir", "")):
            continue
        rows = [np.asarray(c["chunk"], np.float64) for c in dd["cycles"]
                if c["commanded"] is not None and len(np.atleast_1d(c["commanded"])) > 0]
        if len(rows) >= 4:
            _rows_per_run.append(np.concatenate(rows))
    print(f"deploy-executed robot source: {len(_rows_per_run)} runs")
    _rv = _rows_per_run[-4:]                    # hold out 4 runs as robot-val
    _rt = _rows_per_run[:-4]
    R_tr = np.concatenate([windows(a)[0] for a in _rt])
    R_VA_DEPLOY = np.concatenate([windows(a)[0] for a in _rv])
else:
    R_tr = np.concatenate([windows(load_rows(f)[0])[0] for f in rob_tr])
    R_VA_DEPLOY = None

# --aug-robot: enrich the ROBOT class with LABEL-FREE synthetic robot-plausible
# primitives ANCHORED IN THE JAR WORKSPACE (min-jerk transports + still holds,
# built from generic machinery, no feasibility labels of the jar data): closes
# the task/domain gap so identity-k* can measure style, not workspace.
AUG = "--aug-robot" in sys.argv
if AUG:
    import scripts.feascritic_synth_exp as fx
    anch = np.concatenate([load_rows(f)[0] for f in jar_tr])[::7]      # jar-region poses
    envd = {"p97": 0.004, "p99": 0.0055, "z_lo": float(anch[:, 2].min()),
            "z_hi": float(anch[:, 2].max()), "res_pos": 0.0012}
    fx.T = T                                                            # 8-step primitives
    syn_tr = fx.gen_transport(anch[rng.integers(0, len(anch), 4000)], envd, rng)
    still = np.repeat(anch[rng.integers(0, len(anch), 2000)][:, None, :], T, axis=1).copy()
    still[..., :3] += rng.normal(0, 0.0012, still[..., :3].shape)
    if "--texture" in sys.argv:
        # match the jar exports' noise texture (0.008 euler jitter, pos residual)
        # so the classifier cannot separate the feasible phase by TEXTURE alone
        for Wsyn in (syn_tr, still):
            Wsyn[..., 3:6] += rng.normal(0, 0.008, Wsyn[..., 3:6].shape)
    R_tr = np.concatenate([R_tr, syn_tr, still])
print(f"train windows: human(jar)={len(H_tr)}  robot={len(R_tr)}"
      f"{' (handover + jar-workspace synthetic primitives)' if AUG else ' (handover only)'}"
      f"  — NO feasibility labels used")

st = (np.concatenate([H_tr, R_tr]).reshape(-1, 7).min(0),
      np.concatenate([H_tr, R_tr]).reshape(-1, 7).max(0))


def norm(x):
    return (2 * (x - st[0]) / (st[1] - st[0] + 1e-8) - 1).astype(np.float32)


cls = HumanRobotClassifier(obs_horizon=1, state_cond_dim=7, action_dim=7,
                           cfg=HumanRobotClassifierConfig(num_train_timesteps=KMAX)).to(DEV)
opt = torch.optim.AdamW(cls.parameters(), lr=1e-4, weight_decay=1e-4)

for step in range(3000):
    hw = H_tr[rng.integers(0, len(H_tr), 128)]
    rw = R_tr[rng.integers(0, len(R_tr), 128)]
    W = np.concatenate([hw, rw])
    y = torch.tensor([0.0] * 128 + [1.0] * 128, device=DEV)      # paper: robot=1, human=0
    if "--texture" in sys.argv:
        # texture equalization on BOTH classes: kill processing-texture shortcuts
        W = W.copy()
        W[..., 3:6] += rng.normal(0, 0.008, W[..., 3:6].shape)
        W[..., :3] += rng.normal(0, 0.0012, W[..., :3].shape)
    A = torch.from_numpy(norm(W)).to(DEV)
    S = A[:, :1, :]
    b = types.SimpleNamespace(action=A, state_cond=S, label=y)
    loss = cls.loss(b)                                            # uniform-k noising inside (Eq. 2)
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(cls.parameters(), 5.0)
    opt.step()
    if step % 500 == 0 or step == 2999:
        print(f"step {step:4d} bce={loss.item():.4f}")
cls.eval()

K_GRID = np.arange(0, KMAX, 2)


@torch.no_grad()
def p_robot_curve(W, draws=8, bs=1024):
    """(N,T,7) -> (N, len(K_GRID)) P(robot | k) averaged over noise draws"""
    A_all = torch.from_numpy(norm(W)).to(DEV)
    out = np.zeros((len(W), len(K_GRID)), np.float32)
    for i0 in range(0, len(W), bs):
        A = A_all[i0:i0 + bs]
        S = A[:, :1, :]
        b = types.SimpleNamespace(action=A, state_cond=S)
        for ki, k in enumerate(K_GRID):
            ts = torch.full((len(A),), int(k), dtype=torch.long, device=DEV)
            ps = [torch.sigmoid(cls.unified_forward(b, timesteps=ts).squeeze(-1)) for _ in range(draws)]
            out[i0:i0 + bs, ki] = torch.stack(ps).mean(0).cpu().numpy()
    return out


def kstar_from_curve(P):
    """Eq. 3: first k on the grid with P>=0.5 (2 consecutive grid points to
    de-noise the crossing); never crosses -> KMAX"""
    ks = np.full(len(P), float(KMAX))
    hit = (P >= 0.5)
    for i in range(len(P)):
        idx = np.where(hit[i][:-1] & hit[i][1:])[0]
        if len(idx):
            ks[i] = K_GRID[idx[0]]
        elif hit[i][-1]:
            ks[i] = K_GRID[-1]
    return ks


OUT.mkdir(parents=True, exist_ok=True)

# ---- per-episode k* profiles on HELD-OUT jar episodes ----
fig, axes = plt.subplots(4, 2, figsize=(14, 12), sharex=False)
all_ap, all_rot, ep_stats = [], [], []
for ax, f in zip(axes.ravel(), jar_va):
    a, fm = load_rows(f)
    W, t0 = windows(a)
    ks = kstar_from_curve(p_robot_curve(W))
    wl = np.array([fm[t:t + T].mean() for t in t0])       # 1=feasible-phase window
    ap = ks[wl > 0.99]; rot = ks[wl < 0.01]
    all_ap.append(ap); all_rot.append(rot)
    ep_stats.append((f.stem, float(np.median(ap)) if len(ap) else -1,
                     float(np.median(rot)) if len(rot) else -1))
    ax.plot(t0, ks, lw=1.5, color="k")
    infe = fm < 0.5
    ax.fill_between(np.arange(len(a)), 0, KMAX, where=infe, alpha=0.25, color="red",
                    label="rotate phase (eval-only label)")
    ax.set_title(f"{f.stem}   median k*: feas={np.median(ap) if len(ap) else float('nan'):.0f} "
                 f"rot={np.median(rot) if len(rot) else float('nan'):.0f}", fontsize=9)
    ax.set_ylim(0, KMAX + 2); ax.set_ylabel("k*")
axes.ravel()[0].legend(fontsize=7)
fig.suptitle("per-window k* along HELD-OUT jar episodes (identity classifier, NO feasibility labels in training)")
fig.tight_layout()
fig.savefig(OUT / "kstar_episodes.png", dpi=110)

ap = np.concatenate(all_ap); rot = np.concatenate(all_rot)

# robot sanity reference
if R_VA_DEPLOY is not None:
    R_va = R_VA_DEPLOY
else:
    R_va = np.concatenate([windows(load_rows(f)[0])[0] for f in rob_va])
ks_rob = kstar_from_curve(p_robot_curve(R_va))
tau = float(np.quantile(ks_rob, 0.99)) + 10.0
print(f"[tau preview] robot-val q99 + 10 = {tau:.0f}")


def auroc(pos, neg):
    x = np.concatenate([pos, neg]); y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    o = np.argsort(x); r = np.empty(len(x)); r[o] = np.arange(1, len(x) + 1)
    return float((r[y == 1].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))

fig2, ax = plt.subplots(1, 2, figsize=(13, 4.5))
bins = np.arange(0, KMAX + 4, 4)
ax[0].hist(ks_rob, bins=bins, alpha=0.6, label=f"ROBOT val (n={len(ks_rob)})", density=True)
ax[0].hist(ap, bins=bins, alpha=0.6, label=f"jar FEASIBLE-phase (n={len(ap)})", density=True)
ax[0].hist(rot, bins=bins, alpha=0.6, label=f"jar ROTATE-phase (n={len(rot)})", density=True)
ax[0].set_xlabel("k* (min indistinguishability step)"); ax[0].legend(fontsize=8)
ax[0].set_title("k* distributions (held-out)")
thrs = np.arange(10, 101, 5)
veto = [(rot > t).mean() for t in thrs]
fpr = [(ap > t).mean() for t in thrs]
ax[1].plot(thrs, veto, label="rotate veto rate (k*>thr)")
ax[1].plot(thrs, fpr, label="feasible false-veto rate")
ax[1].set_xlabel("deploy threshold K_thr"); ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
ax[1].set_title("deployment rule sweep: k* > K_thr -> HOLD")
fig2.tight_layout()
fig2.savefig(OUT / "kstar_summary.png", dpi=110)

a_roc = auroc(rot, ap)   # rotate should have HIGHER k*
stats = {
    "kstar_median": {"robot_val": float(np.median(ks_rob)),
                     "jar_feasible_phase": float(np.median(ap)),
                     "jar_rotate_phase": float(np.median(rot))},
    "kstar_p90": {"robot_val": float(np.percentile(ks_rob, 90)),
                  "jar_feasible_phase": float(np.percentile(ap, 90)),
                  "jar_rotate_phase": float(np.percentile(rot, 90))},
    "auroc_kstar_rotate_vs_feasible": a_roc,
    "threshold_sweep": {int(t): {"rotate_veto": float((rot > t).mean()),
                                 "feasible_fpr": float((ap > t).mean())} for t in thrs},
    "per_episode_medians": ep_stats,
}
(OUT / "stats.json").write_text(json.dumps(stats, indent=1))
torch.save({"model_state_dict": cls.state_dict(), "stats": [st[0].tolist(), st[1].tolist()]},
           OUT / "identity_cls.pth")
print("\n=== k* RESULTS (held-out, label-free training) ===")
print(f"median k*: robot-val {np.median(ks_rob):.0f} | jar feasible-phase {np.median(ap):.0f} | jar ROTATE {np.median(rot):.0f}")
print(f"AUROC (k* separates rotate from feasible): {a_roc:.3f}")
for t in (30, 50, 70):
    print(f"  K_thr={t}: rotate veto {(rot > t).mean():.1%} | feasible false-veto {(ap > t).mean():.1%}")
print(f"saved -> {OUT}")
