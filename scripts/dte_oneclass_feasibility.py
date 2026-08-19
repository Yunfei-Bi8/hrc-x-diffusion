"""ONE-CLASS noise-conditional feasibility score (Diffusion Time Estimation).

Refs: Livernoche et al., On Diffusion Modeling for Anomaly Detection, ICLR 2024 (DTE);
Graham et al., CVPR-W 2023 (multi-level reconstruction OOD); Mahmood et al., ICLR 2021
(multiscale score matching); Vincent 2011 (denoising<->score matching).

Trained ONLY on feasible (robot-executed) chunks — no negatives, no counterfactuals.
Invariances injected by POSITIVE-ONLY augmentation (speed rescale + texture jitter):
task-prior-free, hence transferable across tasks.

Evaluated under the IDENTICAL conformal + e-detector stack as knn_conformal.py,
against kstar (two-class) and kNN rows.
"""
import glob, pickle, sys, types
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from scipy.spatial import cKDTree
try:
    import line_profiler
except ImportError:
    _lp = types.ModuleType("line_profiler"); _lp.profile = lambda f: f; sys.modules["line_profiler"] = _lp
sys.path.insert(0, "/home/admin_025/X-Diffusion")
from models.xdiffusion.hr_classifier import HumanRobotClassifier, HumanRobotClassifierConfig
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
import h5py

T = 8; DEV = "cuda"; NB = 101
TASK = Path("/home/admin_025/X-Diffusion-Data/retargeted/jar_uni")
rng = np.random.default_rng(0); torch.manual_seed(0)
LO = np.array([-0.5, -0.5, -0.5, 0.0]); HI = np.array([0.5, 0.5, 0.5, 1.0])

def feat(W):
    W = np.asarray(W, float).copy()
    W[..., :3] -= W[:, :1, :3]
    return (2 * (W - LO) / (HI - LO) - 1).reshape(len(W), -1)

def load4(f):
    with h5py.File(f) as h:
        a = np.concatenate([np.array(h["ee_euler"]), np.array(h["gripper_open"])[:, None]], 1)
        fm = np.array(h["feas_mask"]) if "feas_mask" in h else np.ones(len(a), np.int32)
        nb = int(h.attrs.get("uni_bridge", 0))
    return a[nb:].astype(float), fm[nb:]

def nonoverlap(a):
    n = len(a) // T
    return np.stack([a[i*T:(i+1)*T] for i in range(n)]) if n else np.zeros((0, T, 4))

def sliding(a, stride=2):
    return np.stack([a[t:t+T] for t in range(0, len(a)-T, stride)]) if len(a) > T else np.zeros((0, T, 4))

# ---------- data & splits (identical to knn_conformal.py) ----------
teleop = []
for f in sorted(TASK.glob("robot/demo*.h5")):
    if int(f.stem[4:]) >= 40: continue
    a, _ = load4(f); teleop.append(a)
runs = []
for pk in sorted(glob.glob("/home/admin_025/mt_pi_codebase/deploy_logs/xdiff_naive_*/commands.pkl")):
    try: d = pickle.load(open(pk, "rb"))
    except Exception: continue
    if "jar" not in str(d.get("run_dir", "")): continue
    rows = [np.asarray(c["chunk"], float)[:, 3:7] for c in d["cycles"]
            if c["commanded"] is not None and len(np.atleast_1d(c["commanded"]))]
    rows = [r for r in rows if np.abs(np.diff(r[:, :3], axis=0)).sum(0).max() < 0.30]
    if len(rows) >= 10: runs.append(np.stack(rows))
tel_ref, tel_evalF, tel_cal = teleop[:-4], teleop[-4:-2], teleop[-2:]
run_ref  = [r for i, r in enumerate(runs) if i % 4 in (0, 2)]
run_cal  = [r for i, r in enumerate(runs) if i % 4 == 1]
run_evalF= [r for i, r in enumerate(runs) if i % 4 == 3]
CAL_W = np.concatenate([nonoverlap(a) for a in tel_cal] + list(run_cal))
evalF_streams = [nonoverlap(a) for a in tel_evalF] + list(run_evalF)
appW, rotW, fastA = [], [], []
for f in sorted(TASK.glob("human/demo*.h5")):
    eid = int(f.stem[4:])
    if not (100 <= eid < 200): continue
    a, fm = load4(f)
    W = nonoverlap(a); n = len(W)
    if n < 4 or not (fm == 0).any(): continue
    onset_f = int(np.argmax(fm == 0)); onset_c = onset_f // T
    for i in range(n):
        seg = fm[i*T:(i+1)*T]
        if seg.max() == 0: rotW.append(W[i])
        elif seg.min() == 1 and (i+1)*T <= onset_f: appW.append(W[i])
    if onset_c >= 2: fastA.append((W, onset_c))
appW = np.stack(appW); rotW = np.stack(rotW)
slowA = []
for base in [r for r in run_evalF if len(r) >= 12][:40]:
    for rate in (rng.uniform(0.014, 0.02), rng.uniform(0.02, 0.028)):
        Wm = base.copy(); onset = len(Wm)//2; phase = 0.0
        for c in range(onset, len(Wm)):
            ramp = phase + np.arange(T)*rate
            Wm[c, :, 2] += ramp; phase = ramp[-1] + rate
        slowA.append((Wm, onset))
print(f"ref: {len(tel_ref)}tel+{len(run_ref)}runs  cal chunks {len(CAL_W)}  evalF {len(evalF_streams)} "
      f"app {len(appW)} rot {len(rotW)} fast {len(fastA)} slow {len(slowA)}", flush=True)

# ---------- DTE one-class model ----------
sched = DDPMScheduler(num_train_timesteps=NB, beta_schedule="squaredcos_cap_v2")
AC = sched.alphas_cumprod.numpy()

class DTENet(nn.Module):
    """MLP variant (v1)."""
    def __init__(self, d=32, h=256, nb=NB):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(),
                                 nn.Linear(h, h), nn.SiLU(), nn.Linear(h, nb))
    def forward(self, x): return self.net(x)

class DTEConv(nn.Module):
    """Temporal-conv variant: sees the chunk as a (4ch x 8) sequence — coherent
    ramps vs incoherent wobble are SHAPE-over-time patterns a conv captures."""
    def __init__(self, c=4, h=128, nb=NB):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(c, h, 3, padding=1), nn.SiLU(), nn.GroupNorm(8, h),
            nn.Conv1d(h, h, 3, padding=1), nn.SiLU(), nn.GroupNorm(8, h),
            nn.Conv1d(h, h, 3, padding=1), nn.SiLU())
        self.head = nn.Sequential(nn.Linear(2*h, 256), nn.SiLU(), nn.Linear(256, nb))
    def forward(self, x):                    # x: (B,32) flattened (8,4)
        z = x.view(-1, T, 4).transpose(1, 2)  # (B,4,8)
        z = self.conv(z)
        z = torch.cat([z.mean(-1), z.max(-1).values], -1)
        return self.head(z)

def sample_rescaled(eps_list, n):
    out = np.empty((n, T, 4))
    for i in range(n):
        a = eps_list[rng.integers(0, len(eps_list))]
        r = rng.uniform(0.6, 1.8); L = max(3, int(round(T*r)))
        if len(a) <= L+1: L = len(a)-1
        t0 = rng.integers(0, len(a)-L)
        seg = a[t0:t0+L]; xs = np.linspace(0, L-1, T)
        for d in range(4): out[i, :, d] = np.interp(xs, np.arange(L), seg[:, d])
    return out

RUNREF = np.concatenate(run_ref)
def train_dte(augment, steps=4000, arch="mlp"):
    torch.manual_seed(1)
    net = (DTEConv() if arch == "conv" else DTENet()).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-4)
    ac = torch.as_tensor(AC, dtype=torch.float32, device=DEV)
    for s in range(steps):
        if augment:
            A_ = sample_rescaled(tel_ref, 128)                      # speed invariance
            B_ = RUNREF[rng.integers(0, len(RUNREF), 128)].copy()
            W = np.concatenate([A_, B_])
            W[..., :3] += rng.normal(0, rng.uniform(0, 0.02), W[..., :3].shape)  # texture inv.
        else:
            A_ = np.concatenate([sliding(a, 4) for a in tel_ref])
            W = np.concatenate([A_[rng.integers(0, len(A_), 128)],
                                RUNREF[rng.integers(0, len(RUNREF), 128)]])
        x0 = torch.as_tensor(feat(W), dtype=torch.float32, device=DEV)
        t = torch.randint(0, NB, (len(x0),), device=DEV)
        eps = torch.randn_like(x0)
        xt = ac[t].sqrt()[:, None]*x0 + (1-ac[t]).sqrt()[:, None]*eps
        loss = nn.functional.cross_entropy(net(xt), t)
        opt.zero_grad(); loss.backward(); opt.step()
        if s % 1000 == 0: print(f"  dte(aug={augment}) step {s} ce={loss.item():.3f}", flush=True)
    net.eval(); return net

@torch.no_grad()
def dte_score(net, W4, bs=8192):
    if len(W4) == 0: return np.zeros(0)
    tb = torch.arange(NB, dtype=torch.float32, device=DEV)
    out = []
    for i0 in range(0, len(W4), bs):
        x = torch.as_tensor(feat(W4[i0:i0+bs]), dtype=torch.float32, device=DEV)
        post = torch.softmax(net(x), -1)
        out.append((post @ tb).cpu().numpy())          # posterior-mean diffusion time
    return np.concatenate(out)

print("training DTE mlp(aug)...", flush=True); net_aug = train_dte(True)
print("training DTE mlp(no-aug)...", flush=True); net_na = train_dte(False)
print("training DTE conv(aug)...", flush=True); net_cv = train_dte(True, arch="conv")
print("training DTE conv(no-aug)...", flush=True); net_cvna = train_dte(False, arch="conv")

# ---------- baseline scores ----------
REF = np.concatenate([feat(sliding(a)) for a in tel_ref] + [feat(r) for r in run_ref])
tree = cKDTree(REF)
def knn_score(W4):
    return tree.query(feat(W4), k=1)[0] if len(W4) else np.zeros(0)
ck = torch.load("/home/admin_025/X-Diffusion-Data/_private/runs/kstar_uni/kstar_cls.pth", map_location=DEV, weights_only=False)
cst = np.array(ck["stats"])
cls = HumanRobotClassifier(obs_horizon=1, state_cond_dim=4, action_dim=4,
                           cfg=HumanRobotClassifierConfig(num_train_timesteps=101)).to(DEV)
cls.load_state_dict(ck["model_state_dict"]); cls.eval()
KG = np.arange(0, 101, 4)
@torch.no_grad()
def kstar_score(W4, draws=6, bs=2048):
    if len(W4) == 0: return np.zeros(0)
    Wr = np.asarray(W4, float).copy(); Wr[..., :3] -= Wr[:, :1, :3]
    A_all = torch.as_tensor(((2*(Wr-cst[0])/(cst[1]-cst[0]+1e-8)-1)).astype(np.float32), device=DEV)
    P = np.zeros((len(W4), len(KG)), np.float32)
    for i0 in range(0, len(W4), bs):
        A = A_all[i0:i0+bs]; b = types.SimpleNamespace(action=A, state_cond=A[:, :1, :])
        for ki, kk in enumerate(KG):
            ts = torch.full((len(A),), int(kk), dtype=torch.long, device=DEV)
            ps = [torch.sigmoid(cls.unified_forward(b, timesteps=ts).squeeze(-1)) for _ in range(draws)]
            P[i0:i0+bs, ki] = torch.stack(ps).mean(0).cpu().numpy()
    ks = np.full(len(W4), 101.0); hit = P >= 0.5
    for i in range(len(W4)):
        idx = np.where(hit[i][:-1] & hit[i][1:])[0]
        if len(idx): ks[i] = KG[idx[0]]
        elif hit[i][-1]: ks[i] = KG[-1]
    return ks

FAM = {"kstar": kstar_score, "knn1": knn_score,
       "dte-mlp": lambda W: dte_score(net_aug, W), "dte-mlp-na": lambda W: dte_score(net_na, W),
       "dte-conv": lambda W: dte_score(net_cv, W), "dte-conv-na": lambda W: dte_score(net_cvna, W)}
cal = {f: np.sort(fn(CAL_W)) for f, fn in FAM.items()}
def pval(s, f):
    c = cal[f]; return (1 + len(c) - np.searchsorted(c, s, side="left")) / (len(c) + 1)

EVF = np.concatenate(evalF_streams)
def auroc(pos, neg):
    x = np.concatenate([pos, neg]); y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    o = np.argsort(x); r = np.empty(len(x)); r[o] = np.arange(1, len(x)+1)
    return float((r[y == 1].sum() - len(pos)*(len(pos)+1)/2) / (len(pos)*len(neg)))

print("\n=== PER-CHUNK (identical conformal layer, alpha=0.01) ===", flush=True)
for f, fn in FAM.items():
    sF, sA, sR = fn(EVF), fn(appW), fn(rotW)
    pF, pA, pR = pval(sF, f), pval(sA, f), pval(sR, f)
    print(f"[{f:9s}] AUROC(rot|feas) {auroc(sR, sF):.3f}  rotate veto {(pR<0.01).mean():5.1%}  "
          f"approach FV {(pA<0.01).mean():4.1%}  feasible FA {(pF<0.01).mean():4.1%}")

KAP = np.array([0.2, 0.35, 0.5, 0.65, 0.8])
def edet(ps, alpha=0.01):
    Wv = np.ones(len(KAP))
    for t, p in enumerate(ps):
        Wv = np.maximum(1.0, Wv) * (KAP * p**(KAP-1))
        if Wv.mean() >= 1/alpha: return t
    return None

print("\n=== SEQUENTIAL e-detector (alpha=0.01) ===", flush=True)
for f, fn in FAM.items():
    faF = sum(edet(pval(fn(Wst), f)) is not None for Wst in evalF_streams)
    line = f"[{f:9s}] feasible-stream FA {faF}/{len(evalF_streams)}"
    for name, streams in (("fast", fastA), ("slow", slowA)):
        delays, miss, pre = [], 0, 0
        for Wst, onset in streams:
            t = edet(pval(fn(Wst), f))
            if t is not None and t < onset: pre += 1; continue
            if t is None: miss += 1; delays.append(len(Wst)-onset)
            else: delays.append(t-onset)
        line += f"  | {name}: delay {np.median(delays):.1f} miss {miss/len(streams):.0%} preFA {pre}"
    print(line, flush=True)

torch.save({"model_state_dict": net_cv.state_dict(), "arch": "conv", "feat": "eul_grip_rel_32", "nbins": NB,
            "score": "dte_posterior_mean", "cal_scores": cal["dte-conv"].tolist()},
           "/home/admin_025/X-Diffusion-Data/_private/runs/kstar_uni/dte_oneclass.pth")
print("\nsaved dte_oneclass.pth")
