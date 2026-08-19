"""kNN + conformal p-value feasibility monitor — offline replay, zero training.

Sets:
  D_ref  = robot teleop (episodes[:-4]) sliding windows  +  executed deploy chunks (runs idx%4 in {0,2})
  D_cal  = teleop[-2:] + deploy runs idx%4==1   (episode-level disjoint from D_ref)
  D_evalF= teleop[-4:-2] + deploy runs idx%4==3 (feasible eval, untouched)
Test:
  role-A approach/rotate windows, full role-A streams (fast onset),
  slow-rotation streams injected on eval-feasible deploy streams.
Score families under the SAME conformal layer: kNN distance | k* (frozen classifier).
"""
import glob, pickle, sys, types
import numpy as np
import torch
from pathlib import Path
from scipy.spatial import cKDTree
try:
    import line_profiler
except ImportError:
    _lp = types.ModuleType("line_profiler"); _lp.profile = lambda f: f; sys.modules["line_profiler"] = _lp
sys.path.insert(0, "/home/admin_025/X-Diffusion")
from models.xdiffusion.hr_classifier import HumanRobotClassifier, HumanRobotClassifierConfig
import h5py

T = 8; DEV = "cuda"
TASK = Path("/home/admin_025/X-Diffusion-Data/retargeted/jar_uni")
rng = np.random.default_rng(0)
LO = np.array([-0.5, -0.5, -0.5, 0.0]); HI = np.array([0.5, 0.5, 0.5, 1.0])

def feat(W):                       # (n,8,4) absolute -> (n,32) rel-normalized
    W = np.asarray(W, float).copy()
    W[..., :3] -= W[:, :1, :3]
    Wn = 2 * (W - LO) / (HI - LO) - 1
    return Wn.reshape(len(W), -1)

def load4(f):
    with h5py.File(f) as h:
        a = np.concatenate([np.array(h["ee_euler"]), np.array(h["gripper_open"])[:, None]], 1)
        fm = np.array(h["feas_mask"]) if "feas_mask" in h else np.ones(len(a), np.int32)
        role = h.attrs.get("role", "B"); nb = int(h.attrs.get("uni_bridge", 0))
    return a[nb:].astype(float), fm[nb:], role

def nonoverlap(a):
    n = len(a) // T
    return np.stack([a[i*T:(i+1)*T] for i in range(n)]) if n else np.zeros((0, T, 4))

def sliding(a, stride=2):
    idx = range(0, len(a) - T, stride)
    return np.stack([a[t:t+T] for t in idx]) if len(a) > T else np.zeros((0, T, 4))

# ---------- load feasible sources ----------
teleop = []
for f in sorted(TASK.glob("robot/demo*.h5")):
    if int(f.stem[4:]) >= 40: continue
    a, _, _ = load4(f); teleop.append(a)
runs = []
for pk in sorted(glob.glob("/home/admin_025/mt_pi_codebase/deploy_logs/xdiff_naive_*/commands.pkl")):
    try: d = pickle.load(open(pk, "rb"))
    except Exception: continue
    if "jar" not in str(d.get("run_dir", "")): continue
    rows = [np.asarray(c["chunk"], float)[:, 3:7] for c in d["cycles"]
            if c["commanded"] is not None and len(np.atleast_1d(c["commanded"]))]
    rows = [r for r in rows if np.abs(np.diff(r[:, :3], axis=0)).sum(0).max() < 0.30]
    if len(rows) >= 10: runs.append(np.stack(rows))
print(f"teleop episodes {len(teleop)}, deploy runs {len(runs)}")

# ---------- episode-level splits ----------
tel_ref, tel_evalF, tel_cal = teleop[:-4], teleop[-4:-2], teleop[-2:]
run_ref  = [r for i, r in enumerate(runs) if i % 4 in (0, 2)]
run_cal  = [r for i, r in enumerate(runs) if i % 4 == 1]
run_evalF= [r for i, r in enumerate(runs) if i % 4 == 3]

REF = np.concatenate([feat(sliding(a)) for a in tel_ref] + [feat(r) for r in run_ref])
tree = cKDTree(REF)
def knn_score(W4, k=5):
    if len(W4) == 0: return np.zeros(0)
    d, _ = tree.query(feat(W4), k=k)
    return d[:, -1] if k > 1 else d
print(f"D_ref: {len(REF)} chunks   D_cal episodes: {len(tel_cal)+len(run_cal)}")

# ---------- frozen classifier k* (ablation score) ----------
ck = torch.load("/home/admin_025/X-Diffusion-Data/_private/runs/kstar_uni/kstar_cls.pth", map_location=DEV, weights_only=False)
cst = np.array(ck["stats"])
cls = HumanRobotClassifier(obs_horizon=1, state_cond_dim=4, action_dim=4,
                           cfg=HumanRobotClassifierConfig(num_train_timesteps=101)).to(DEV)
cls.load_state_dict(ck["model_state_dict"]); cls.eval()
KG = np.arange(0, 101, 4)
@torch.no_grad()
def kstar_score(W4, draws=8, bs=2048):
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

# ---------- calibration ----------
CAL_W = np.concatenate([nonoverlap(a) for a in tel_cal] + [r for r in run_cal])
cal = {"knn": np.sort(knn_score(CAL_W)), "kstar": np.sort(kstar_score(CAL_W))}
print(f"D_cal chunks: {len(CAL_W)}")
def pval(s, fam):
    c = cal[fam]
    return (1 + len(c) - np.searchsorted(c, s, side="left")) / (len(c) + 1)

# ---------- eval sets ----------
evalF_streams = [nonoverlap(a) for a in tel_evalF] + [r for r in run_evalF]
appW, rotW, fastA = [], [], []
for f in sorted(TASK.glob("human/demo*.h5")):
    eid = int(f.stem[4:])
    if not (100 <= eid < 200): continue
    a, fm, _ = load4(f)
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
print(f"evalF streams {len(evalF_streams)}  approach {len(appW)}  rotate {len(rotW)}  fastA {len(fastA)}  slowA {len(slowA)}")

score_fn = {"knn": knn_score, "kstar": kstar_score}

# ---------- per-chunk metrics ----------
print("\n=== PER-CHUNK (conformal p, identical decision layer) ===")
alphas = np.array([0.002, 0.005, 0.01, 0.02, 0.05, 0.1])
EVF = np.concatenate(evalF_streams)
for fam in ("knn", "kstar"):
    pF = pval(score_fn[fam](EVF), fam)
    pA = pval(score_fn[fam](appW), fam)
    pR = pval(score_fn[fam](rotW), fam)
    fa = [(pF < a).mean() for a in alphas]
    print(f"[{fam:5s}] validity: alpha {alphas.tolist()}")
    print(f"         empFA {[f'{v:.3f}' for v in fa]}")
    print(f"         @a=0.01: rotate veto {(pR<0.01).mean():.1%}  approach false-veto {(pA<0.01).mean():.1%}  feasible FA {(pF<0.01).mean():.1%}")

# ---------- sequential: e-detector ----------
KAPPAS = np.array([0.2, 0.35, 0.5, 0.65, 0.8])
def edetector_alarm(ps, alpha):
    W = np.ones(len(KAPPAS))
    for t, p in enumerate(ps):
        W = np.maximum(1.0, W) * (KAPPAS * p**(KAPPAS-1))
        if W.mean() >= 1/alpha: return t
    return None
def mofn_alarm(ks, tau=17.0, M=2, N=3):
    v = []
    for t, k in enumerate(ks):
        v.append(k > tau); v = v[-N:]
        if sum(v) >= M: return t
    return None

print("\n=== SEQUENTIAL (alpha=0.01) ===")
for fam in ("knn", "kstar"):
    for name, streams in (("fast", fastA), ("slow", slowA)):
        delays, miss, fa = [], 0, 0
        for Wst, onset in streams:
            ps = pval(score_fn[fam](Wst), fam)
            t = edetector_alarm(ps, 0.01)
            if t is not None and t < onset: fa += 1; continue
            if t is None: miss += 1; delays.append(len(Wst)-onset)
            else: delays.append(t-onset)
        # false alarms on feasible eval streams
        faF = sum(edetector_alarm(pval(score_fn[fam](Wst), fam), 0.01) is not None for Wst in evalF_streams)
        print(f"[{fam:5s}|{name}] delay med {np.median(delays):.1f} cyc  miss {miss/len(streams):.1%}  "
              f"pre-onset FA {fa}/{len(streams)}  feasible-stream FA {faF}/{len(evalF_streams)}")
# deployed-rule reference on the same streams
for name, streams in (("fast", fastA), ("slow", slowA)):
    delays, miss = [], 0
    for Wst, onset in streams:
        t = mofn_alarm(kstar_score(Wst))
        if t is None: miss += 1; delays.append(len(Wst)-onset)
        elif t >= onset: delays.append(t-onset)
    faF = sum(mofn_alarm(kstar_score(Wst)) is not None for Wst in evalF_streams)
    print(f"[old tau=17 2of3|{name}] delay med {np.median(delays):.1f}  miss {miss/len(streams):.1%}  feasible-stream FA {faF}/{len(evalF_streams)}")

# ---------- k sweep ----------
print("\n=== kNN k-sweep (@a=0.01) ===")
for kk in (1, 5, 10, 20):
    calk = np.sort(knn_score(CAL_W, k=kk))
    def pv(s): return (1+len(calk)-np.searchsorted(calk, s, side="left"))/(len(calk)+1)
    pR = pv(knn_score(rotW, k=kk)); pA = pv(knn_score(appW, k=kk)); pF = pv(knn_score(EVF, k=kk))
    print(f"k={kk:2d}: rotate veto {(pR<0.01).mean():.1%}  approach FV {(pA<0.01).mean():.1%}  feasible FA {(pF<0.01).mean():.1%}")

np.savez("/home/admin_025/X-Diffusion/figures/knn_conformal_data.npz",
         cal_knn=cal["knn"], cal_kstar=cal["kstar"])
print("\ndone")
