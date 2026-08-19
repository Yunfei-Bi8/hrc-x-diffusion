"""Diagnose the 08-19 B-mode false vetoes: session drift of the DTE conformal
calibration. Score the three runs' EXECUTED chunks under the current cal, then
recalibrate with recent executed chunks and re-verify (incl. rotate power)."""
import glob, pickle, sys, types
import numpy as np
import torch, torch.nn as nn
from pathlib import Path
try:
    import line_profiler
except ImportError:
    _lp = types.ModuleType("line_profiler"); _lp.profile = lambda f: f; sys.modules["line_profiler"] = _lp
sys.path.insert(0, "/home/admin_025/X-Diffusion")
import h5py
T = 8; DEV = "cuda"; NB = 101
TASK = Path("/home/admin_025/X-Diffusion-Data/retargeted/jar_uni")
LO = np.array([-0.5,-0.5,-0.5,0.0]); HI = np.array([0.5,0.5,0.5,1.0])
def feat(W):
    W = np.asarray(W, float).copy(); W[..., :3] -= W[:, :1, :3]
    return (2*(W-LO)/(HI-LO)-1).reshape(len(W), -1)
class DTEConv(nn.Module):
    def __init__(self, c=4, h=128, nb=NB):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(c,h,3,padding=1), nn.SiLU(), nn.GroupNorm(8,h),
                                  nn.Conv1d(h,h,3,padding=1), nn.SiLU(), nn.GroupNorm(8,h),
                                  nn.Conv1d(h,h,3,padding=1), nn.SiLU())
        self.head = nn.Sequential(nn.Linear(2*h,256), nn.SiLU(), nn.Linear(256,nb))
    def forward(self, x):
        z = x.view(-1, T, 4).transpose(1,2); z = self.conv(z)
        return self.head(torch.cat([z.mean(-1), z.max(-1).values], -1))
CKP = "/home/admin_025/X-Diffusion-Data/_private/runs/kstar_uni/dte_oneclass.pth"
ck = torch.load(CKP, map_location=DEV, weights_only=False)
net = DTEConv().to(DEV); net.load_state_dict(ck["model_state_dict"]); net.eval()
cal_old = np.sort(np.asarray(ck["cal_scores"], np.float64))
@torch.no_grad()
def score(W4):
    tb = torch.arange(NB, dtype=torch.float32, device=DEV)
    x = torch.as_tensor(feat(W4), dtype=torch.float32, device=DEV)
    return (torch.softmax(net(x), -1) @ tb).cpu().numpy()
def s_alpha(cal, a):
    n = len(cal); return float(cal[min(n-1, int(np.ceil((n+1)*(1-a)))-1)])

# runs: split old-jar vs the recent DTE sessions (0818+)
old_runs, new_runs = [], []
for pk in sorted(glob.glob("/home/admin_025/mt_pi_codebase/deploy_logs/xdiff_naive_*/commands.pkl")):
    try: d = pickle.load(open(pk, "rb"))
    except Exception: continue
    if "jar" not in str(d.get("run_dir", "")): continue
    rows = [np.asarray(c["chunk"], float)[:, 3:7] for c in d["cycles"]
            if c["commanded"] is not None and len(np.atleast_1d(c["commanded"]))]
    rows = [r for r in rows if np.abs(np.diff(r[:, :3], axis=0)).sum(0).max() < 0.30]
    if len(rows) < 6: continue
    tag = pk.split("xdiff_naive_")[1][:8]
    (new_runs if tag >= "20260818" else old_runs).append((tag, np.stack(rows)))
print(f"old-session runs {len(old_runs)}  new-session (>=08-18) runs {len(new_runs)}")

sa = s_alpha(cal_old, 0.01)
print(f"current s_alpha(1%)={sa:.3f}")
newW = np.concatenate([w for _, w in new_runs])
s_new = score(newW)
print(f"NEW-session executed chunks: n={len(newW)}  frac s> s_alpha: {(s_new>sa).mean():.1%} "
      f"(nominal 1%)  s med {np.median(s_new):.3f} q99 {np.quantile(s_new,.99):.3f}")
oldW = np.concatenate([w for _, w in old_runs][-30:])
s_old = score(oldW)
print(f"OLD-session executed chunks: frac s>s_alpha: {(s_old>sa).mean():.1%}  s med {np.median(s_old):.3f}")

# per-cycle pessimistic-over-3-draws inflation (deploy votes on min over draws)
q = (s_new > sa).mean()
print(f"per-cycle flag prob with 3-draw pessimism ~ {1-(1-q)**3:.1%} -> 2-of-3 fires readily" )

# ---- recalibration: fold recent executed chunks into cal ----
cal_new = np.sort(np.concatenate([cal_old, s_new]))
for a in (0.01, 0.005, 0.0033):
    sa2 = s_alpha(cal_new, a)
    fa_new = (s_new > sa2).mean(); fa_cyc = 1-(1-fa_new)**3
    print(f"RECAL alpha={a}: s_alpha={sa2:.3f}  new-session chunk FA {fa_new:.2%}  per-cycle(3draw) {fa_cyc:.2%}")

# rotate power must survive the higher threshold
rotW = []
for f in sorted(TASK.glob("human/demo*.h5")):
    eid = int(f.stem[4:])
    if not (100 <= eid < 200): continue
    with h5py.File(f) as h:
        a_ = np.concatenate([np.array(h["ee_euler"]), np.array(h["gripper_open"])[:, None]], 1)
        fm = np.array(h["feas_mask"]); nb = int(h.attrs.get("uni_bridge", 0))
    a_, fm = a_[nb:], fm[nb:]
    n = len(a_)//T
    for i in range(n):
        if fm[i*T:(i+1)*T].max() == 0: rotW.append(a_[i*T:(i+1)*T])
rotW = np.stack(rotW); s_rot = score(rotW)
for a in (0.01, 0.0033):
    sa2 = s_alpha(cal_new, a)
    print(f"rotate veto @recal alpha={a}: {(s_rot>sa2).mean():.1%}  (old cal was {(s_rot>sa).mean():.1%})")

np.save("/tmp/claude-1000/-home-admin-025-mt-pi-codebase/78a38b32-b231-4a83-ab10-6c5d577f1100/scratchpad/cal_new.npy", cal_new)
print(f"cal sizes: old {len(cal_old)} -> new {len(cal_new)}")
