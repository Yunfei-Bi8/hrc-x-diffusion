"""Necessity experiment: rotation-EMISSION rate without any shield.
Rolls mode-A (left-hand proto cond) from the trained start, N seeds x 40 cycles,
for each policy; measures per-rollout whether/when rotation-grade chunks are
emitted, plus at-lid behavior quality. Also an OFF-DISTRIBUTION variant
(perturbed start, the real-world drift scenario)."""
import json, sys, types
import numpy as np
import torch
from pathlib import Path
try:
    import line_profiler
except ImportError:
    _lp = types.ModuleType("line_profiler"); _lp.profile = lambda f: f; sys.modules["line_profiler"] = _lp
sys.path.insert(0, "/home/admin_025/X-Diffusion")
from models.xdiffusion.xDP import DiffusionPolicy, DiffusionPolicyConfig
from models.xdiffusion.hr_classifier import HumanRobotClassifier, HumanRobotClassifierConfig

DEV = "cuda"; PROTO = np.array([928, 250, 1, 22, 0], np.float32)
LID = np.array([0.944, 0.134, 0.541])
ck = torch.load("/home/admin_025/X-Diffusion-Data/_private/runs/kstar_uni/kstar_cls.pth", map_location=DEV, weights_only=False)
cst = np.array(ck["stats"]); TAU = float(ck["tau"])
cls = HumanRobotClassifier(obs_horizon=1, state_cond_dim=4, action_dim=4,
                           cfg=HumanRobotClassifierConfig(num_train_timesteps=101)).to(DEV)
cls.load_state_dict(ck["model_state_dict"]); cls.eval()
KG = np.arange(0, 101, 4)

@torch.no_grad()
def kstar(ch7, draws=8):
    ch = np.asarray(ch7, float)[:, 3:7].copy(); ch[:, :3] -= ch[:1, :3]
    A = torch.as_tensor(((2*(ch[None]-cst[0])/(cst[1]-cst[0]+1e-8)-1)).astype(np.float32), device=DEV)
    b = types.SimpleNamespace(action=A, state_cond=A[:, :1, :])
    P = np.zeros(len(KG))
    for ki, k in enumerate(KG):
        ts = torch.full((1,), int(k), dtype=torch.long, device=DEV)
        P[ki] = float(torch.stack([torch.sigmoid(cls.unified_forward(b, timesteps=ts).squeeze(-1)) for _ in range(draws)]).mean())
    hit = P >= 0.5
    idx = np.where(hit[:-1] & hit[1:])[0]
    return float(KG[idx[0]]) if len(idx) else (float(KG[-1]) if hit[-1] else 101.0)

def load_policy(run):
    P = Path(f"/home/admin_025/X-Diffusion-Data/_private/runs/{run}")
    stats = np.load(P / "dataset_stats/train.npy", allow_pickle=True).item()
    st_s = {k: np.asarray(v, np.float32).reshape(-1) for k, v in stats["state_cond"].items()}
    st_a = {k: np.asarray(v, np.float32) for k, v in stats["action"].items()}
    pc = DiffusionPolicyConfig(); pc.ddim.num_train_timesteps = 101; pc.delta_actions = False
    pol = DiffusionPolicy(num_points=5, action_dim=7, obs_horizon=1, pred_horizon=8, action_horizon=8,
                          state_cond_dim=int(st_s["max"].shape[0]), cfg=pc, use_image=False).to(DEV)
    sd = torch.load(P / "checkpoints/latest.pth", map_location=DEV, weights_only=False)
    pol.load_state_dict(sd["model_state_dict"]); pol.eval()
    sp = json.load(open(P / "start_pose.json"))
    s7 = np.array(sp["pos"] + sp["euler_xyz"] + [1.0])
    def act(s12):
        sn = (np.asarray(s12, np.float32)[None] - st_s["min"]) / (st_s["max"] - st_s["min"] + 1e-6) * 2 - 1
        with torch.no_grad():
            an = pol.act(torch.as_tensor(sn).reshape(1, 1, -1).to(DEV).float(), first_action_abs=None, cpu=True).numpy()[0]
        return (an + 1) / 2 * (st_a["max"] - st_a["min"] + 1e-6) + st_a["min"]
    return act, s7

def measure(run, n_seeds=12, cycles=40, perturb=0.0, tag=""):
    act, s7 = load_policy(run)
    emit = 0; first_emits = []; min_dl = []; drift = []
    rngp = np.random.default_rng(7)
    for seed in range(n_seeds):
        torch.manual_seed(1000 + seed)
        cur = s7.copy()
        if perturb > 0:
            cur[:3] += rngp.normal(0, perturb, 3)
            cur[3:6] += rngp.normal(0, 0.05, 3)
        fe = None; mdl = 1e9; arrived_at = None; arrive_pos = None; max_drift = 0.0
        for c in range(cycles):
            ch = act(np.concatenate([cur, PROTO]))
            ea = np.abs(np.diff(ch[:, 3:6], axis=0)).sum(0).max()
            ks = kstar(ch)
            dl = np.linalg.norm(ch[-1, :3] - LID) * 1000
            mdl = min(mdl, dl)
            if (ea >= 0.30 or ks > TAU) and fe is None:
                fe = c
            if dl < 90 and arrived_at is None:
                arrived_at = c; arrive_pos = ch[-1, :3].copy()
            if arrive_pos is not None:
                max_drift = max(max_drift, float(np.linalg.norm(ch[-1, :3] - arrive_pos)) * 1000)
            cur = np.concatenate([ch[-1][:3], ch[-1][3:6], [1.0 if ch[:3, 6].min() > 0.5 else 0.0]])
        emit += fe is not None
        if fe is not None: first_emits.append(fe)
        min_dl.append(mdl); drift.append(max_drift)
    print(f"[{run}{tag}] emission: {emit}/{n_seeds} rollouts "
          f"(first @ med {np.median(first_emits) if first_emits else -1:.0f} cyc)  "
          f"min d_lid med {np.median(min_dl):.0f}mm  post-arrival drift med {np.median(drift):.0f}mm")
    return emit / n_seeds

import sys
runs = sys.argv[1:] or ["policy_jar_uni_v10", "policy_jar_ab_nofeas"]
print(f"emission = rotation-grade chunk (eulA>=0.30 or k*>tau={TAU:.0f}) within 40 cycles, NO shield")
for r in runs:
    measure(r, tag="")
    measure(r, perturb=0.03, tag=" +30mm-start-perturb")
