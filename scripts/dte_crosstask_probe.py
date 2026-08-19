"""Cross-task probe: score HANDOVER-task executed chunks with the JAR-trained
one-class conv-DTE. Handover feasible motion CONTAINS real wrist reorientation
-> under the jar task's feasible-set it should read as off-manifold (task-
specificity), while a handover-trained DTE (same recipe, zero negatives) should
accept them (transfer-of-METHOD, not of the feasible set)."""
import glob, pickle, sys, types
import numpy as np
import torch, torch.nn as nn
T = 8; DEV = "cuda"; NB = 101
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
ck = torch.load("/home/admin_025/X-Diffusion-Data/_private/runs/kstar_uni/dte_oneclass.pth",
                map_location=DEV, weights_only=False)
net = DTEConv().to(DEV); net.load_state_dict(ck["model_state_dict"]); net.eval()
cal = np.array(ck["cal_scores"])
@torch.no_grad()
def score(W4):
    tb = torch.arange(NB, dtype=torch.float32, device=DEV)
    x = torch.as_tensor(feat(W4), dtype=torch.float32, device=DEV)
    return (torch.softmax(net(x), -1) @ tb).cpu().numpy()
def pv(s): return (1+len(cal)-np.searchsorted(np.sort(cal), s, side="left"))/(len(cal)+1)

groups = {}
for pk in sorted(glob.glob("/home/admin_025/mt_pi_codebase/deploy_logs/xdiff_naive_*/commands.pkl")):
    try: d = pickle.load(open(pk, "rb"))
    except Exception: continue
    rd = str(d.get("run_dir", ""))
    tag = ("handover" if "handover" in rd else "hrc" if "hrc" in rd else None)
    if tag is None: continue
    rows = [np.asarray(c["chunk"], float)[:, 3:7] for c in d["cycles"]
            if c["commanded"] is not None and len(np.atleast_1d(c["commanded"]))]
    if rows: groups.setdefault(tag, []).extend(rows)
for tag, rows in groups.items():
    W = np.stack(rows)
    ea = np.abs(np.diff(W[:, :, :3], axis=1)).sum(1).max(-1)
    s = score(W); p = pv(s)
    print(f"[{tag}] n={len(W)} executed chunks | euler-activity med {np.median(ea):.2f} "
          f"q90 {np.quantile(ea,.9):.2f} | jar-DTE flag rate (p<0.01): {(p<0.01).mean():.1%} "
          f"| quiet-subset (eulA<0.15) flag: {(p[ea<0.15]<0.01).mean() if (ea<0.15).any() else -1:.1%}")
