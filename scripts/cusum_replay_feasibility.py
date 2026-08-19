"""Final replay: clean + SLOW-ROTATION (F5 evasion) stream sets, 3 rules, figure."""
import glob, pickle, sys, types
import numpy as np
import torch
from pathlib import Path
try:
    import line_profiler
except ImportError:
    _lp = types.ModuleType("line_profiler"); _lp.profile = lambda f: f; sys.modules["line_profiler"] = _lp
sys.path.insert(0, "/home/admin_025/X-Diffusion")
from models.xdiffusion.hr_classifier import HumanRobotClassifier, HumanRobotClassifierConfig
import h5py
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family":"serif","font.serif":["DejaVu Serif"],
    "mathtext.fontset":"dejavuserif","axes.linewidth":0.8,"figure.dpi":140})

DEV="cuda"; T=8
TASK=Path("/home/admin_025/X-Diffusion-Data/retargeted/jar_uni")
rng=np.random.default_rng(0); torch.manual_seed(0)
ck=torch.load("/home/admin_025/X-Diffusion-Data/_private/runs/kstar_uni/kstar_cls.pth",map_location=DEV,weights_only=False)
st=np.array(ck["stats"]); TAU_DEP=float(ck["tau"])
cls=HumanRobotClassifier(obs_horizon=1,state_cond_dim=4,action_dim=4,
    cfg=HumanRobotClassifierConfig(num_train_timesteps=101)).to(DEV)
cls.load_state_dict(ck["model_state_dict"]); cls.eval()
KG=np.arange(0,101,4)
def rel(W): W=W.copy(); W[...,:3]-=W[:,:1,:3]; return W
@torch.no_grad()
def kstar_batch(W4,draws=8,bs=1024):
    Wr=rel(W4)
    A_all=torch.as_tensor(((2*(Wr-st[0])/(st[1]-st[0]+1e-8)-1)).astype(np.float32),device=DEV)
    P=np.zeros((len(W4),len(KG)),np.float32)
    for i0 in range(0,len(W4),bs):
        A=A_all[i0:i0+bs]; b=types.SimpleNamespace(action=A,state_cond=A[:,:1,:])
        for ki,k in enumerate(KG):
            ts=torch.full((len(A),),int(k),dtype=torch.long,device=DEV)
            ps=[torch.sigmoid(cls.unified_forward(b,timesteps=ts).squeeze(-1)) for _ in range(draws)]
            P[i0:i0+bs,ki]=torch.stack(ps).mean(0).cpu().numpy()
    ks=np.full(len(W4),101.0); hit=P>=0.5
    for i in range(len(W4)):
        idx=np.where(hit[i][:-1]&hit[i][1:])[0]
        if len(idx): ks[i]=KG[idx[0]]
        elif hit[i][-1]: ks[i]=KG[-1]
    return ks
def load4(f):
    with h5py.File(f) as h:
        a=np.concatenate([np.array(h["ee_euler"]),np.array(h["gripper_open"])[:,None]],1)
        fm=np.array(h["feas_mask"]) if "feas_mask" in h else np.ones(len(a),np.int32)
        role=h.attrs.get("role","B"); nb=int(h.attrs.get("uni_bridge",0))
    return a[nb:].astype(np.float64),fm[nb:],role
def chunks_of(a,fm=None):
    n=len(a)//T; W=np.stack([a[i*T:(i+1)*T] for i in range(n)])
    onset=None
    if fm is not None and (fm==0).any(): onset=int(np.argmax(fm==0))//T
    return W,onset

feas_streams,onset_streams=[],[]; feas_isB=[]
for f in sorted(TASK.glob("human/demo*.h5")):
    eid=int(f.stem[4:])
    if eid>=200: continue
    a,fm,role=load4(f); W,onset=chunks_of(a,fm)
    if len(W)<4: continue
    if role=="B": feas_streams.append(W); feas_isB.append(True)
    elif onset is not None and onset>=2: onset_streams.append((W,onset))
for f in sorted(TASK.glob("robot/demo*.h5")):
    if int(f.stem[4:])>=40: continue
    a,fm,_=load4(f); W,_=chunks_of(a)
    if len(W)>=4: feas_streams.append(W); feas_isB.append(False)
dep_all=[]
for pk in sorted(glob.glob("/home/admin_025/mt_pi_codebase/deploy_logs/xdiff_naive_*/commands.pkl")):
    try: d=pickle.load(open(pk,"rb"))
    except Exception: continue
    if "jar" not in str(d.get("run_dir","")): continue
    rows=[np.asarray(c["chunk"],np.float64)[:,3:7] for c in d["cycles"]
          if c["commanded"] is not None and len(np.atleast_1d(c["commanded"]))]
    rows=[r for r in rows if np.abs(np.diff(r[:,:3],axis=0)).sum(0).max()<0.30]
    if len(rows)>=10: feas_streams.append(np.stack(rows)); feas_isB.append(False); dep_all.extend(rows)
print(f"feasible {len(feas_streams)}  fast-onset {len(onset_streams)}")

# SLOW-ROTATION streams (F5 evasion, mirrors run 013126's 25deg creep): take a
# feasible stream, from its midpoint inject a GENTLE coherent yaw ramp continuing
# ACROSS chunks — each chunk's own eulA stays 0.10-0.20 (sub-threshold per chunk).
slow_streams=[]
for W in feas_streams[:120]:
    if len(W)<10: continue
    Wm=W.copy(); onset=len(W)//2
    rate=rng.uniform(0.014,0.028)              # rad/STEP -> per-chunk eulA ~ rate*7 = 0.10-0.20
    ax=2                                        # yaw (like the real creep)
    phase=0.0
    for c in range(onset,len(Wm)):
        ramp=phase+np.arange(T)*rate
        Wm[c,:,ax]=Wm[c,:,ax]+ramp
        phase=ramp[-1]+rate                     # CONTINUES across chunks (slow global twist)
    slow_streams.append((Wm,onset))
print(f"slow-onset {len(slow_streams)}")

all_W=np.concatenate(feas_streams+[w for w,_ in onset_streams]+[w for w,_ in slow_streams])
lens=[len(w) for w in feas_streams]+[len(w) for w,_ in onset_streams]+[len(w) for w,_ in slow_streams]
ks_all=kstar_batch(all_W)
seq=[];i0=0
for L in lens: seq.append(ks_all[i0:i0+L]); i0+=L
nf=len(feas_streams); no=len(onset_streams)
feas_ks_all=seq[:nf]
feas_ks=[k for k,isb in zip(feas_ks_all,feas_isB) if not isb]   # robot + executed policy = deploy distribution
bh_ks  =[k for k,isb in zip(feas_ks_all,feas_isB) if isb]       # B-human replay (stress reference)
print(f"FA eval set: {len(feas_ks)} deploy-realistic streams (+{len(bh_ks)} B-human as stress)")
fast_ks=list(zip(seq[nf:nf+no],[o for _,o in onset_streams]))
slow_ks=list(zip(seq[nf+no:],[o for _,o in slow_streams]))
post=np.concatenate([k[o:] for k,o in slow_ks])
print("slow post-onset k* dist:",{int(v):int((post==v).sum()) for v in np.unique(post)[:8]},
      " frac>tau17:",float((post>17).mean()))

# label-free LLR
f0_src=[]
for f in sorted(TASK.glob("robot/demo*.h5")):
    if int(f.stem[4:])>=40: continue
    a,_,_=load4(f); W,_=chunks_of(a); f0_src.append(W)
f0_src=np.concatenate(f0_src+([np.stack(dep_all)] if dep_all else []))
def counterfactual(W,n):
    """COMPOSITE post-change family: rotation intensity log-uniform over
    [0.012, 0.12] rad/step (mixture-CUSUM for a composite alternative) —
    the narrow strong-only band left slow rotation with NEGATIVE LLR."""
    idx=rng.integers(0,len(W),n); C=rel(W[idx]).copy()
    for i in range(n):
        ax=rng.integers(0,3)
        if rng.random()<0.5:
            r=np.exp(rng.uniform(np.log(0.012),np.log(0.12)))*(1 if rng.random()<0.5 else -1)
            C[i,:,ax]+=np.arange(T)*r
        else:
            C[i,:,ax]+=np.exp(rng.uniform(np.log(0.08),np.log(0.6)))*np.sin(np.arange(T)*rng.uniform(0.6,1.6)+rng.uniform(0,2*np.pi))
    return C
def counterfactual_band(W,n,lo,hi):
    idx=rng.integers(0,len(W),n); C=rel(W[idx]).copy()
    for i in range(n):
        ax=rng.integers(0,3)
        if rng.random()<0.5:
            r=np.exp(rng.uniform(np.log(lo),np.log(hi)))*(1 if rng.random()<0.5 else -1)
            C[i,:,ax]+=np.arange(T)*r
        else:
            C[i,:,ax]+=np.exp(rng.uniform(np.log(lo*7),np.log(hi*5)))*np.sin(np.arange(T)*rng.uniform(0.6,1.6)+rng.uniform(0,2*np.pi))
    return C
ks_f0=kstar_batch(f0_src)
ks_f1_slow=kstar_batch(counterfactual_band(f0_src,3000,0.012,0.045))
ks_f1_fast=kstar_batch(counterfactual_band(f0_src,3000,0.055,0.12))
BINS=np.append(KG,101.0)
def pmf(ks):
    p=np.array([(ks==b).sum() for b in BINS],float)+0.5; return p/p.sum()
b_of={b:i for i,b in enumerate(BINS)}
LLR_S=np.log(pmf(ks_f1_slow))-np.log(pmf(ks_f0))
LLR_F=np.log(pmf(ks_f1_fast))-np.log(pmf(ks_f0))
def llr_s(k): return LLR_S[b_of[k]]
def llr_f(k): return LLR_F[b_of[k]]
print("LLR_slow @0,8,12,16,24:",[round(float(llr_s(v)),2) for v in (0,8,12,16,24)])
print("LLR_fast @0,24,40,72,101:",[round(float(llr_f(v)),2) for v in (0,24,40,72,101)])

def alarm_mofn(ks,tau,M=2,N=3):
    v=[]
    for t,k in enumerate(ks):
        v.append(k>tau); v=v[-N:]
        if sum(v)>=M: return t
    return None
def alarm_thresh(ks,tau):
    for t,k in enumerate(ks):
        if k>tau: return t
    return None
def cusum_peaks(ks,llr_fn):
    Sv=0.0; pk=0.0
    for k in ks:
        Sv=max(0.0,Sv+llr_fn(k)); pk=max(pk,Sv)
    return pk
# per-channel thresholds from the (1-a/2) quantile of feasible-stream CUSUM PEAKS
PK_S=np.array([cusum_peaks(ks,llr_s) for ks in feas_ks])
PK_F=np.array([cusum_peaks(ks,llr_f) for ks in feas_ks])
def h_pair(alpha):
    q=1-alpha/2
    return (float(np.quantile(PK_S,min(q,1.0)))+1e-6, float(np.quantile(PK_F,min(q,1.0)))+1e-6)
def alarm_cusum(ks,hpair):
    hs,hf=hpair; Ss=Sf=0.0
    for t,k in enumerate(ks):
        Ss=max(0.0,Ss+llr_s(k)); Sf=max(0.0,Sf+llr_f(k))
        if Ss>hs or Sf>hf: return t
    return None
def evaluate(alarm_fn,param,onset_set):
    fa=0; dom=len(feas_ks); delays=[]; miss=0
    for ks in feas_ks:
        if alarm_fn(ks,param) is not None: fa+=1
    for ks,onset in onset_set:
        dom+=1
        t=alarm_fn(ks,param)
        if t is not None and t<onset: fa+=1; continue
        horizon=len(ks)-onset
        if t is None: miss+=1; delays.append(horizon)
        else: delays.append(t-onset)
    return fa/dom,float(np.mean(delays)),miss/max(1,len(onset_set))

taus=np.arange(4,100,2)
alphas=np.concatenate([np.linspace(0.002,0.03,15),np.linspace(0.04,0.25,12)])
hpairs=[h_pair(a) for a in alphas]
def sweep(onset_set):
    return {"deployed":np.array([evaluate(alarm_mofn,t,onset_set) for t in taus]),
            "plain":np.array([evaluate(alarm_thresh,t,onset_set) for t in taus]),
            "cusum":np.array([evaluate(alarm_cusum,hp,onset_set) for hp in hpairs])}
FAST=sweep(fast_ks); SLOW=sweep(slow_ks)
depF=evaluate(alarm_mofn,TAU_DEP,fast_ks); depS=evaluate(alarm_mofn,TAU_DEP,slow_ks)
def at_fa(c,fam):
    ok=c[c[:,0]<=fam]; return ok[np.argmin(ok[:,1])] if len(ok) else None
for tag,res in (("FAST",FAST),("SLOW",SLOW)):
    for k in ("plain","deployed","cusum"):
        r=at_fa(res[k],0.01)
        print(f"{tag:5s} {k:9s} @FA<=1%: FA={r[0]:.1%} delay={r[1]:.2f} miss={r[2]:.1%}" if r is not None else f"{tag} {k}: none<=1%")

INK="#22303a"; C={"deployed":"#c2571f","plain":"#8a8f94","cusum":"#1f6fb2"}
LBL={"deployed":r"deployed: $k^\ast\!>\!\tau$ + 2-of-3 vote","plain":r"plain: first $k^\ast\!>\!\tau$",
     "cusum":"CUSUM sequential test (ours)"}
def pareto(c):
    o=c[np.argsort(c[:,0],kind="stable")]; best=[]; bd=np.inf
    for r in o:
        if r[1]<bd-1e-9: best.append(r); bd=r[1]
    return np.array(best)
fig,axes=plt.subplots(1,3,figsize=(15.8,4.7))
for ax,(res,ttl,deppt,ymax) in zip(axes[:2],
        [(FAST,"(a) abrupt rotation onset (real role-A streams)",depF,7),
         (SLOW,"(b) SLOW rotation (evasion regime, per-chunk signal sub-threshold)",depS,16)]):
    for key in ("plain","deployed","cusum"):
        pf=pareto(res[key])
        ax.step(pf[:,0]*100,pf[:,1],where="post",color=C[key],lw=2.4 if key=="cusum" else 1.7,
                label=LBL[key],zorder=5 if key=="cusum" else 3)
        ax.plot(res[key][:,0]*100,res[key][:,1],".",color=C[key],ms=3,alpha=0.3)
    ax.plot(deppt[0]*100,deppt[1],marker="*",ms=16,mfc="#7a2f8a",mec="white",ls="None",zorder=8,
            label=r"deployed op. point ($\tau{=}17$, 2-of-3)")
    ax.set_xlabel("false-alarm rate on feasible streams  [%]",fontsize=10.5)
    ax.set_ylabel("detection delay after onset  [cycles]",fontsize=10.5)
    ax.set_title(ttl,fontsize=11.5)
    ax.set_xlim(-0.3,12); ax.set_ylim(0,ymax)
    sec=ax.secondary_yaxis("right",functions=(lambda x:x*0.6,lambda x:x/0.6))
    sec.set_ylabel("[s @ 0.6 s/cycle]",fontsize=9,color="#666"); sec.tick_params(labelsize=8,colors="#666")
    ax.grid(alpha=0.25,lw=0.6)
axes[0].legend(fontsize=8.4,loc="upper right",framealpha=0.95)

ax=axes[2]; ks,onset=slow_ks[3]; tt=np.arange(len(ks))
ax2=ax.twinx()
ax.bar(tt,ks,color=["#9fb8c9" if t<onset else "#e59f7e" for t in tt],width=0.75,label=r"$k^\ast_t$ (per chunk)")
ax.set_ylim(0,TAU_DEP+6)
ax.axhline(TAU_DEP,color=C["deployed"],ls=(0,(4,2)),lw=1.3)
ax.text(1.0,TAU_DEP+0.8,r"$\tau{=}17$: per-chunk rule never fires",color=C["deployed"],fontsize=8.5,va="bottom")
sv=fv=0.0; Ss=[]
for k in ks:
    sv=max(0.0,sv+llr_s(k)); fv=max(0.0,fv+llr_f(k)); Ss.append(max(sv,fv))
h_star=h_pair(0.01)[0]
ax2.plot(tt,Ss,color=C["cusum"],lw=2.4,label=r"CUSUM $S_t$")
ax2.axhline(h_star,color=C["cusum"],ls=(0,(4,2)),lw=1.4)
ax2.text(0.3,h_star+0.4,r"$h$",color=C["cusum"],fontsize=11)
ax.axvline(onset-0.5,color=INK,ls=":",lw=1.4)
ax.text(onset+0.5,TAU_DEP+3.6,"slow rotation\nbegins",fontsize=8.5,color=INK,va="top")
al=next((t for t,s in enumerate(Ss) if s>h_star),None)
if al is not None:
    ax2.plot([al],[Ss[al]],marker="v",ms=11,mfc="#d1483f",mec="white",ls="None",zorder=9)
    ax2.text(al,Ss[al]+0.8,"alarm",color="#d1483f",fontsize=10,ha="center",weight="bold")
ax.set_xlabel("control cycle $t$",fontsize=10.5); ax.set_ylabel(r"$k^\ast_t$",fontsize=11)
ax2.set_ylabel(r"$S_t$",fontsize=11)
ax.set_title("(c) mechanism: weak evidence accumulates",fontsize=11.5)
h1,l1=ax.get_legend_handles_labels(); h2,l2=ax2.get_legend_handles_labels()
ax.legend(h1+h2,l1+l2,fontsize=8.6,loc="upper left",framealpha=0.95)
fig.suptitle("Sequential change detection (CUSUM) vs per-chunk thresholding for the feasibility veto — replay on real jar-task streams",
             fontsize=13.5,weight="bold",y=1.03,color=INK)
fig.tight_layout()
fig.savefig("/home/admin_025/X-Diffusion/figures/cusum_replay_fig.pdf",bbox_inches="tight")
fig.savefig("/home/admin_025/X-Diffusion/figures/cusum_replay_fig.png",dpi=300,bbox_inches="tight")
print("saved")
