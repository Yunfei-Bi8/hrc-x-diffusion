"""Feasibility critic v2: REAL affordance negatives from the 10-08 rotating-jar
dataset + the validated synthetic recipe (state-channel, low-k, mixed-span).

Labels for real jar windows (24-step, episode-disjoint):
  afford = from the exporter's feas_mask (pure-approach/lift -> 1, pure-rotate -> 0,
           mixed windows excluded from training for label purity)
  kin    = the ANALYTIC envelope check per window (auto-label: the S-slowdown puts
           the approach inside the robot envelope; whatever still violates gets
           kin=0 from physics, not from guesswork)
Gates: held-out jar episodes -> per-phase AUROC + head attribution + the synthetic
battery spot-check. Saves jarcrit.pth (+ stats) for the deployment shield.
Run (mtpi): python scripts/feascritic_jar.py --steps 3000
"""
import argparse, json, sys, types
from pathlib import Path

import h5py
import numpy as np
import torch

sys.path.insert(0, "/home/admin_025/X-Diffusion")
import scripts.feascritic_synth_exp as fx
from models.xdiffusion.hr_classifier import FeasibilityCritic, FeasibilityCriticConfig

JAR = Path("/home/admin_025/X-Diffusion-Data/retargeted/jar_infeasible/human")
OUT = Path("/home/admin_025/X-Diffusion-Data/_private/runs/feascritic_synth")
DEV = "cuda"
T = fx.T
N_HELDOUT = 8


def load_jar():
    demos = sorted(JAR.glob("demo*.h5"))
    assert demos, f"no jar demos under {JAR}"
    tr, va = [], []
    for i, f in enumerate(demos):
        with h5py.File(f) as h:
            a = np.concatenate([np.array(h["ee_pos"]), np.array(h["ee_euler"]),
                                np.array(h["gripper_open"])[:, None]], 1).astype(np.float64)
            fm = np.array(h["feas_mask"])
        (va if i >= len(demos) - N_HELDOUT else tr).append((a, fm))
    def windows(eps):
        pos, neg, trans = [], [], []
        for a, fm in eps:
            for t in range(0, len(a) - T, 2):
                w, m = a[t:t + T], fm[t:t + T]
                if m.all():
                    pos.append(w)
                elif (1 - m).all():
                    neg.append(w)
                elif m[:16].mean() > 0.5 and (1 - m[16:]).all():
                    # TRANSITION window: feasible-dominant context, rotate SUFFIX —
                    # exactly the deploy scoring shape [16 executed || 8 predicted]
                    # at rotation ONSET. Robot-observed 08-12: without these the
                    # afford head scores onset windows mid-range -> slow veto.
                    trans.append(w)
        z = np.zeros((0, T, 7))
        return (np.stack(pos) if pos else z), (np.stack(neg) if neg else z), \
               (np.stack(trans) if trans else z)
    return windows(tr), windows(va)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--run-name", default="jarcrit")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-draws", type=int, default=4)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    train, val = fx.load_pool()
    env = fx.measure_envelope(train)
    (jtr_pos, jtr_neg, jtr_trn), (jva_pos, jva_neg, jva_trn) = load_jar()
    print(f"jar windows: train pos {len(jtr_pos)} / rotate-neg {len(jtr_neg)} / transition {len(jtr_trn)}; "
          f"heldout pos {len(jva_pos)} / rotate-neg {len(jva_neg)} / transition {len(jva_trn)}")
    ek_trn = fx.envelope_check(jtr_trn, env) if len(jtr_trn) else np.zeros(0, bool)
    ek_tr = fx.envelope_check(jtr_neg, env)
    print(f"rotate windows inside kin envelope: train {ek_tr.mean():.0%} "
          f"(these get kin=1: clean dissociation; the rest kin=0 by physics)")

    allw = np.concatenate([train["robot"], train["human"], jtr_pos, jtr_neg])
    st = (allw.reshape(-1, 7).min(0), allw.reshape(-1, 7).max(0))
    anch = np.concatenate([train["robot"], train["human"]])[:, 0, :]
    anch_closed = anch[anch[:, 6] < 0.5]
    base_tr = np.concatenate([train["robot"], train["human"]])
    mov = np.linalg.norm(np.diff(base_tr[..., :3], axis=1), axis=-1).max(1) >= 0.5 * env["p95"]
    base_tr_mov = base_tr[mov]

    cfg = FeasibilityCriticConfig(num_train_timesteps=101, finger_dim=0,
                                  head_names=["identity", "kin_feas", "afford_feas"])
    crit = FeasibilityCritic(obs_horizon=1, state_cond_dim=7, action_dim=14, cfg=cfg).to(DEV)
    opt = torch.optim.AdamW(crit.parameters(), lr=1e-4, weight_decay=1e-4)

    def norm(x):
        return (2 * (x - st[0]) / (st[1] - st[0] + 1e-8) - 1).astype(np.float32)

    def to_batch(W, S, L=None):
        A = torch.from_numpy(norm(W)).to(DEV)
        Sc = torch.from_numpy(norm(S)).to(DEV)
        A = torch.cat([A, Sc.expand(-1, A.shape[1], -1)], dim=-1)
        b = types.SimpleNamespace(action=A, state_cond=Sc)
        if L is not None:
            b.labels = {"identity": torch.from_numpy(L[:, 0]).to(DEV),
                        "kin_feas": torch.from_numpy(L[:, 2]).to(DEV),
                        "afford_feas": torch.from_numpy(L[:, 3]).to(DEV)}
            b.label_mask = {"identity": torch.from_numpy(L[:, 1] > 0.5).to(DEV)}
        return b

    def make_batch():
        Ws, Ss, labs = [], [], []
        def add(W, lab):
            if len(W) == 0:
                return
            Ws.append(W); Ss.append(W[:, :1].copy()); labs.extend([lab] * len(W))
        def addS(W, S, lab):
            Ws.append(W); Ss.append(S); labs.extend([lab] * len(W))
        add(train["robot"][rng.integers(0, len(train["robot"]), 20)], (1, 1, 1, 1))
        add(train["human"][rng.integers(0, len(train["human"]), 44)], (0, 1, 1, 1))
        W = fx.gen_checked(fx.gen_transport, anch, 24, env, rng)
        add(W, (0, 0, 1, 1))
        # STILL-HOLDING positives: freeze real rows (+tiny noise) -> teaches
        # "quasi-static + closed = feasible", the explicit contrast to rotate
        # windows (quasi-static + euler churn + grasp cycles). Without these the
        # critic conflated legitimate holds with the slowed rotation class.
        r0 = np.concatenate([train["robot"], train["human"]])[
            rng.integers(0, len(train["robot"]) + len(train["human"]), 12)][:, :1, :]
        Wst = np.repeat(r0, fx.T, axis=1).copy()
        Wst[..., :3] += rng.normal(0, 0.0008, Wst[..., :3].shape)
        add(Wst, (0, 0, 1, 1))
        # REAL jar: approach/lift positives + rotate negatives (kin auto-label)
        jp = jtr_pos[rng.integers(0, len(jtr_pos), 24)]
        add(jp, (0, 0, 1, 1))
        idx = rng.integers(0, len(jtr_neg), 40)
        jn = jtr_neg[idx]
        for w, ek in zip(jn, ek_tr[idx]):
            add(w[None], (0, 0, float(ek), 0))
        if len(jtr_trn):
            idx = rng.integers(0, len(jtr_trn), 20)
            for w, ek in zip(jtr_trn[idx], ek_trn[idx]):
                add(w[None], (0, 0, float(ek), 0))
        # synthetic kin negatives (mixed spans) + synthetic afford augmentation
        for _ in range(6):
            fam = fx.KIN_FAMS[rng.integers(0, len(fx.KIN_FAMS))]
            base = base_tr_mov if fam == "speed" else base_tr
            W0 = base[rng.integers(0, len(base), 8)]
            u = rng.random()
            s = 0 if u < 0.5 else (16 if u < 0.75 else int(rng.integers(4, 15)))
            Wc = W0.copy(); Wc[:, s:] = fx.corrupt(W0[:, s:], fam, env, rng)
            addS(Wc, W0[:, :1].copy(), (0, 0, 0, 1))
        for _ in range(3):
            fam = fx.AFF_FAMS[rng.integers(0, len(fx.AFF_FAMS))]
            pool = anch_closed if fam in ("pen_spin", "cap_twist") else anch
            W = fx.gen_checked(fx.gen_afford, pool, 8, env, rng, fam=fam)
            add(W, (0, 0, 1, 0))
        W = np.concatenate(Ws); S = np.concatenate(Ss); L = np.array(labs, np.float32)
        return W, S, L

    def sample_k(n):
        ks = torch.randint(0, 21, (n,), device=DEV)
        ks[torch.rand(n, device=DEV) < 0.5] = 0
        return ks

    crit.train()
    for step in range(args.steps):
        W, S, L = make_batch()
        b = to_batch(W, S, L)
        losses = crit.loss_multitask(b, timesteps=sample_k(len(b.action)))
        opt.zero_grad(); losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(crit.parameters(), 5.0)
        opt.step()
        if step % 500 == 0 or step == args.steps - 1:
            print(f"step {step:5d}  " + "  ".join(f"{k}={v.item():.4f}" for k, v in losses.items()))

    # ---------------- gates ----------------
    crit.eval()

    @torch.no_grad()
    def score(W, k=0):
        b = to_batch(W, W[:, :1].copy())
        ts = torch.full((len(W),), k, dtype=torch.long, device=DEV)
        return torch.stack([torch.sigmoid(crit(b, timesteps=ts))
                            for _ in range(args.eval_draws)]).mean(0).cpu().numpy()

    res = {}
    vw = np.concatenate([val["robot"], val["human"]])
    Pref = score(vw); Pjp = score(jva_pos); Pjn = score(jva_neg)
    ek_va = fx.envelope_check(jva_neg, env)
    print(f"\n=== GATES (held-out jar episodes n={N_HELDOUT}) ===")
    print(f"handover feasible ref : P_kin {Pref[:,1].mean():.3f}  P_aff {Pref[:,2].mean():.3f}")
    print(f"jar approach/lift POS : P_kin {Pjp[:,1].mean():.3f}  P_aff {Pjp[:,2].mean():.3f}")
    print(f"jar ROTATE negatives  : P_kin {Pjn[:,1].mean():.3f}  P_aff {Pjn[:,2].mean():.3f} "
          f"(env-pass {ek_va.mean():.0%})")
    a_rot = fx.auroc if hasattr(fx, "auroc") else None
    def auroc(pos, neg):
        x = np.concatenate([pos, neg]); y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
        o = np.argsort(x); r = np.empty(len(x)); r[o] = np.arange(1, len(x) + 1)
        return float((r[y == 1].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))
    res["aff_auroc_heldout"] = auroc(np.concatenate([Pjp[:, 2], Pref[:, 2]]), Pjn[:, 2])
    res["kin_on_rotate_inenv"] = float(Pjn[ek_va, 1].mean()) if ek_va.any() else None
    # GAP-MIDPOINT calibration (robot-observed 08-12: the feasible P_aff pool is
    # SATURATED near 1.0, so its 5% quantile lands at ~0.95 — a razor-thin margin
    # that voted on benign descent dips 0.008 below it and flapped the banner.
    # The real class gap is huge (feasible ~0.95-1.0 vs rotate ~0.00-0.14): put the
    # threshold mid-gap, clipped to a sane band.)
    feasP_aff = np.concatenate([Pjp[:, 2], Pref[:, 2]])
    feasP_kin = np.concatenate([Pjp[:, 1], Pref[:, 1]])
    thr_aff = float(np.clip(0.5 * (np.quantile(feasP_aff, 0.05) + np.quantile(Pjn[:, 2], 0.95)),
                            0.30, 0.90))
    thr_kin = float(np.clip(np.quantile(feasP_kin, 0.05), 0.10, 0.60))
    veto = float(((Pjn[:, 2] < thr_aff) | (Pjn[:, 1] < thr_kin)).mean())
    fpr = float(((Pjp[:, 2] < thr_aff) | (Pjp[:, 1] < thr_kin)).mean())
    res.update(thr_aff=thr_aff, thr_kin=thr_kin, rotate_veto=veto, approach_fpr=fpr)
    print(f"AFF AUROC (feas-vs-rotate, heldout): {res['aff_auroc_heldout']:.3f}")
    print(f"thresholds: aff {thr_aff:.3f} kin {thr_kin:.3f} -> "
          f"rotate veto {veto:.1%} | approach false-veto {fpr:.1%}")
    if len(jva_trn):
        Pt = score(jva_trn)
        tveto = float(((Pt[:, 2] < thr_aff) | (Pt[:, 1] < thr_kin)).mean())
        res["transition_veto"] = tveto
        print(f"TRANSITION (onset) windows: P_aff {Pt[:,2].mean():.3f} -> veto {tveto:.1%}  "
              f"(the deploy first-detection shape — the latency metric)")
    # synthetic spot check (no regression)
    for fam in ("whipsaw", "teleport"):
        W0 = vw[rng.integers(0, len(vw), 300)]
        a = auroc(Pref[:, 1], score(fx.corrupt(W0, fam, env, rng))[:, 1])
        res[f"kin_{fam}"] = a
        print(f"synthetic {fam} kin-AUROC: {a:.3f}")

    OUT.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": crit.state_dict(),
                "stats": [st[0].tolist(), st[1].tolist()],
                "thr_aff": thr_aff, "thr_kin": thr_kin},
               OUT / f"{args.run_name}.pth")
    (OUT / f"{args.run_name}_gates.json").write_text(json.dumps(res, indent=1))
    print(f"saved {OUT}/{args.run_name}.pth  GATE {'PASS' if res['aff_auroc_heldout'] > 0.9 and fpr < 0.15 else 'REVIEW'}")


if __name__ == "__main__":
    main()
