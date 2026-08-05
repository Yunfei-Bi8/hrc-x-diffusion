# TUM UR10e Unified Handover — ONE policy for both directions (2026-08-05)

**Best handover policy to date, ROBOT-VERIFIED**: a SINGLE X-Diffusion policy that
decides at runtime whether to do a **robot-first** handover (robot grasps the bowl and
hands it to the human) or a **human-first** handover (human grasps first and hands it to
the robot), chosen purely from the live human-hand condition — no explicit mode switch,
no timing feature. Run: `policy_handover_uni_v3` (wandb `x-diffusion-hrc /
combined-handover-v1.2-side`). Supersedes the two per-mode policies
(`policy_handover_tfix`, `policy_handover_hf_tfix2`), which remain as rollbacks.

Read `README_TUM_HANDOVER.md` (robot-first) and `README_TUM_HUMAN_FIRST` history first;
this file records only the unification layer.

## THE deployment command (canonical)

```bash
cd ~/Tum_lsy_ur10e_pipeline/ur10_clearpath/Yunfei/crisp_gym
pixi run -e jazzy-lerobot python /home/admin_025/mt_pi_codebase/mt_pi/scripts/deploy_xdiffusion_ur10e.py \
    --run-dir /home/admin_025/X-Diffusion-Data/_private/runs/policy_handover_uni_v3 \
    --ckpt latest --no-safety-observer \
    --hamer-interferer --interferer-hand auto \
    --open-horizon 8 --open-settle-cycles 2 --open-settle-dpx 12 \
    --no-home --goto-start
```

- `--interferer-hand auto`: daemon detects handedness by box side (`--is-right -1`).
- Entry-edge SIDE lock (deploy): the hand identity is decided at FIRST detection —
  cx `< --side-split (1000)` → initiator/left, else receiver/right — and held until 20
  absent cycles. The initiator's territory runs from the left edge THROUGH the bowl
  (cx up to ~950); the receiver only ever enters from the right edge (1150+). A midline
  (640) rule mislocked the human-first initiator (already at the bowl after the ~15-20s
  goto-start), and a short unlock relocked mid-offer — both robot-verified failures.
- Optional dampers for out-of-protocol hand timing (a mid-task withdrawal is undefined
  by data): `--draw-consistency --anchor-jump-mm 40`.

## How one policy selects the mode (no timing feature needed)

The 12-dim condition at the decision point already carries the mode:

| condition at the unified start | meaning | branch |
|---|---|---|
| sentinel (no hand) | nobody there | robot-first: descend & grasp |
| initiator LEFT hand visible at the bowl (side=0) | human is grasping first | human-first: wait & receive |
| receiver from the right edge (side=1) | mid-task rf receiver | keep going (rf) |

Branch matrix at the unified start (0.924, 0.146, 0.654) confirms directional
selection: SENTINEL → dy +9.4 (toward the rf band), LEFT_AT_BOWL → dy −10.7 (toward the
hf receive side). Per-mode open-loop 6.3 mm (rf) / 6.8 mm (hf).

## Unified dataset `handover_bowl_uni` (existing dirs untouched)

- robot 12 = rf-tfix 6 + hf-tfix 6; human 136 = rf-view 70 (demo00000-69) + hf-view 66
  (demo00100-165). Same 29-07 two-human scene read under BOTH role assignments.
- Three build-time conflict fixes (`make_uni_bridges.py` + inline): (1) hf prepend
  condition SENTINEL→first-real-initiator-row backfill (unified sentinel must mean
  GO-robot-first, not hold); (2) hf-view human lead-trim to first-visible; (3) glide
  BRIDGE prefixes from the unified start to each demo's own start (rf with sentinel, hf
  with the initiator condition) — teaches the mode DECISION at the exact deploy start,
  since the rf/hf robot y-bands are disjoint (0.17-0.28 vs 0.03-0.13).
- **5th condition dim `side`** (0=initiator/left, 1=receiver/right, 0.5=absent): the fix
  that made robot-first work. Without it the hf in-air receive (z 0.49-0.52) aliased
  with rf descent/carry — dist distributions are identical across modes (259 vs 249),
  cx separates (1115 vs 885) but loses to the z+phase prior. Appended to all 148 h5s
  from view ground-truth (no re-export); loader dropout sentinel is width-aware; deploy
  supplies it via the entry-edge lock.

## Training

Winning recipe + `neg_dropout_scope: closed` (still-dropout would re-teach
sentinel→hold, which contradicts the unified sentinel→go-robot-first). Single classifier
(7-dim slice, side excluded). Configs: `configs/classifier_handover_uni.yaml`,
`configs/policy_handover_uni_v3.yaml`. v1 (no bridge) hedged at the gap start; v2
(bridge) branched but rf aliased hf; v3 (side bit) fixed it and IMPROVED hf
(HF-native wait 11.6→4.8 mm). v1/v2 runs retained.

## Lineage of the per-mode policies this unifies

- robot-first: v1 (ignored condition) → v2 neg-dropout → tfix (time-base resample +
  pipelined deploy). See `README_TUM_HANDOVER.md`.
- human-first: reversed roles on the same data, v1.2 → hf_tfix2. Reactive
  wait-then-receive.
- Both were ~80% robot-verified before unification; the unified v3 folds them into one.

## Related infra fixes shipped alongside

- **Timestamp-uniform export** (`--resample-hz 15`): the 07-29 recordings stalled 27-32%
  of frame transitions and ran at a true ~10 Hz despite fps=15 metadata → 11-18% teleport
  steps in the old exports; resampling on the true camera_01 header clock removed them.
- **Pipelined deploy** (~2.3× faster cycles): batched-draw worker, next-request pre-send,
  absolute-deadline stepping.
- **CRISP gain guard**: every arm-driver restart silently zeroes cartesian task-space
  damping (yaml `d=-1` auto-critical sentinel resolves to 0 at controller load; only a
  SetParameters push applies it). Auto-restored now at three entry points
  (startup_robot.py, master_launch.sh, deploy gain guard). Measured cost of d=0: 68 mm
  following error, ~1.2 s lag.
