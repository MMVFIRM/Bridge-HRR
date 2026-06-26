# Bridge-HRR v5.1 — Compositional geometry for two-hop KG reasoning (strict gate)

Two-hop composition on FB15k-237 is recovered not by holographic memory and not by
attention readout, but by geometry trained to compose: RotatE atomic scoring + a
path-consistency objective (theta_h + theta_r1 + theta_r2 -> theta_t over observed
2-hops). The composed key is evaluated by nearest-neighbour over the entity codebook.

**v5.1 corrects v5.** A review caught that the held-out builder protected the bridge
edge but not the composed target, which can be reachable through an alternate train
middle. With the strict train-closure gate (now default), the headline shrinks a lot:
on genuinely-unseen composites, path-consistency gives MRR 0.105 vs a 0.082 popularity
floor and 0.022 for atomic-only RotatE. Real and ablation-clean, but modest — about
half the v5 number was leakage. See RESULTS_v5_1.md.

The honest one-liner: **composition is not a free lunch of atomic geometry — it must be
paid for in the objective — and even then, the clean-generalization payoff here is small.**

## Layout
- `bridge_hrr_v5/rotate_path.py` — NumPy RotatE + path-consistency trainer (torch-free).
- `bridge_hrr_v5/compose_eval.py` — held-out builder, strict train-closure gate,
  ALL/STRICT/LEAKY composition report, degree sweep, filtered ranking.
- `bridge_hrr_v5/manifest.py` — auditable checkpoint/result provenance (sha256 + config).
- `bridge_hrr_v5/cli.py` — `train` and `eval` (strict-gated) commands.
- `bridge_hrr_v5/checkpoints/` — d=64 demo checkpoints + `.manifest.json` provenance.
- `result_strict.json` — the strict-gated result artifact.
- `tests/` — smoke + scientific tests (strict-gate leak rejection, filtered-MRR
  correctness, deterministic ideal-codes composition control). No FB15k needed.

## Open / not claimed
Multi-seed robustness, higher dimension, deeper/branching composition, and unseen
relation pairs are untested. d=64, one seed.

## Data
FB15k-237 train/valid/test.txt from
https://raw.githubusercontent.com/villmow/datasets_knowledge_embedding/master/FB15k-237/
