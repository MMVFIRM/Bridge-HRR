"""CLI: train RotatE+path codes, and run the held-out composition degree sweep.

  python -m bridge_hrr_v5.cli train --data-dir DATA --out CKPT.npz \
      --dim 128 --epochs 40 --warmup-epochs 24            # atomic warmup then +path
  python -m bridge_hrr_v5.cli eval  --data-dir DATA --checkpoint CKPT.npz \
      [--pure-checkpoint PURE.npz] --n 2500                 # degree-stratified sweep
"""
from __future__ import annotations
import argparse, json, time
import numpy as np

from .kg_io import load_fb15k237
from .rotate_path import RotatePathModel, TrainConfig, sample_train_paths
from . import compose_eval as ce


def cmd_train(a):
    kg = load_fb15k237(a.data_dir)
    cfg = TrainConfig(dim=a.dim, lr=a.lr, batch_size=a.batch_size,
                      negatives=a.negatives, seed=a.seed, use_path_loss=True)
    m = RotatePathModel(kg.n_entities, kg.n_relations, cfg)
    paths = sample_train_paths(kg.train, cfg.n_paths, seed=cfg.seed)
    t0 = time.time()
    # phase 1: atomic-only warmup (path loss off) so geometry is grounded first
    m.cfg.use_path_loss = False
    m.train_epochs(kg.train, paths, a.warmup_epochs)
    if a.pure_out:
        m.save(a.pure_out)
        print(f"saved atomic-only control -> {a.pure_out}")
    # phase 2: joint atomic + path-consistency
    m.cfg.use_path_loss = True
    m.train_epochs(kg.train, paths, a.epochs - a.warmup_epochs)
    m.save(a.out)
    print(f"trained dim={a.dim} epochs={a.epochs} (warmup {a.warmup_epochs}) "
          f"in {time.time()-t0:.0f}s -> {a.out}")


def cmd_eval(a):
    kg = load_fb15k237(a.data_dir)
    at = ce.build_alltrue(kg.train, kg.valid, kg.test)
    # build the full (unfiltered) set, then split by the strict train-closure gate
    Q, stats = ce.build_heldout_twohop(kg.train, kg.valid, kg.test, at,
                                       n=a.n, seed=a.seed, strict_composite=False)
    ans = ce.answer_sets(Q, at)
    model = RotatePathModel.load(a.checkpoint)
    pure = RotatePathModel.load(a.pure_checkpoint) if a.pure_checkpoint else None
    report = ce.composition_report(model, Q, ans, kg.train, pure_model=pure)
    # degree sweep on the STRICT subset only
    train_out = ce.build_train_out(kg.train)
    strict = ~ce.leak_mask(Q, train_out)
    Qs = Q[strict]; As = [ans[i] for i in range(len(Q)) if strict[i]]
    report["degree_sweep_strict"] = ce.degree_sweep(model, Qs, As, kg.train, pure_model=pure)
    out = {"n_sampled": len(Q), "report": report}
    print(json.dumps(out, indent=2))
    if a.out_json:
        json.dump(out, open(a.out_json, "w"), indent=2)


def main():
    p = argparse.ArgumentParser(prog="bridge_hrr_v5")
    sub = p.add_subparsers(required=True)
    t = sub.add_parser("train"); t.set_defaults(fn=cmd_train)
    t.add_argument("--data-dir", required=True); t.add_argument("--out", required=True)
    t.add_argument("--pure-out", default=None, help="save atomic-only control checkpoint")
    t.add_argument("--dim", type=int, default=128); t.add_argument("--epochs", type=int, default=40)
    t.add_argument("--warmup-epochs", type=int, default=24)
    t.add_argument("--lr", type=float, default=5e-3); t.add_argument("--batch-size", type=int, default=2048)
    t.add_argument("--negatives", type=int, default=16); t.add_argument("--seed", type=int, default=0)
    e = sub.add_parser("eval"); e.set_defaults(fn=cmd_eval)
    e.add_argument("--data-dir", required=True); e.add_argument("--checkpoint", required=True)
    e.add_argument("--pure-checkpoint", default=None); e.add_argument("--n", type=int, default=2500)
    e.add_argument("--seed", type=int, default=5); e.add_argument("--out-json", default=None)
    args = p.parse_args(); args.fn(args)


if __name__ == "__main__":
    main()
