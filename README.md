"""RotatE + path-consistency trainer (pure NumPy, hand-derived gradients).

Two objectives, both phase-space RotatE scoring  s = mean_d cos(theta_q - theta_cand):

  atomic:  query phase = theta_h + theta_r           target tail t   (single edge)
  path:    query phase = theta_h + theta_r1 + theta_r2  target tail t (observed 2-hop)

Sampled-softmax cross-entropy with in-batch random negatives. The path objective
is what makes held-out two-hop composition generalize; the atomic objective keeps
the entity/relation geometry grounded. NumPy is used so the trainer runs anywhere
(no torch dependency); on a GPU box you would port this to torch unchanged in math.

Held-out cleanliness: path samples are drawn ONLY from observed train 2-hops, so a
held-out bridge edge (m, r2, t) in valid/test is never a training target. Evaluating
composition on held-out bridge edges is therefore a genuine generalization test.
"""
from __future__ import annotations

import os
from collections import defaultdict
from dataclasses import dataclass, asdict

import numpy as np


@dataclass
class TrainConfig:
    dim: int = 64
    lr: float = 5e-3
    batch_size: int = 2048
    negatives: int = 16
    score_scale: float = 20.0
    seed: int = 0
    use_path_loss: bool = True
    n_paths: int = 400_000  # number of observed 2-hop paths to sample once


def sample_train_paths(train: np.ndarray, n_paths: int, seed: int = 0) -> np.ndarray:
    """Sample observed 2-hop TRAIN paths (h, r1, r2, t): h-r1->m, m-r2->t, both in train."""
    rng = np.random.default_rng(seed)
    out_by_head: dict[int, np.ndarray] = {}
    tmp: dict[int, list] = defaultdict(list)
    for h, r, t in train.tolist():
        tmp[int(h)].append((int(r), int(t)))
    for k, v in tmp.items():
        out_by_head[k] = np.array(v, dtype=np.int64)
    src = train[rng.choice(len(train), n_paths, replace=True)]
    paths = []
    for h, r1, m in src.tolist():
        e = out_by_head.get(int(m))
        if e is None:
            continue
        r2, t = e[rng.integers(len(e))]
        paths.append((int(h), int(r1), int(r2), int(t)))
    return np.array(paths, dtype=np.int64)


class RotatePathModel:
    def __init__(self, n_entities: int, n_relations: int, cfg: TrainConfig):
        self.N, self.R, self.cfg = n_entities, n_relations, cfg
        rng = np.random.default_rng(cfg.seed)
        d = cfg.dim
        self.ep = rng.uniform(-np.pi, np.pi, (n_entities, d)).astype(np.float32)
        self.rp = rng.uniform(-np.pi, np.pi, (n_relations, d)).astype(np.float32)
        self._mE = np.zeros_like(self.ep); self._vE = np.zeros_like(self.ep)
        self._mR = np.zeros_like(self.rp); self._vR = np.zeros_like(self.rp)
        self._g = 0

    # ---- one sampled-softmax CE step; query_rels is a list of relation columns added to head ----
    def _step(self, heads, rel_cols, tails):
        cfg = self.cfg; d = cfg.dim; B = len(heads)
        a = self.ep[heads].copy()
        for rc in rel_cols:
            a = a + self.rp[rc]
        cid = np.concatenate([tails[:, None],
                              np.random.randint(0, self.N, (B, cfg.negatives))], axis=1)
        diff = a[:, None, :] - self.ep[cid]
        sind = np.sin(diff)
        s = np.cos(diff).mean(2) * cfg.score_scale
        s -= s.max(1, keepdims=True)
        P = np.exp(s); P /= P.sum(1, keepdims=True)
        onehot = np.zeros_like(P); onehot[:, 0] = 1.0
        g = cfg.score_scale * (P - onehot) / B                      # dL/d(meancos_c)
        grad_a = -(1.0 / d) * np.einsum('bc,bcd->bd', g, sind)      # (B,d)
        grad_ec = (1.0 / d) * g[:, :, None] * sind                  # (B,C,d)
        gE = np.zeros_like(self.ep); gR = np.zeros_like(self.rp)
        np.add.at(gE, heads, grad_a)
        for rc in rel_cols:
            np.add.at(gR, rc, grad_a)
        np.add.at(gE, cid.reshape(-1), grad_ec.reshape(-1, d))
        self._adam(gE, gR)

    def _adam(self, gE, gR, b1=0.9, b2=0.999, eps=1e-8):
        self._g += 1; lr = self.cfg.lr
        for p, m, v, gp in [(self.ep, self._mE, self._vE, gE),
                            (self.rp, self._mR, self._vR, gR)]:
            m *= b1; m += (1 - b1) * gp
            v *= b2; v += (1 - b2) * gp * gp
            p -= lr * (m / (1 - b1 ** self._g)) / (np.sqrt(v / (1 - b2 ** self._g)) + eps)

    def train_epochs(self, train: np.ndarray, paths: np.ndarray, n_epochs: int):
        bs = self.cfg.batch_size
        for _ in range(n_epochs):
            pa = np.random.permutation(len(train))
            pp = np.random.permutation(len(paths)) if (self.cfg.use_path_loss and len(paths)) else None
            for s0 in range(0, len(train), bs):
                b = train[pa[s0:s0 + bs]]
                self._step(b[:, 0], [b[:, 1]], b[:, 2])           # atomic
                if pp is not None:
                    j = s0 % len(paths)
                    p = paths[pp[j:j + bs]]
                    self._step(p[:, 0], [p[:, 1], p[:, 2]], p[:, 3])  # path

    def composed_scores(self, heads, r1, r2) -> np.ndarray:
        """Pure-geometry two-hop prediction: scores over all entities for theta_h+theta_r1+theta_r2."""
        q = self.ep[heads] + self.rp[r1] + self.rp[r2]
        return (np.cos(q) @ np.cos(self.ep).T + np.sin(q) @ np.sin(self.ep).T) / self.cfg.dim

    def atomic_scores(self, heads, rels) -> np.ndarray:
        q = self.ep[heads] + self.rp[rels]
        return (np.cos(q) @ np.cos(self.ep).T + np.sin(q) @ np.sin(self.ep).T) / self.cfg.dim

    # ---- persistence ----
    def save(self, path: str):
        np.savez(path, ep=self.ep, rp=self.rp, mE=self._mE, vE=self._vE,
                 mR=self._mR, vR=self._vR, g=self._g, cfg=np.array(list(asdict(self.cfg).items()), dtype=object))

    @classmethod
    def load(cls, path: str, cfg: TrainConfig | None = None) -> "RotatePathModel":
        z = np.load(path, allow_pickle=True)
        c = cfg or TrainConfig(dim=z['ep'].shape[1])
        m = cls(z['ep'].shape[0], z['rp'].shape[0], c)
        m.ep = z['ep']; m.rp = z['rp']
        for k, attr in [('mE', '_mE'), ('vE', '_vE'), ('mR', '_mR'), ('vR', '_vR')]:
            if k in z: setattr(m, attr, z[k])
        m._g = int(z['g']) if 'g' in z else 0
        return m
