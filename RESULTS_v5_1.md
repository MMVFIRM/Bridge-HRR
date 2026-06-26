from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class KGDataset:
    """Integer-id knowledge graph split bundle."""

    train: np.ndarray
    valid: np.ndarray
    test: np.ndarray
    entity_to_id: dict[str, int]
    relation_to_id: dict[str, int]
    id_to_entity: list[str]
    id_to_relation: list[str]

    @property
    def n_entities(self) -> int:
        return len(self.id_to_entity)

    @property
    def n_relations(self) -> int:
        return len(self.id_to_relation)

    @property
    def full(self) -> np.ndarray:
        return np.vstack([self.train, self.valid, self.test]).astype(np.int64)


def _read_raw_triples(path: Path) -> list[tuple[str, str, str]]:
    triples: list[tuple[str, str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 3:
                parts = line.split()
            if len(parts) != 3:
                raise ValueError(f"{path}:{line_no}: expected 3 columns, got {len(parts)}")
            triples.append((parts[0], parts[1], parts[2]))
    return triples


def _encode_triples(
    raw: Iterable[tuple[str, str, str]],
    entity_to_id: dict[str, int],
    relation_to_id: dict[str, int],
    id_to_entity: list[str],
    id_to_relation: list[str],
) -> np.ndarray:
    out: list[tuple[int, int, int]] = []
    for h, r, t in raw:
        if h not in entity_to_id:
            entity_to_id[h] = len(id_to_entity)
            id_to_entity.append(h)
        if t not in entity_to_id:
            entity_to_id[t] = len(id_to_entity)
            id_to_entity.append(t)
        if r not in relation_to_id:
            relation_to_id[r] = len(id_to_relation)
            id_to_relation.append(r)
        out.append((entity_to_id[h], relation_to_id[r], entity_to_id[t]))
    return np.asarray(out, dtype=np.int64)


def load_fb15k237(data_dir: str | Path) -> KGDataset:
    """Load an FB15k-237 style directory containing train/valid/test txt files.

    Expected files: train.txt, valid.txt, test.txt. The loader also accepts
    valid.txt/valid.tsv and train.tsv/test.tsv variants when present.
    """
    data = Path(data_dir)
    if not data.exists():
        raise FileNotFoundError(f"data_dir does not exist: {data}")

    def pick(*names: str) -> Path:
        for name in names:
            p = data / name
            if p.exists():
                return p
        raise FileNotFoundError(f"None of these files found in {data}: {names}")

    train_raw = _read_raw_triples(pick("train.txt", "train.tsv"))
    valid_raw = _read_raw_triples(pick("valid.txt", "valid.tsv", "dev.txt", "dev.tsv"))
    test_raw = _read_raw_triples(pick("test.txt", "test.tsv"))

    e2id: dict[str, int] = {}
    r2id: dict[str, int] = {}
    id2e: list[str] = []
    id2r: list[str] = []
    train = _encode_triples(train_raw, e2id, r2id, id2e, id2r)
    valid = _encode_triples(valid_raw, e2id, r2id, id2e, id2r)
    test = _encode_triples(test_raw, e2id, r2id, id2e, id2r)
    return KGDataset(train=train, valid=valid, test=test, entity_to_id=e2id, relation_to_id=r2id, id_to_entity=id2e, id_to_relation=id2r)
