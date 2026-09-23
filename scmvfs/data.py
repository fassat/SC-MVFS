from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import StratifiedShuffleSplit


@dataclass(frozen=True)
class MultiViewDataset:
    name: str
    views: list[np.ndarray]
    labels: np.ndarray

    @property
    def n_samples(self) -> int:
        return int(self.labels.size)

    @property
    def n_views(self) -> int:
        return len(self.views)

    @property
    def n_clusters(self) -> int:
        return int(np.unique(self.labels).size)


def load_multiview_mat(path: str | Path) -> MultiViewDataset:
    path = Path(path)
    data = loadmat(path, squeeze_me=False, struct_as_record=False)
    x_key = next((key for key in ("X", "x", "views", "data") if key in data), None)
    y_key = next((key for key in ("y", "Y", "labels", "label", "gt") if key in data), None)
    if x_key is None or y_key is None:
        keys = sorted(key for key in data if not key.startswith("__"))
        raise KeyError(f"Expected views and labels in {path}; found {keys}.")

    labels = np.asarray(data[y_key]).reshape(-1)
    _, labels = np.unique(labels, return_inverse=True)
    labels = labels.astype(np.int64, copy=False)
    raw_views = data[x_key]
    if not (isinstance(raw_views, np.ndarray) and raw_views.dtype == object):
        raw_views = np.asarray([raw_views], dtype=object)

    views: list[np.ndarray] = []
    for index, item in enumerate(raw_views.ravel(order="F")):
        array = np.asarray(item)
        if array.ndim != 2:
            raise ValueError(f"View {index} is not a matrix.")
        if array.shape[0] == labels.size:
            view = array
        elif array.shape[1] == labels.size:
            view = array.T
        else:
            raise ValueError(f"View {index} is not aligned with the labels.")
        views.append(np.asarray(view, dtype=np.float64, order="C"))
    if len(views) < 2:
        raise ValueError("SC-MVFS requires at least two views.")
    return MultiViewDataset(path.stem, views, labels)


def development_split(
    labels: np.ndarray, ratio: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    if not 0.0 < ratio < 1.0:
        raise ValueError("development ratio must lie in (0, 1).")
    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=ratio, random_state=seed
    )
    evaluation, development = next(splitter.split(np.zeros(labels.size), labels))
    return development.astype(np.int64), evaluation.astype(np.int64)


def balanced_missing_mask(
    n_samples: int,
    n_views: int,
    missing_ratio: float,
    seed: int,
    max_attempts: int = 200,
) -> np.ndarray:
    if n_views < 2 or not 0.0 <= missing_ratio < 1.0:
        raise ValueError("Invalid number of views or missing ratio.")
    maximum = 1.0 - 1.0 / n_views
    if missing_ratio > maximum + 1e-12:
        raise ValueError(
            f"missing_ratio={missing_ratio} is infeasible for {n_views} views."
        )

    target = int(np.floor(missing_ratio * n_samples + 0.5))
    targets = np.full(n_views, target, dtype=np.int64)
    capacity = n_samples * (n_views - 1)
    if int(targets.sum()) > capacity:
        total = min(
            capacity,
            int(np.floor(missing_ratio * n_samples * n_views + 0.5)),
        )
        base, remainder = divmod(total, n_views)
        targets.fill(base)
        order = np.random.default_rng(seed).permutation(n_views)
        targets[order[:remainder]] += 1

    master = np.random.default_rng(seed)
    for _ in range(max_attempts):
        mask = np.ones((n_samples, n_views), dtype=bool)
        rng = np.random.default_rng(int(master.integers(0, 2**32 - 1)))
        success = True
        for view in rng.permutation(n_views):
            candidates = np.flatnonzero(mask.sum(axis=1) > 1)
            count = int(targets[view])
            if candidates.size < count:
                success = False
                break
            mask[rng.choice(candidates, size=count, replace=False), view] = False
        if success and np.all(mask.sum(axis=0) == n_samples - targets):
            return mask
    raise RuntimeError("Could not construct a balanced missing-view mask.")


def select_features(
    completed_views: Sequence[np.ndarray], selected_indices: np.ndarray
) -> np.ndarray:
    return np.concatenate(completed_views, axis=1)[:, selected_indices]


def selected_counts(
    selected_indices: np.ndarray, dimensions: Sequence[int]
) -> list[int]:
    offsets = np.cumsum([0, *dimensions])
    return [
        int(np.sum((selected_indices >= offsets[v]) & (selected_indices < offsets[v + 1])))
        for v in range(len(dimensions))
    ]


def minmax_normalize(
    views: Sequence[np.ndarray], observed_mask: np.ndarray
) -> list[np.ndarray]:
    normalized = []
    for view_index, view in enumerate(views):
        array = np.asarray(view, dtype=np.float64)
        observed = array[observed_mask[:, view_index]]
        if observed.shape[0] == 0:
            raise ValueError(f"View {view_index} has no observed samples.")
        minimum = observed.min(axis=0)
        scale = observed.max(axis=0) - minimum
        scale[scale < 1e-12] = 1.0
        normalized.append((array - minimum) / scale)
    return normalized
