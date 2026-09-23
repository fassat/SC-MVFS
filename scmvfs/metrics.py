from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import normalized_mutual_info_score


def _squared_distances(x: np.ndarray, centers: np.ndarray) -> np.ndarray:
    delta = x[:, None, :] - centers[None, :, :]
    return np.sum(delta * delta, axis=2)


def _kmeans_plus_plus(
    x: np.ndarray, n_clusters: int, rng: np.random.Generator
) -> np.ndarray:
    centers = np.empty((n_clusters, x.shape[1]), dtype=np.float64)
    centers[0] = x[int(rng.integers(x.shape[0]))]
    closest = _squared_distances(x, centers[:1])[:, 0]
    for index in range(1, n_clusters):
        total = float(closest.sum())
        choice = (
            int(rng.integers(x.shape[0]))
            if total <= 0.0 or not np.isfinite(total)
            else int(rng.choice(x.shape[0], p=closest / total))
        )
        centers[index] = x[choice]
        closest = np.minimum(
            closest, _squared_distances(x, centers[index : index + 1])[:, 0]
        )
    return centers


def fit_kmeans(
    x: np.ndarray,
    n_clusters: int,
    seed: int,
    n_init: int = 20,
    max_iter: int = 300,
    tol: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64, order="C")
    master = np.random.default_rng(seed)
    best_inertia = float("inf")
    best_centers = None
    best_labels = None
    for _ in range(n_init):
        rng = np.random.default_rng(int(master.integers(0, 2**32 - 1)))
        centers = _kmeans_plus_plus(x, n_clusters, rng)
        previous = float("inf")
        for _ in range(max_iter):
            distances = _squared_distances(x, centers)
            labels = np.argmin(distances, axis=1)
            inertia = float(distances[np.arange(x.shape[0]), labels].sum())
            updated = centers.copy()
            assigned = distances[np.arange(x.shape[0]), labels]
            for cluster in range(n_clusters):
                members = x[labels == cluster]
                updated[cluster] = (
                    members.mean(axis=0)
                    if members.size
                    else x[int(np.argmax(assigned))]
                )
            shift = float(np.sum((updated - centers) ** 2))
            centers = updated
            if shift <= tol or abs(previous - inertia) <= tol:
                break
            previous = inertia
        distances = _squared_distances(x, centers)
        labels = np.argmin(distances, axis=1)
        inertia = float(distances[np.arange(x.shape[0]), labels].sum())
        if inertia < best_inertia:
            best_inertia = inertia
            best_centers = centers.copy()
            best_labels = labels.copy()
    if best_centers is None or best_labels is None:
        raise RuntimeError("K-means did not produce a solution.")
    return best_centers, best_labels


def clustering_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.int64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.int64).reshape(-1)
    size = max(int(y_true.max()), int(y_pred.max())) + 1
    contingency = np.zeros((size, size), dtype=np.int64)
    np.add.at(contingency, (y_pred, y_true), 1)
    rows, columns = linear_sum_assignment(contingency.max() - contingency)
    return float(contingency[rows, columns].sum() / y_true.size)


def evaluate_kmeans(
    features: np.ndarray,
    labels: np.ndarray,
    n_clusters: int,
    seed: int,
    n_init: int = 20,
) -> dict[str, float]:
    _, predictions = fit_kmeans(features, n_clusters, seed, n_init)
    return {
        "acc": clustering_accuracy(labels, predictions),
        "nmi": float(normalized_mutual_info_score(labels, predictions)),
    }
