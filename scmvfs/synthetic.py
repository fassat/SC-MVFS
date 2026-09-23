from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SyntheticData:
    views: list[np.ndarray]
    labels: np.ndarray
    feature_types: np.ndarray
    shortcut_blocks: list[np.ndarray]


def _ridge_r2(
    source: np.ndarray,
    target: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    alpha: float = 1.0,
) -> float:
    x_train, x_test = source[train], source[test]
    y_train, y_test = target[train], target[test]
    x_mean = x_train.mean(axis=0)
    x_scale = np.maximum(x_train.std(axis=0, ddof=1), 1e-8)
    y_mean = y_train.mean(axis=0)
    y_scale = np.maximum(y_train.std(axis=0, ddof=1), 1e-8)
    x_train = (x_train - x_mean) / x_scale
    x_test = (x_test - x_mean) / x_scale
    y_train = (y_train - y_mean) / y_scale
    y_test = (y_test - y_mean) / y_scale
    coefficients = np.linalg.solve(
        x_train.T @ x_train + alpha * np.eye(x_train.shape[1]),
        x_train.T @ y_train,
    )
    prediction = x_test @ coefficients
    residual = np.sum((y_test - prediction) ** 2)
    total = np.sum((y_test - y_test.mean(axis=0)) ** 2)
    return float(1.0 - residual / max(float(total), 1e-12))


def shortcut_diagnostics(blocks: list[np.ndarray], seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed + 73939)
    order = rng.permutation(blocks[0].shape[0])
    split = int(round(0.6 * order.size))
    train, test = order[:split], order[split:]
    correlations, all_r2, subset_r2 = [], [], []
    for source_index, source in enumerate(blocks):
        source_z = (source - source.mean(0)) / np.maximum(source.std(0, ddof=1), 1e-8)
        for target_index, target in enumerate(blocks):
            if source_index == target_index:
                continue
            target_z = (target - target.mean(0)) / np.maximum(target.std(0, ddof=1), 1e-8)
            correlations.append(np.abs(source_z.T @ target_z / (source.shape[0] - 1)).ravel())
            all_r2.append(_ridge_r2(source, target, train, test))
            draws = []
            for _ in range(8):
                chosen = rng.choice(source.shape[1], size=8, replace=False)
                draws.append(_ridge_r2(source[:, chosen], target, train, test))
            subset_r2.append(float(np.mean(draws)))
    return {
        "individual_correlation": float(np.mean(np.concatenate(correlations))),
        "all_r2": float(np.mean(all_r2)),
        "subset_r2": float(np.mean(subset_r2)),
    }


def generate_shortcut_data(eta: float, seed: int) -> SyntheticData:
    rng = np.random.default_rng(seed)
    sample_count, view_count, cluster_count, latent_dim = 600, 3, 5, 6
    labels = np.arange(sample_count, dtype=np.int64) % cluster_count
    rng.shuffle(labels)
    centers = rng.normal(size=(cluster_count, latent_dim))
    centers -= centers.mean(axis=0)
    centers = 2.5 * centers / np.maximum(np.linalg.norm(centers, axis=1, keepdims=True), 1e-12)
    sample_scale = np.exp(0.3 * rng.normal(size=sample_count) - 0.045)
    shared_residual = rng.normal(size=(sample_count, latent_dim))
    rng.normal(size=(view_count, sample_count, latent_dim))
    shortcut_latent = rng.normal(size=(sample_count, 48))
    views, feature_types, shortcut_blocks = [], [], []
    for view_index in range(view_count):
        loadings = 0.2 * rng.normal(size=(8, latent_dim))
        for feature_index in range(8):
            loadings[feature_index, (feature_index + 2 * view_index) % latent_dim] += 1.0
        loadings /= np.maximum(np.linalg.norm(loadings, axis=1, keepdims=True), 1e-12)
        latent = centers[labels] + 0.55 * sample_scale[:, None] * shared_residual
        clean = latent @ loadings.T
        clean_scale = clean.std(axis=0, ddof=1)
        noise_scale = clean_scale / np.sqrt(2.0)
        primary = clean + sample_scale[:, None] * noise_scale * rng.normal(size=clean.shape)
        redundant_sources = np.arange(8, dtype=np.int64)
        rng.shuffle(redundant_sources)
        redundant = primary[:, redundant_sources] + (
            0.15
            * noise_scale[redundant_sources]
            * rng.normal(size=primary.shape)
        )
        beta = float(np.mean(np.maximum(clean_scale, 1e-6)))
        nuisance = beta * rng.normal(size=(sample_count, 32))
        shortcut_loadings = rng.normal(size=(96, 48))
        shortcut_loadings /= np.maximum(
            np.linalg.norm(shortcut_loadings, axis=1, keepdims=True), 1e-12
        )
        shared_shortcut = shortcut_latent @ shortcut_loadings.T
        shared_shortcut /= np.maximum(shared_shortcut.std(axis=0, ddof=1), 1e-12)
        shortcut = beta * (
            eta * shared_shortcut
            + np.sqrt(max(1.0 - eta**2, 0.0)) * rng.normal(size=shared_shortcut.shape)
        )
        block = np.concatenate([primary, redundant, shortcut, nuisance], axis=1)
        kinds = np.asarray(
            ["primary"] * 8 + ["redundant"] * 8 + ["shortcut"] * 96 + ["noise"] * 32,
            dtype=object,
        )
        permutation = rng.permutation(block.shape[1])
        views.append(np.asarray(block[:, permutation], dtype=np.float64, order="C"))
        feature_types.extend(kinds[permutation].tolist())
        shortcut_blocks.append(shortcut)
    return SyntheticData(
        views=views,
        labels=labels,
        feature_types=np.asarray(feature_types, dtype=object),
        shortcut_blocks=shortcut_blocks,
    )
