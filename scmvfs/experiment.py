from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from .data import (
    balanced_missing_mask,
    development_split,
    load_multiview_mat,
    minmax_normalize,
    select_features,
)
from .metrics import evaluate_kmeans
from .model import SCMVFS, SCMVFSConfig


def default_config(
    *,
    seed: int,
    feature_ratio: float,
    group_lambda: float,
    device: str,
    dtype: str,
    **overrides,
) -> SCMVFSConfig:
    config = SCMVFSConfig(
        hidden_dim=128,
        latent_dim=32,
        warmup_epochs=5,
        em_rounds=8,
        m_epochs=3,
        em_tolerance=1e-3,
        em_patience=2,
        batch_size=128,
        learning_rate=1e-3,
        weight_decay=1e-5,
        group_lambda=group_lambda,
        mixture_kl_weight=0.1,
        posterior_weight=0.5,
        assignment_weight=0.5,
        masked_reconstruction_weight=1.0,
        view_dropout=0.35,
        variance_floor=1e-4,
        training_feature_ratio=feature_ratio,
        selector_temperature=0.5,
        selector_view_temperature=1.0,
        selector_gumbel_start=1.0,
        selector_gumbel_end=0.05,
        device=device,
        dtype=dtype,
        seed=seed,
    )
    return replace(config, **overrides)


def split_dataset(path: str | Path, seed: int, dev_ratio: float = 0.2):
    dataset = load_multiview_mat(path)
    development, evaluation = development_split(dataset.labels, dev_ratio, seed)
    return (
        dataset,
        [view[development] for view in dataset.views],
        dataset.labels[development],
        [view[evaluation] for view in dataset.views],
        dataset.labels[evaluation],
    )


def run_once(
    views: list[np.ndarray],
    labels: np.ndarray,
    n_clusters: int,
    *,
    missing_ratio: float,
    feature_ratio: float,
    root_seed: int,
    group_lambda: float,
    device: str,
    dtype: str,
    config_overrides: dict | None = None,
):
    mask_seed = root_seed * 1_000_003 + int(round(missing_ratio * 1000.0))
    model_seed = mask_seed + 14_000_000
    mask = balanced_missing_mask(len(labels), len(views), missing_ratio, mask_seed)
    processed = minmax_normalize(views, mask)
    config = default_config(
        seed=model_seed,
        feature_ratio=feature_ratio,
        group_lambda=group_lambda,
        device=device,
        dtype=dtype,
        **(config_overrides or {}),
    )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    model = SCMVFS(n_clusters, config).fit(processed, mask)
    selected = model.select_global_indices(feature_ratio)
    completed = model.complete_selected(processed, mask, selected)
    features = select_features(completed, selected)
    metrics = evaluate_kmeans(features, labels, n_clusters, model_seed, 20)
    memory = (
        torch.cuda.max_memory_allocated() / (1024.0**2)
        if torch.cuda.is_available()
        else 0.0
    )
    return {
        **metrics,
        "fit_seconds": model.runtime_seconds_,
        "memory_mib": memory,
        "rounds": model.n_iter_,
        "termination": model.termination_reason_,
        "selected": selected,
        "model": model,
        "mask": mask,
        "processed": processed,
    }


def tune_group_lambda(
    dev_views: list[np.ndarray],
    dev_labels: np.ndarray,
    n_clusters: int,
    *,
    root_seed: int,
    device: str,
    dtype: str,
) -> float:
    best_value = None
    best_score = -np.inf
    for value in (1e-5, 1e-3):
        result = run_once(
            dev_views,
            dev_labels,
            n_clusters,
            missing_ratio=0.3,
            feature_ratio=0.3,
            root_seed=root_seed + 104729,
            group_lambda=value,
            device=device,
            dtype=dtype,
        )
        score = 0.5 * (result["acc"] + result["nmi"])
        if best_value is None or score > best_score or (
            score == best_score and value < best_value
        ):
            best_score = score
            best_value = value
        del result["model"]
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    assert best_value is not None
    return float(best_value)
