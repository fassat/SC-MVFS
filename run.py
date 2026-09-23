from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from scmvfs import SCMVFS, SCMVFSConfig
from scmvfs.data import (
    balanced_missing_mask,
    development_split,
    load_multiview_mat,
    minmax_normalize,
    select_features,
    selected_counts,
)
from scmvfs.metrics import evaluate_kmeans


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the SC-MVFS main experiment.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--missing-ratio", type=float, default=0.3)
    parser.add_argument("--feature-ratio", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dev-ratio", type=float, default=0.2)
    parser.add_argument(
        "--group-lambda",
        type=float,
        choices=[1e-5, 1e-3],
        default=None,
        help="Use a frozen central-setting value and skip development tuning.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--dtype", choices=["auto", "float32", "float64"], default="auto"
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def model_config(
    *, seed: int, feature_ratio: float, group_lambda: float, args: argparse.Namespace
) -> SCMVFSConfig:
    return SCMVFSConfig(
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
        device=args.device,
        dtype=args.dtype,
        seed=seed,
        verbose=args.verbose,
    )


def fit_and_evaluate(
    views: list[np.ndarray],
    labels: np.ndarray,
    mask: np.ndarray,
    n_clusters: int,
    config: SCMVFSConfig,
    feature_ratio: float,
    kmeans_seed: int,
) -> tuple[SCMVFS, np.ndarray, dict[str, float]]:
    processed = minmax_normalize(views, mask)
    model = SCMVFS(n_clusters, config).fit(processed, mask)
    selected = model.select_global_indices(feature_ratio)
    completed = model.complete_selected(processed, mask, selected)
    features = select_features(completed, selected)
    metrics = evaluate_kmeans(
        features, labels, n_clusters, seed=kmeans_seed, n_init=20
    )
    return model, selected, metrics


def tune_group_lambda(
    views: list[np.ndarray],
    labels: np.ndarray,
    n_clusters: int,
    args: argparse.Namespace,
) -> tuple[float, list[dict]]:
    tune_seed = args.seed + 104729
    mask = balanced_missing_mask(
        labels.size, len(views), 0.3, tune_seed
    )
    records = []
    for value in (1e-5, 1e-3):
        config = model_config(
            seed=tune_seed,
            feature_ratio=0.3,
            group_lambda=value,
            args=args,
        )
        model, selected, metrics = fit_and_evaluate(
            views,
            labels,
            mask,
            n_clusters,
            config,
            0.3,
            tune_seed,
        )
        score = 0.5 * (metrics["acc"] + metrics["nmi"])
        records.append(
            {
                "group_lambda": value,
                "score": score,
                **metrics,
                "selected_feature_count": int(selected.size),
                "fit_seconds": float(model.runtime_seconds_),
            }
        )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    best = max(records, key=lambda item: (item["score"], -item["group_lambda"]))
    return float(best["group_lambda"]), records


def main() -> None:
    args = parse_args()
    dataset = load_multiview_mat(args.dataset)
    development, evaluation = development_split(
        dataset.labels, args.dev_ratio, args.seed
    )
    dev_views = [view[development] for view in dataset.views]
    eval_views = [view[evaluation] for view in dataset.views]
    dev_labels = dataset.labels[development]
    eval_labels = dataset.labels[evaluation]

    print(
        f"{dataset.name}: n={dataset.n_samples}, views={dataset.n_views}, "
        f"dims={[view.shape[1] for view in dataset.views]}, "
        f"clusters={dataset.n_clusters}",
        flush=True,
    )
    if args.group_lambda is None:
        group_lambda, _ = tune_group_lambda(
            dev_views, dev_labels, dataset.n_clusters, args
        )
    else:
        group_lambda = float(args.group_lambda)
    print(f"selected lambda_2,1={group_lambda:g}", flush=True)

    dimensions = [view.shape[1] for view in dataset.views]
    mask_seed = args.seed * 1_000_003 + int(round(args.missing_ratio * 1000.0))
    model_seed = mask_seed + 14_000_000
    mask = balanced_missing_mask(
        eval_labels.size, dataset.n_views, args.missing_ratio, mask_seed
    )
    config = model_config(
        seed=model_seed,
        feature_ratio=args.feature_ratio,
        group_lambda=group_lambda,
        args=args,
    )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    model, selected, metrics = fit_and_evaluate(
        eval_views,
        eval_labels,
        mask,
        dataset.n_clusters,
        config,
        args.feature_ratio,
        model_seed,
    )
    peak_memory = (
        torch.cuda.max_memory_allocated() / (1024.0**2)
        if torch.cuda.is_available()
        else 0.0
    )
    print("\nSC-MVFS result", flush=True)
    print(f"seed: {args.seed}", flush=True)
    print(f"ACC: {metrics['acc']:.6f}", flush=True)
    print(f"NMI: {metrics['nmi']:.6f}", flush=True)
    print(f"selected features: {selected.size}", flush=True)
    print(f"selected per view: {selected_counts(selected, dimensions)}", flush=True)
    print(f"fit seconds: {model.runtime_seconds_:.3f}", flush=True)
    print(f"peak GPU memory MiB: {peak_memory:.3f}", flush=True)
    print(f"alternating rounds: {model.n_iter_}", flush=True)
    print(f"termination: {model.termination_reason_}", flush=True)


if __name__ == "__main__":
    main()
