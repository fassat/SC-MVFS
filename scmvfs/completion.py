from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

from .data import balanced_missing_mask, select_features
from .metrics import evaluate_kmeans
from .model import SCMVFS


class PureCompletion(SCMVFS):
    def fit_completion(
        self,
        views: Sequence[np.ndarray],
        selected: np.ndarray,
        support: str,
        epochs: int = 100,
    ) -> "PureCompletion":
        if support not in {"full", "selected"}:
            raise ValueError("support must be full or selected")
        arrays = [np.asarray(view, dtype=np.float64) for view in views]
        sample_count = arrays[0].shape[0]
        view_count = len(arrays)
        self.dimensions_ = [view.shape[1] for view in arrays]
        normalized = self._normalize_fit(
            arrays, np.ones((sample_count, view_count), dtype=bool)
        )
        self._build_networks(self.dimensions_)
        targets = [
            torch.as_tensor(view, device=self.device, dtype=self.dtype)
            for view in normalized
        ]
        keep = np.isin(np.arange(sum(self.dimensions_)), selected)
        gates = [
            torch.as_tensor(block, device=self.device, dtype=self.dtype)
            for block in np.split(keep, np.cumsum(self.dimensions_)[:-1])
        ]
        assert self.encoders_ is not None and self.decoders_ is not None
        parameters = list(self.encoders_.parameters()) + list(self.decoders_.parameters())
        optimizer = torch.optim.Adam(
            parameters,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        generator = np.random.default_rng(self.config.seed + 1701)
        for epoch in range(epochs):
            mask = balanced_missing_mask(
                sample_count, view_count, 0.3, self.config.seed + 3701 + epoch
            )
            order = generator.permutation(sample_count)
            for start in range(0, sample_count, self.config.batch_size):
                indices = order[start : start + self.config.batch_size]
                source = torch.as_tensor(
                    mask[indices], device=self.device, dtype=torch.bool
                )
                target = [view[indices] for view in targets]
                inputs = [
                    torch.where(source[:, view_index, None], view, torch.zeros_like(view))
                    for view_index, view in enumerate(target)
                ]
                if support == "selected":
                    inputs = [view * gate for view, gate in zip(inputs, gates)]
                mean, _ = self._infer(inputs, source)
                loss = self._reconstruction_loss(mean, target, ~source)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters, 5.0)
                optimizer.step()
        return self

    def complete(
        self,
        views: Sequence[np.ndarray],
        mask: np.ndarray,
        selected: np.ndarray,
        support: str,
    ) -> list[np.ndarray]:
        if self.encoders_ is None or self.decoders_ is None or self.dimensions_ is None:
            raise RuntimeError("Call fit_completion first")
        arrays = [np.asarray(view, dtype=np.float64) for view in views]
        normalized = self._normalize_transform(arrays, mask)
        offsets = np.cumsum([0, *self.dimensions_])
        tensors = []
        for view_index, value in enumerate(normalized):
            if support == "selected":
                local = selected[
                    (selected >= offsets[view_index])
                    & (selected < offsets[view_index + 1])
                ] - offsets[view_index]
                supported = np.zeros_like(value)
                supported[:, local] = value[:, local]
                value = supported
            tensors.append(torch.as_tensor(value, device=self.device, dtype=self.dtype))
        mask_tensor = torch.as_tensor(mask, device=self.device, dtype=torch.bool)
        with torch.no_grad():
            mean, _ = self._infer(tensors, mask_tensor)
            predictions = [decoder(mean).cpu().numpy() for decoder in self.decoders_]
        completed = [view.copy() for view in arrays]
        for view_index in range(len(arrays)):
            decoded = predictions[view_index] * self.scales_[view_index] + self.locations_[view_index]
            completed[view_index][~mask[:, view_index]] = decoded[~mask[:, view_index]]
        return completed


def completion_metrics(
    truth: Sequence[np.ndarray], completed: Sequence[np.ndarray], mask: np.ndarray
) -> tuple[float, float]:
    mse = []
    rmse = []
    for view_index, (actual, predicted) in enumerate(zip(truth, completed)):
        hidden = ~mask[:, view_index]
        error = np.asarray(actual)[hidden] - np.asarray(predicted)[hidden]
        value = float(np.mean(error**2))
        mse.append(value)
        rmse.append(float(np.sqrt(value)))
    return float(np.mean(mse)), float(np.mean(rmse))


def evaluate_completion(
    model: PureCompletion,
    truth: Sequence[np.ndarray],
    mask: np.ndarray,
    selected: np.ndarray,
    support: str,
    labels: np.ndarray,
    n_clusters: int,
    seed: int,
) -> dict[str, float]:
    completed = model.complete(truth, mask, selected, support)
    mse, rmse = completion_metrics(truth, completed, mask)
    clustering = evaluate_kmeans(
        select_features(completed, selected), labels, n_clusters, seed, 20
    )
    return {**clustering, "mse": mse, "rmse": rmse}
