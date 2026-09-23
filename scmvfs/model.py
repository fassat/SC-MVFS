from __future__ import annotations
import copy
from dataclasses import dataclass
from time import perf_counter
from typing import Sequence
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .metrics import fit_kmeans
from .utils import resolve_device, resolve_dtype

@dataclass(frozen=True)
class SCMVFSConfig:
    hidden_dim: int = 128
    latent_dim: int = 32
    warmup_epochs: int = 5
    em_rounds: int = 8
    m_epochs: int = 3
    em_tolerance: float = 0.001
    em_patience: int = 2
    batch_size: int = 128
    learning_rate: float = 0.001
    weight_decay: float = 1e-05
    group_lambda: float = 1e-05
    mixture_kl_weight: float = 0.1
    posterior_weight: float = 0.5
    assignment_weight: float = 0.5
    masked_reconstruction_weight: float = 1.0
    view_dropout: float = 0.35
    variance_floor: float = 0.0001
    training_feature_ratio: float = 0.3
    training_feature_count: int | None = None
    selector_temperature: float = 0.5
    selector_view_temperature: float = 1.0
    selector_gumbel_start: float = 1.0
    selector_gumbel_end: float = 0.05
    reconstruction_input: str = 'selected'
    teacher_pattern: str = 'observed'
    device: str = 'auto'
    dtype: str = 'auto'
    seed: int = 0
    verbose: bool = False

class _Encoder(nn.Module):

    def __init__(self, dimension: int, hidden: int, latent: int):
        super().__init__()
        self.input = nn.Linear(dimension, hidden)
        self.norm = nn.LayerNorm(hidden)
        self.output = nn.Linear(hidden, 2 * latent)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = F.gelu(self.norm(self.input(x)))
        mean, log_variance = self.output(hidden).chunk(2, dim=1)
        return (mean, log_variance.clamp(-7.0, 5.0))

class _Decoder(nn.Module):

    def __init__(self, dimension: int, hidden: int, latent: int):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(latent, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Linear(hidden, dimension))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.network(z)

class SCMVFS:
    """SC-MVFS model used by the main experiment."""

    def __init__(self, n_clusters: int, config: SCMVFSConfig | None=None):
        if n_clusters < 2:
            raise ValueError('n_clusters must be at least two.')
        self.n_clusters = int(n_clusters)
        self.config = config or SCMVFSConfig()
        self._validate_config()
        self.device = resolve_device(self.config.device)
        self.dtype = resolve_dtype(self.config.dtype, self.device)
        self.encoders_: nn.ModuleList | None = None
        self.decoders_: nn.ModuleList | None = None
        self.selector_logits_: nn.Parameter | None = None
        self.selector_view_logits_: nn.Parameter | None = None
        self.locations_: list[np.ndarray] | None = None
        self.scales_: list[np.ndarray] | None = None
        self.dimensions_: list[int] | None = None
        self.mixture_weights_: torch.Tensor | None = None
        self.mixture_means_: torch.Tensor | None = None
        self.mixture_variances_: torch.Tensor | None = None
        self.feature_scores_: np.ndarray | None = None
        self.selection_order_: np.ndarray | None = None
        self.selected_indices_: np.ndarray | None = None
        self.history_: list[dict[str, float]] = []
        self.runtime_seconds_: float = 0.0
        self.n_iter_: int = 0
        self.converged_: bool = False
        self.termination_reason_: str = 'not_fitted'

    def _validate_config(self) -> None:
        c = self.config
        if min(c.hidden_dim, c.latent_dim, c.em_rounds, c.m_epochs, c.batch_size) < 1:
            raise ValueError('network sizes and iteration counts must be positive.')
        if c.em_tolerance <= 0 or c.em_patience < 1:
            raise ValueError('EM tolerance must be positive and patience at least one.')
        if c.warmup_epochs < 0 or c.learning_rate <= 0 or c.weight_decay < 0:
            raise ValueError('invalid optimizer configuration.')
        if c.group_lambda < 0:
            raise ValueError('group_lambda must be nonnegative.')
        if min(c.mixture_kl_weight, c.posterior_weight, c.assignment_weight, c.masked_reconstruction_weight) < 0:
            raise ValueError('loss weights must be nonnegative.')
        if not 0 <= c.view_dropout < 1:
            raise ValueError('view_dropout must lie in [0, 1).')
        if c.variance_floor <= 0:
            raise ValueError('variance_floor must be positive.')
        if not 0 < c.training_feature_ratio <= 1:
            raise ValueError('training_feature_ratio must lie in (0, 1].')
        if c.training_feature_count is not None and c.training_feature_count < 1:
            raise ValueError('training_feature_count must be positive when supplied.')
        if c.selector_temperature <= 0:
            raise ValueError('selector_temperature must be positive.')
        if c.selector_view_temperature <= 0:
            raise ValueError('selector_view_temperature must be positive.')
        if c.selector_gumbel_start < 0 or c.selector_gumbel_end < 0:
            raise ValueError('selector Gumbel scales must be nonnegative.')
        if c.selector_gumbel_end > c.selector_gumbel_start:
            raise ValueError('selector_gumbel_end cannot exceed its start value.')
        if c.reconstruction_input not in {'selected', 'full'}:
            raise ValueError('reconstruction_input must be selected or full.')
        if c.teacher_pattern not in {'observed', 'reduced'}:
            raise ValueError('teacher_pattern must be observed or reduced.')

    @staticmethod
    def _validate_data(views: Sequence[np.ndarray], observed_mask: np.ndarray) -> tuple[list[np.ndarray], np.ndarray]:
        if len(views) < 2:
            raise ValueError('SCMVFS requires at least two views.')
        arrays = [np.asarray(x, dtype=np.float64) for x in views]
        n = arrays[0].shape[0]
        if any((x.ndim != 2 or x.shape[0] != n for x in arrays)):
            raise ValueError('All views must be aligned sample-by-feature matrices.')
        mask = np.asarray(observed_mask, dtype=bool)
        if mask.shape != (n, len(arrays)):
            raise ValueError('observed_mask has an incompatible shape.')
        if np.any(mask.sum(axis=1) == 0) or np.any(mask.sum(axis=0) == 0):
            raise ValueError('Every sample and every view must have observations.')
        return (arrays, mask)

    def _normalize_fit(self, arrays: Sequence[np.ndarray], mask: np.ndarray) -> list[np.ndarray]:
        locations: list[np.ndarray] = []
        scales: list[np.ndarray] = []
        normalized: list[np.ndarray] = []
        for v, array in enumerate(arrays):
            observed = mask[:, v]
            location = array[observed].mean(axis=0)
            scale = array[observed].std(axis=0, ddof=0)
            scale = np.where(scale > 1e-08, scale, 1.0)
            value = (array - location[None, :]) / scale[None, :]
            value[~observed] = 0.0
            locations.append(location)
            scales.append(scale)
            normalized.append(np.asarray(value, dtype=np.float64, order='C'))
        self.locations_ = locations
        self.scales_ = scales
        return normalized

    def _normalize_transform(self, arrays: Sequence[np.ndarray], mask: np.ndarray) -> list[np.ndarray]:
        if self.locations_ is None or self.scales_ is None:
            raise RuntimeError('Model normalization is not fitted.')
        values = []
        for v, array in enumerate(arrays):
            value = (np.asarray(array) - self.locations_[v]) / self.scales_[v]
            value = np.asarray(value, dtype=np.float64)
            value[~mask[:, v]] = 0.0
            values.append(value)
        return values

    def _build_networks(self, dimensions: Sequence[int]) -> None:
        torch.manual_seed(self.config.seed)
        if self.device.type == 'cuda':
            torch.cuda.manual_seed_all(self.config.seed)
        self.encoders_ = nn.ModuleList([_Encoder(d, self.config.hidden_dim, self.config.latent_dim) for d in dimensions]).to(device=self.device, dtype=self.dtype)
        self.decoders_ = nn.ModuleList([_Decoder(d, self.config.hidden_dim, self.config.latent_dim) for d in dimensions]).to(device=self.device, dtype=self.dtype)
        self.selector_logits_ = nn.Parameter(
            torch.zeros(sum(dimensions), device=self.device, dtype=self.dtype)
        )
        self.selector_view_logits_ = nn.Parameter(
            torch.zeros(len(dimensions), device=self.device, dtype=self.dtype)
        )

    def _effective_selector_logits(self) -> torch.Tensor:
        """Log probabilities of a learned view/within-view hierarchy."""
        if self.selector_logits_ is None or self.selector_view_logits_ is None or self.dimensions_ is None:
            raise RuntimeError('Selector is not initialized.')
        blocks = torch.split(self.selector_logits_, self.dimensions_)
        log_view = torch.log_softmax(self.selector_view_logits_ / self.config.selector_view_temperature, dim=0)
        adjusted = [torch.log_softmax(local, dim=0) + log_view[view_index] for view_index, local in enumerate(blocks)]
        return torch.cat(adjusted)

    def _view_probabilities(self, *, detach: bool) -> torch.Tensor:
        """Return temperature-calibrated learned view allocation mass."""
        if self.selector_view_logits_ is None:
            raise RuntimeError('Selector is not initialized.')
        probabilities = torch.softmax(self.selector_view_logits_ / self.config.selector_view_temperature, dim=0)
        return probabilities.detach() if detach else probabilities

    def _learned_view_counts(self, count: int, probabilities: np.ndarray | None=None) -> np.ndarray:
        """Capacity-aware largest-remainder allocation from learned view mass."""
        if self.selector_view_logits_ is None or self.dimensions_ is None:
            raise RuntimeError('Selector is not initialized.')
        if probabilities is None:
            probabilities = self._view_probabilities(detach=True).cpu().numpy().astype(np.float64)
        dimensions = np.asarray(self.dimensions_, dtype=np.int64)
        ideal = probabilities * int(count)
        allocation = np.minimum(np.floor(ideal).astype(np.int64), dimensions)
        remaining = int(count - allocation.sum())
        while remaining > 0:
            available = allocation < dimensions
            if not np.any(available):
                break
            priority = ideal - allocation
            priority[~available] = -np.inf
            choice = int(np.argmax(priority))
            allocation[choice] += 1
            remaining -= 1
        return allocation

    def _select_learned_count(self, feature_count: int) -> np.ndarray:
        if self.selector_logits_ is None or self.dimensions_ is None:
            raise RuntimeError('Selector is not initialized.')
        allocation = self._learned_view_counts(int(feature_count))
        offsets = np.cumsum([0, *self.dimensions_])
        logits = self.selector_logits_.detach().cpu().numpy()
        blocks = []
        for view_index, local_count in enumerate(allocation):
            if local_count <= 0:
                continue
            begin, end = (offsets[view_index], offsets[view_index + 1])
            local = np.argsort(-logits[begin:end], kind='stable')[:local_count]
            blocks.append(local.astype(np.int64) + begin)
        return np.sort(np.concatenate(blocks)).astype(np.int64)

    def _exact_k_gate(self, generator: torch.Generator | None=None, noise_scale: float=0.0) -> torch.Tensor:
        """Exploratory global exact-K gate with a sigmoid surrogate gradient."""
        if self.selector_logits_ is None:
            raise RuntimeError('Selector is not initialized.')
        logits = self._effective_selector_logits()
        count = int(self.config.training_feature_count) if self.config.training_feature_count is not None else max(1, int(np.ceil(self.config.training_feature_ratio * logits.numel())))
        count = min(count, logits.numel())
        perturbed = logits
        if noise_scale > 0:
            uniform = torch.rand(logits.shape, device=logits.device, dtype=logits.dtype, generator=generator).clamp_(1e-06, 1.0 - 1e-06)
            gumbel = -torch.log(-torch.log(uniform))
            perturbed = logits + float(noise_scale) * gumbel
        hard = torch.zeros_like(logits)
        assert self.selector_view_logits_ is not None
        view_logits = self.selector_view_logits_.detach()
        if noise_scale > 0:
            view_uniform = torch.rand(view_logits.shape, device=view_logits.device, dtype=view_logits.dtype, generator=generator).clamp_(1e-06, 1.0 - 1e-06)
            view_logits = view_logits + float(noise_scale) * -torch.log(-torch.log(view_uniform))
        view_probabilities = torch.softmax(view_logits / self.config.selector_view_temperature, dim=0).cpu().numpy().astype(np.float64)
        allocation = self._learned_view_counts(count, view_probabilities)
        offsets = np.cumsum([0, *self.dimensions_])
        for view_index, local_count in enumerate(allocation):
            if local_count <= 0:
                continue
            begin, end = (offsets[view_index], offsets[view_index + 1])
            local = torch.topk(perturbed[begin:end], int(local_count), sorted=False).indices
            hard[local + begin] = 1.0
        with torch.no_grad():
            lower = perturbed.min() - 20.0 * self.config.selector_temperature
            upper = perturbed.max() + 20.0 * self.config.selector_temperature
            for _ in range(32):
                threshold = 0.5 * (lower + upper)
                mass = torch.sigmoid((perturbed - threshold) / self.config.selector_temperature).sum()
                if mass > count:
                    lower = threshold
                else:
                    upper = threshold
            threshold = 0.5 * (lower + upper)
        soft = torch.sigmoid((perturbed - threshold) / self.config.selector_temperature)
        return hard + soft - soft.detach()

    def _apply_gate(self, views: Sequence[torch.Tensor], gate: torch.Tensor) -> list[torch.Tensor]:
        if self.dimensions_ is None:
            raise RuntimeError('Feature dimensions are unavailable.')
        blocks = torch.split(gate, self.dimensions_)
        return [value * block[None, :] for value, block in zip(views, blocks)]

    def _infer(self, views: Sequence[torch.Tensor], mask: torch.Tensor, encoders=None) -> tuple[torch.Tensor, torch.Tensor]:
        encoders = self.encoders_ if encoders is None else encoders
        if encoders is None:
            raise RuntimeError('Networks are not initialized.')
        precision_sum = torch.ones((mask.shape[0], self.config.latent_dim), device=self.device, dtype=self.dtype)
        weighted_mean = torch.zeros_like(precision_sum)
        for v, encoder in enumerate(encoders):
            mean, log_variance = encoder(views[v])
            precision = torch.exp(-log_variance).clamp(max=10000.0)
            observed = mask[:, v:v + 1].to(self.dtype)
            precision_sum = precision_sum + observed * precision
            weighted_mean = weighted_mean + observed * precision * mean
        variance = precision_sum.reciprocal().clamp_min(self.config.variance_floor)
        mean = variance * weighted_mean
        return (mean, torch.log(variance))

    def _responsibilities(self, mean: torch.Tensor, log_variance: torch.Tensor) -> torch.Tensor:
        if self.mixture_weights_ is None:
            raise RuntimeError('Mixture is not initialized.')
        q_variance = torch.exp(log_variance)[:, None, :]
        difference = mean[:, None, :] - self.mixture_means_[None, :, :]
        variance = self.mixture_variances_[None, :, :]
        logits = torch.log(self.mixture_weights_.clamp_min(1e-12))[None, :]
        logits = logits - 0.5 * torch.sum(torch.log(variance) + (q_variance + difference.square()) / variance, dim=2)
        return torch.softmax(logits, dim=1)

    def _initialize_mixture(self, mean: torch.Tensor) -> None:
        values = mean.detach().cpu().numpy().astype(np.float64)
        centers, labels = fit_kmeans(values, self.n_clusters, self.config.seed, n_init=10)
        weights = np.bincount(labels, minlength=self.n_clusters).astype(np.float64)
        weights = np.maximum(weights, 1.0)
        weights /= weights.sum()
        global_variance = np.var(values, axis=0) + self.config.variance_floor
        variances = np.empty_like(centers)
        for cluster in range(self.n_clusters):
            members = values[labels == cluster]
            variances[cluster] = np.var(members, axis=0) + self.config.variance_floor if members.shape[0] > 1 else global_variance
        self.mixture_weights_ = torch.as_tensor(weights, device=self.device, dtype=self.dtype)
        self.mixture_means_ = torch.as_tensor(centers, device=self.device, dtype=self.dtype)
        self.mixture_variances_ = torch.as_tensor(variances, device=self.device, dtype=self.dtype)

    @torch.no_grad()
    def _update_mixture(self, mean: torch.Tensor, log_variance: torch.Tensor, responsibilities: torch.Tensor) -> None:
        mass = responsibilities.sum(dim=0).clamp_min(0.001)
        weights = mass / mass.sum()
        centers = responsibilities.T @ mean / mass[:, None]
        q_variance = torch.exp(log_variance)
        differences = mean[:, None, :] - centers[None, :, :]
        variances = torch.sum(responsibilities[:, :, None] * (q_variance[:, None, :] + differences.square()), dim=0) / mass[:, None]
        self.mixture_weights_ = weights.clamp_min(1e-08)
        self.mixture_weights_ /= self.mixture_weights_.sum()
        self.mixture_means_ = centers
        self.mixture_variances_ = variances.clamp_min(self.config.variance_floor)

    def _reconstruction_loss(self, latent: torch.Tensor, views: Sequence[torch.Tensor], mask: torch.Tensor) -> torch.Tensor:
        if self.decoders_ is None:
            raise RuntimeError('Networks are not initialized.')
        losses = []
        for v, decoder in enumerate(self.decoders_):
            observed = mask[:, v]
            if torch.any(observed):
                prediction = decoder(latent[observed])
                losses.append(F.mse_loss(prediction, views[v][observed]))
        if not losses:
            return latent.sum() * 0.0
        return torch.stack(losses).mean()

    def _pattern_losses(self, gated_views: Sequence[torch.Tensor], target_views: Sequence[torch.Tensor], observed_mask: torch.Tensor, target_mean: torch.Tensor, target_log_variance: torch.Tensor, target_responsibilities: torch.Tensor, generator: torch.Generator, teacher_encoders=None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute losses under reduced observation patterns."""
        masked_reconstructions: list[torch.Tensor] = []
        posteriors: list[torch.Tensor] = []
        assignments: list[torch.Tensor] = []
        pattern_objectives: list[torch.Tensor] = []
        for _ in range(1):
            reduced_mask = self._drop_views(observed_mask, generator)
            reduced_mean, reduced_log_variance = self._infer(gated_views, reduced_mask)
            reduced_responsibilities = self._responsibilities(reduced_mean, reduced_log_variance)
            reconstruction_latent = (
                reduced_mean
                if self.config.reconstruction_input == 'selected'
                else self._infer(target_views, reduced_mask)[0]
            )
            if self.config.teacher_pattern == 'reduced':
                with torch.no_grad():
                    target_mean, target_log_variance = self._infer(
                        target_views, reduced_mask, teacher_encoders
                    )
                    target_responsibilities = self._responsibilities(
                        target_mean, target_log_variance
                    )
            reconstruction_target_mask = observed_mask
            masked_reconstruction = self._reconstruction_loss(reconstruction_latent, target_views, reconstruction_target_mask)
            posterior = self._posterior_preservation_loss(reduced_mean, reduced_log_variance, target_mean, target_log_variance)
            assignment = torch.sum(target_responsibilities * (torch.log(target_responsibilities.clamp_min(1e-08)) - torch.log(reduced_responsibilities.clamp_min(1e-08))), dim=1).mean()
            masked_reconstructions.append(masked_reconstruction)
            posteriors.append(posterior)
            assignments.append(assignment)
            pattern_objectives.append(self.config.masked_reconstruction_weight * masked_reconstruction + self.config.posterior_weight * posterior + self.config.assignment_weight * assignment)
        stacked_objectives = torch.stack(pattern_objectives)
        pattern_variance = stacked_objectives.var(unbiased=False)
        return (torch.stack(masked_reconstructions).mean(), torch.stack(posteriors).mean(), torch.stack(assignments).mean(), pattern_variance)

    def _mixture_kl(self, mean: torch.Tensor, log_variance: torch.Tensor, responsibilities: torch.Tensor) -> torch.Tensor:
        q_variance = torch.exp(log_variance)[:, None, :]
        difference = mean[:, None, :] - self.mixture_means_[None, :, :]
        variance = self.mixture_variances_[None, :, :]
        expected_negative_log_prior = 0.5 * torch.sum(responsibilities * torch.sum(torch.log(variance) + (q_variance + difference.square()) / variance, dim=2), dim=1).mean()
        entropy_q = 0.5 * torch.sum(1.0 + log_variance, dim=1).mean()
        categorical = torch.sum(responsibilities * (torch.log(responsibilities.clamp_min(1e-08)) - torch.log(self.mixture_weights_.clamp_min(1e-08))[None, :]), dim=1).mean()
        return expected_negative_log_prior - entropy_q + categorical

    @staticmethod
    def _gaussian_kl(mean: torch.Tensor, log_variance: torch.Tensor, target_mean: torch.Tensor, target_log_variance: torch.Tensor) -> torch.Tensor:
        target_variance = torch.exp(target_log_variance)
        value = target_log_variance - log_variance
        value = value + (torch.exp(log_variance) + (mean - target_mean).square()) / target_variance.clamp_min(1e-08) - 1.0
        return 0.5 * value.sum(dim=1).mean()

    def _posterior_preservation_loss(self, mean: torch.Tensor, log_variance: torch.Tensor, target_mean: torch.Tensor, target_log_variance: torch.Tensor) -> torch.Tensor:
        return self._gaussian_kl(mean, log_variance, target_mean, target_log_variance)

    def _group_penalty(self) -> torch.Tensor:
        if self.encoders_ is None:
            raise RuntimeError('Networks are not initialized.')
        penalties = [torch.linalg.vector_norm(encoder.input.weight, dim=0).sum() for encoder in self.encoders_]
        return torch.stack(penalties).sum()

    def _drop_views(self, mask: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
        reduced = mask.clone()
        random = torch.rand(mask.shape, device=self.device, generator=generator)
        reduced &= random >= self.config.view_dropout
        empty = reduced.sum(dim=1) == 0
        if torch.any(empty):
            rows = torch.nonzero(empty, as_tuple=False).squeeze(1)
            for row in rows.tolist():
                available = torch.nonzero(mask[row], as_tuple=False).squeeze(1)
                choice = available[torch.randint(available.numel(), (1,), device=self.device, generator=generator)]
                reduced[row, choice] = True
        return reduced

    def fit(self, views: Sequence[np.ndarray], observed_mask: np.ndarray) -> 'SCMVFS':
        started = perf_counter()
        arrays, mask_array = self._validate_data(views, observed_mask)
        normalized = self._normalize_fit(arrays, mask_array)
        self.dimensions_ = [x.shape[1] for x in arrays]
        self._build_networks(self.dimensions_)
        assert self.encoders_ is not None and self.decoders_ is not None
        tensors = [torch.as_tensor(x, device=self.device, dtype=self.dtype) for x in normalized]
        mask = torch.as_tensor(mask_array, device=self.device, dtype=torch.bool)
        parameters = list(self.encoders_.parameters()) + list(self.decoders_.parameters())
        assert self.selector_logits_ is not None
        parameters.append(self.selector_logits_)
        assert self.selector_view_logits_ is not None
        if self.selector_view_logits_.requires_grad:
            parameters.append(self.selector_view_logits_)
        optimizer = torch.optim.Adam(parameters, lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
        generator = torch.Generator(device=self.device)
        generator.manual_seed(self.config.seed + 1701)
        n_samples = mask.shape[0]
        for epoch in range(self.config.warmup_epochs):
            order = torch.randperm(n_samples, device=self.device, generator=generator)
            for begin in range(0, n_samples, self.config.batch_size):
                indices = order[begin:begin + self.config.batch_size]
                batch_views = [x[indices] for x in tensors]
                batch_mask = mask[indices]
                mean, _ = self._infer(batch_views, batch_mask)
                loss = self._reconstruction_loss(mean, batch_views, batch_mask)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters, 5.0)
                optimizer.step()
        assert self.selector_logits_ is not None
        with torch.no_grad():
            self.selector_logits_.zero_()
        with torch.no_grad():
            initial_mean, initial_log_variance = self._infer(tensors, mask)
        self._initialize_mixture(initial_mean)
        previous_responsibilities: torch.Tensor | None = None
        stable_rounds = 0
        selector_step = 0
        selector_steps = max(1, self.config.em_rounds * self.config.m_epochs * int(np.ceil(n_samples / self.config.batch_size)))
        fixed_e_step: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
        for em_round in range(self.config.em_rounds):
            if fixed_e_step is None:
                with torch.no_grad():
                    teacher_mean, teacher_log_variance = self._infer(tensors, mask)
                    teacher_responsibilities = self._responsibilities(teacher_mean, teacher_log_variance)
                    self._update_mixture(teacher_mean, teacher_log_variance, teacher_responsibilities)
                    teacher_responsibilities = self._responsibilities(teacher_mean, teacher_log_variance)
            else:
                teacher_mean, teacher_log_variance, teacher_responsibilities = fixed_e_step
            teacher_encoders = None
            if self.config.teacher_pattern == 'reduced':
                teacher_encoders = copy.deepcopy(self.encoders_).eval()
                teacher_encoders.requires_grad_(False)
            responsibility_change = float('inf') if previous_responsibilities is None else float(torch.mean(torch.abs(teacher_responsibilities - previous_responsibilities)).cpu())
            previous_responsibilities = teacher_responsibilities.detach().clone()
            totals = {'loss': 0.0, 'reconstruction': 0.0, 'masked_reconstruction': 0.0, 'posterior': 0.0, 'assignment': 0.0, 'mixture_kl': 0.0, 'group_l21': 0.0}
            batches = 0
            for _ in range(self.config.m_epochs):
                order = torch.randperm(n_samples, device=self.device, generator=generator)
                for begin in range(0, n_samples, self.config.batch_size):
                    indices = order[begin:begin + self.config.batch_size]
                    batch_views = [x[indices] for x in tensors]
                    batch_mask = mask[indices]
                    mean, log_variance = self._infer(batch_views, batch_mask)
                    selector_progress = selector_step / max(selector_steps - 1, 1)
                    gumbel_scale = self.config.selector_gumbel_start * (max(self.config.selector_gumbel_end, 1e-08) / max(self.config.selector_gumbel_start, 1e-08)) ** selector_progress if self.config.selector_gumbel_start > 0 else 0.0
                    gated_views = self._apply_gate(batch_views, self._exact_k_gate(generator=generator, noise_scale=gumbel_scale))
                    responsibilities = self._responsibilities(mean, log_variance)
                    target_mean = teacher_mean[indices]
                    target_log_variance = teacher_log_variance[indices]
                    target_responsibilities = teacher_responsibilities[indices]
                    reconstruction = self._reconstruction_loss(mean, batch_views, batch_mask)
                    masked_reconstruction, posterior, assignment, pattern_variance = self._pattern_losses(gated_views, batch_views, batch_mask, target_mean, target_log_variance, target_responsibilities, generator, teacher_encoders)
                    mixture_kl = self._mixture_kl(mean, log_variance, responsibilities)
                    group = self._group_penalty()
                    loss = reconstruction
                    loss = loss + self.config.masked_reconstruction_weight * masked_reconstruction
                    loss = loss + self.config.posterior_weight * posterior
                    loss = loss + self.config.assignment_weight * assignment
                    loss = loss + self.config.mixture_kl_weight * mixture_kl
                    loss = loss + self.config.group_lambda * group
                    loss = loss + 0.0 * pattern_variance
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(parameters, 5.0)
                    optimizer.step()
                    selector_step += 1
                    totals['loss'] += float(loss.detach().cpu())
                    totals['reconstruction'] += float(reconstruction.detach().cpu())
                    totals['masked_reconstruction'] += float(masked_reconstruction.detach().cpu())
                    totals['posterior'] += float(posterior.detach().cpu())
                    totals['assignment'] += float(assignment.detach().cpu())
                    totals['mixture_kl'] += float(mixture_kl.detach().cpu())
                    totals['group_l21'] += float(group.detach().cpu())
                    batches += 1
            record = {key: value / max(batches, 1) for key, value in totals.items()}
            record['em_round'] = float(em_round + 1)
            record['responsibility_change'] = responsibility_change
            if self.selector_view_logits_ is not None:
                view_probabilities = self._view_probabilities(detach=True)
                record['view_probability_entropy'] = float(-torch.sum(view_probabilities * torch.log(view_probabilities.clamp_min(1e-12))).cpu())
                for view_index, probability in enumerate(view_probabilities.tolist()):
                    record[f'view_probability_{view_index}'] = float(probability)
            self.history_.append(record)
            stable_rounds = stable_rounds + 1 if responsibility_change < self.config.em_tolerance else 0
            if self.config.verbose:
                print(f"EM={em_round + 1} loss={record['loss']:.5f} rec={record['reconstruction']:.5f} post={record['posterior']:.5f} delta_r={responsibility_change:.3e}", flush=True)
            if stable_rounds >= self.config.em_patience:
                self.converged_ = True
                self.termination_reason_ = 'responsibilities_stable'
                break
        self.n_iter_ = len(self.history_)
        if not self.converged_:
            self.termination_reason_ = 'maximum_em_rounds'
        assert self.selector_logits_ is not None
        self.feature_scores_ = self._effective_selector_logits().detach().cpu().numpy().astype(np.float64)
        self.selection_order_ = np.argsort(-self.feature_scores_, kind='stable').astype(np.int64)
        self.runtime_seconds_ = perf_counter() - started
        return self

    def select_global_indices(self, feature_ratio: float) -> np.ndarray:
        if self.feature_scores_ is None or self.selection_order_ is None:
            raise RuntimeError('Call fit before selecting features.')
        if not 0 < feature_ratio <= 1:
            raise ValueError('feature_ratio must lie in (0, 1].')
        count = max(1, int(np.ceil(feature_ratio * self.feature_scores_.size)))
        selected = self._select_learned_count(count)
        self.selected_indices_ = selected
        return selected.copy()

    def select_global_count(self, feature_count: int) -> np.ndarray:
        if self.feature_scores_ is None or self.selection_order_ is None:
            raise RuntimeError('Call fit before selecting features.')
        if not 1 <= int(feature_count) <= self.feature_scores_.size:
            raise ValueError('feature_count is outside the available feature range.')
        selected = self._select_learned_count(int(feature_count))
        self.selected_indices_ = selected
        return selected.copy()

    def complete_selected(self, views: Sequence[np.ndarray], observed_mask: np.ndarray, selected_indices: np.ndarray) -> list[np.ndarray]:
        """Complete missing views using only the final selected coordinates."""
        return self._complete_with_input_support(views, observed_mask, selected_indices=np.asarray(selected_indices, dtype=np.int64))

    def _complete_with_input_support(self, views: Sequence[np.ndarray], observed_mask: np.ndarray, *, selected_indices: np.ndarray) -> list[np.ndarray]:
        if self.encoders_ is None or self.decoders_ is None or self.dimensions_ is None:
            raise RuntimeError('Call fit before completion.')
        arrays, mask_array = self._validate_data(views, observed_mask)
        normalized = self._normalize_transform(arrays, mask_array)
        offsets = np.cumsum([0, *self.dimensions_])
        tensors: list[torch.Tensor] = []
        for v, value in enumerate(normalized):
            local = selected_indices[(selected_indices >= offsets[v]) & (selected_indices < offsets[v + 1])] - offsets[v]
            supported = np.zeros_like(value)
            if local.size:
                supported[:, local] = value[:, local]
            tensors.append(torch.as_tensor(supported, device=self.device, dtype=self.dtype))
        mask = torch.as_tensor(mask_array, device=self.device, dtype=torch.bool)
        with torch.no_grad():
            mean, _ = self._infer(tensors, mask)
            predictions = [decoder(mean).cpu().numpy() for decoder in self.decoders_]
        completed = [x.copy() for x in arrays]
        for v in range(len(arrays)):
            decoded = predictions[v] * self.scales_[v] + self.locations_[v]
            completed[v][~mask_array[:, v]] = decoded[~mask_array[:, v]]
        return completed

