"""Disk usage estimation utilities for data generation pipelines."""

import random
from collections.abc import Callable
from typing import NamedTuple
from typing import Protocol

from transformers import AutoConfig

__all__ = ["DiskEstimate", "compute_disk_estimate_gb", "log_disk_estimate"]

BYTES_PER_FP16 = 2
_SAMPLE_SIZE = 1000


class DiskEstimate(NamedTuple):
    """Result of a disk usage estimate for a hidden-states generation run."""

    total_gb: float
    n_layers: int
    hidden_size: int
    avg_len: float


class _SequenceDataset(Protocol):
    """Minimal interface expected from the dataset argument."""

    def __len__(self) -> int: ...

    def select(self, indices: list[int]) -> "_SequenceDataset": ...

    def __getitem__(self, key: str) -> list: ...


def compute_disk_estimate_gb(
    layer_ids: list[int] | None,
    model_path: str,
    dataset: _SequenceDataset,
    n_samples: int,
    seed: int = 0,
    trust_remote_code: bool = False,
) -> DiskEstimate:
    """Compute an estimated disk usage for generated hidden states files.

    The estimate is based on the average sequence length of a random sample of
    the dataset, so it reflects expected usage rather than a worst-case bound.

    When ``layer_ids`` is ``None``, the layer count is derived from the model
    config using the same auto-selection logic as ``VllmHiddenStatesGenerator``:
    ``[2, nl // 2, nl - 3, nl - 1]``.

    Args:
        layer_ids: Layer IDs to capture, or None to mirror the generator's
            auto-selection.
        model_path: HuggingFace model ID or local path.
        dataset: Dataset with a ``select`` method and an ``'input_ids'`` column.
        n_samples: Number of samples to estimate for.
        seed: Random seed for reproducible sequence-length sampling.
        trust_remote_code: Passed to AutoConfig.from_pretrained.

    Returns:
        A :class:`DiskEstimate` named tuple.
    """
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    if hasattr(config, "text_config"):
        config = config.text_config
    hidden_size: int = config.hidden_size

    if layer_ids is not None:
        n_layers = len(layer_ids)
    else:
        nl = config.num_hidden_layers
        n_layers = len([2, nl // 2, nl - 3, nl - 1])

    sample_size = min(_SAMPLE_SIZE, len(dataset))
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(dataset)), sample_size))
    lengths = [len(seq) for seq in dataset.select(indices)["input_ids"]]
    avg_len = sum(lengths) / sample_size

    total_gb = n_layers * BYTES_PER_FP16 * hidden_size * avg_len * n_samples / (1024**3)

    return DiskEstimate(
        total_gb=total_gb,
        n_layers=n_layers,
        hidden_size=hidden_size,
        avg_len=avg_len,
    )


def log_disk_estimate(
    log_fn: Callable[[str], None],
    layer_ids: list[int] | None,
    model_path: str,
    dataset: _SequenceDataset,
    n_samples: int,
    seed: int = 42,
    trust_remote_code: bool = False,
) -> None:
    """Compute and log an estimated disk usage for generated hidden states files.

    Always prefixes the message with ``DISK ESTIMATE:`` so the output is
    consistent regardless of which ``log_fn`` is passed (e.g. a plain
    ``logging.Logger.warning`` or a styled ``PipelineLogger.disk_estimate``).

    Silently logs a warning and returns without raising if the model config
    cannot be loaded (e.g. network unavailable, invalid path).

    Args:
        log_fn: Callable that accepts a single formatted string.
        layer_ids: Layer IDs to capture, or None to mirror the generator's
            auto-selection.
        model_path: HuggingFace model ID or local path.
        dataset: Dataset with a ``select`` method and an ``'input_ids'`` column.
        n_samples: Number of samples to estimate for.
        seed: Random seed for reproducible sequence-length sampling.
        trust_remote_code: Passed to AutoConfig.from_pretrained.
    """
    try:
        est = compute_disk_estimate_gb(
            layer_ids,
            model_path,
            dataset,
            n_samples,
            seed=seed,
            trust_remote_code=trust_remote_code,
        )
    except Exception as exc:
        log_fn(f"DISK ESTIMATE: could not compute estimate: {exc}")
        return

    log_fn(
        f"DISK ESTIMATE: ~{est.total_gb:.1f} GB for {n_samples} remaining samples "
        f"({est.n_layers} layers × hidden_size {est.hidden_size} × avg_len {est.avg_len:.0f} × fp16)"
    )
