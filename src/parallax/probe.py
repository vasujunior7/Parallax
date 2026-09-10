"""The confidence floor: how far is this patch from anything we validated on?

**Why this is unsupervised.** The design notes describe the probe as "logistic regression on
good vs anomalous activations", but VisA's train split is *normal only* — 8,659 normal
frames, zero anomalies. Fitting a supervised classifier would require borrowing anomalies
from the test split, which leaks the evaluation. The behaviour the notes actually specify —
"if the activation lands far from the cluster of known good activations, return a high OOD
score" — is a density question, not a classification one, and needs no anomaly labels.

So we fit a Gaussian to normal patch activations and score by **Mahalanobis distance**. That
is PaDiM (Defard et al., 2020) with a shared covariance, and it is the honest floor: it
trains on normals alone and never sees a defect during fitting.

Covariance in 768 dimensions is ill-conditioned unless the sample count greatly exceeds the
dimension, so it is regularised by shrinkage toward a scaled identity.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

DEFAULT_SHRINKAGE = 0.10
MIN_SAMPLES_PER_DIM = 2.0


class ProbeError(RuntimeError):
    """Raised when a probe cannot be fitted or applied."""


@dataclass(frozen=True)
class LinearProbe:
    """Gaussian density over normal patch activations, scored by Mahalanobis distance."""

    mean: np.ndarray  # (dim,)
    precision: np.ndarray  # (dim, dim) — inverse covariance
    dim: int
    n_samples: int
    shrinkage: float
    model_id: str

    def score(self, activations: np.ndarray) -> np.ndarray:
        """Squared Mahalanobis distance per patch. Higher means less familiar.

        ``activations`` is ``(n_patches, dim)``; returns ``(n_patches,)``.
        """
        if activations.ndim != 2:
            raise ProbeError(f"expected (n_patches, dim), got {activations.shape}")
        if activations.shape[1] != self.dim:
            raise ProbeError(
                f"activations are {activations.shape[1]}-dim, probe is {self.dim}-dim"
            )

        centred = activations.astype(np.float64) - self.mean
        # einsum avoids materialising the (n_patches, dim) intermediate twice.
        return np.einsum("ij,jk,ik->i", centred, self.precision, centred).astype(np.float32)

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            mean=self.mean,
            precision=self.precision,
            dim=self.dim,
            n_samples=self.n_samples,
            shrinkage=self.shrinkage,
            model_id=self.model_id,
        )
        return path

    @classmethod
    def load(cls, path: Path) -> "LinearProbe":
        with np.load(path) as data:
            return cls(
                mean=data["mean"],
                precision=data["precision"],
                dim=int(data["dim"]),
                n_samples=int(data["n_samples"]),
                shrinkage=float(data["shrinkage"]),
                model_id=str(data["model_id"]),
            )


def fit(
    normal_activations: np.ndarray,
    *,
    shrinkage: float = DEFAULT_SHRINKAGE,
    model_id: str = "unknown",
) -> LinearProbe:
    """Fit the probe on normal patch activations only.

    ``normal_activations`` is ``(n_patches, dim)`` pooled across all normal training frames.
    """
    if normal_activations.ndim != 2:
        raise ProbeError(f"expected (n_patches, dim), got {normal_activations.shape}")
    if not 0.0 <= shrinkage < 1.0:
        raise ProbeError(f"shrinkage must be in [0, 1), got {shrinkage}")

    n_samples, dim = normal_activations.shape
    if n_samples < MIN_SAMPLES_PER_DIM * dim:
        raise ProbeError(
            f"{n_samples} samples for {dim} dimensions is too few to estimate a covariance; "
            f"want at least {int(MIN_SAMPLES_PER_DIM * dim)}"
        )

    samples = normal_activations.astype(np.float64)
    mean = samples.mean(axis=0)
    centred = samples - mean
    covariance = (centred.T @ centred) / (n_samples - 1)

    # Shrink toward a scaled identity: the sample covariance of a 768-dim Gaussian is
    # near-singular at realistic sample counts, and inverting it directly amplifies noise
    # in the low-variance directions — exactly where anomaly signal is easiest to fake.
    identity_scale = np.trace(covariance) / dim
    regularised = (1.0 - shrinkage) * covariance + shrinkage * identity_scale * np.eye(dim)

    try:
        precision = np.linalg.inv(regularised)
    except np.linalg.LinAlgError as exc:  # pragma: no cover - shrinkage makes this unlikely
        raise ProbeError(f"covariance is singular even after shrinkage: {exc}") from exc

    return LinearProbe(
        mean=mean,
        precision=precision,
        dim=dim,
        n_samples=n_samples,
        shrinkage=shrinkage,
        model_id=model_id,
    )
