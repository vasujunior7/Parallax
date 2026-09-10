"""Frozen backbone configuration and activation normalisation.

Holds the details that differ between DINOv2 and DINOv3 so the rest of the pipeline does
not hardcode them, plus the two device and normalisation rules that are easy to get wrong
and expensive to debug on a remote box.

The forward pass itself lives behind ``extract_patch_tokens`` and is the only part that
needs the GPU; everything else here is testable without a model.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# --- backbone variants -------------------------------------------------------------

@dataclass(frozen=True)
class BackboneSpec:
    """Everything that changes when the backbone changes."""

    model_id: str
    patch_size: int
    dim: int
    n_prefix_tokens: int  # CLS, plus register tokens where present
    gated: bool

    def grid_side(self, image_size: int) -> int:
        """Patches per side for a square input. Must divide exactly."""
        if image_size % self.patch_size:
            raise ValueError(
                f"input {image_size} is not a multiple of patch size {self.patch_size}"
            )
        return image_size // self.patch_size

    def n_patches(self, image_size: int) -> int:
        return self.grid_side(image_size) ** 2


# Fallback and licence-clean comparison point. Chosen over plain dinov2-base because its
# token layout (CLS + 4 registers) is identical to DINOv3's, so the swap is a config
# change; registers also absorb the high-norm artefact tokens that would otherwise appear
# in a patch map and read exactly like anomalies.
# Note: this checkpoint declares image_size 518, but we feed 224px tiles — always derive
# the patch count from the actual input via n_patches(), never from the config default.
DINOV2 = BackboneSpec(
    model_id="facebook/dinov2-with-registers-base",
    patch_size=14,
    dim=768,
    n_prefix_tokens=5,
    gated=False,
)

# Default. The backbone the Block-Sparse Featurizer was validated against, which is why the
# shipped bsf/pos_mean.npy — (196, 768), i.e. this grid and dim — applies to us directly.
# Access-gated by Meta; approval granted 2026-09-10. Values verified against the published
# config.json: patch_size 16, hidden_size 768, num_register_tokens 4 (+1 CLS = 5 prefix).
DINOV3 = BackboneSpec(
    model_id="facebook/dinov3-vitb16-pretrain-lvd1689m",
    patch_size=16,
    dim=768,
    n_prefix_tokens=5,
    gated=True,
)

DEFAULT = DINOV3

BACKBONES = {"dinov2": DINOV2, "dinov3": DINOV3}


# --- device --------------------------------------------------------------------------

# Ampere (SM 8.0) introduced bfloat16. The DGX V100s are Volta (SM 7.0) and have fp16
# tensor cores but no bf16, so a hardcoded bfloat16 autocast dies there while working fine
# on any newer development GPU. Always ask the device.
BF16_MIN_CAPABILITY = (8, 0)


def select_dtype(capability: tuple[int, int] | None):
    """Pick the widest half precision the device actually supports.

    ``capability`` is ``torch.cuda.get_device_capability()``, or None for CPU.
    """
    import torch

    if capability is None:
        return torch.float32
    return torch.bfloat16 if capability >= BF16_MIN_CAPABILITY else torch.float16


# --- activation normalisation --------------------------------------------------------

# The featurizers are written against this convention and their sparsity thresholds assume
# it: centre by the per-patch-position mean, then scale so the mean squared activation norm
# equals d. This is applied to raw patch tokens and is SEPARATE from the ImageNet mean/std
# the image processor applies to pixels. Both happen, in that order.

def centre_and_scale(
    activations: np.ndarray, position_mean: np.ndarray | None = None
) -> np.ndarray:
    """Apply the BSF input convention to ``(n_images, n_patches, dim)`` activations.

    Returns a flattened ``(n_images * n_patches, dim)`` matrix, which is what
    ``bsf.BSF.encode`` consumes.
    """
    if activations.ndim != 3:
        raise ValueError(f"expected (n_images, n_patches, dim), got {activations.shape}")

    centred = activations.astype(np.float32)
    if position_mean is not None:
        if position_mean.shape != activations.shape[1:]:
            raise ValueError(
                f"position mean {position_mean.shape} does not match "
                f"(n_patches, dim) = {activations.shape[1:]}"
            )
        centred = centred - position_mean

    flat = centred.reshape(-1, centred.shape[-1])
    rms = np.sqrt((flat**2).sum(axis=1).mean())
    if rms == 0:
        raise ValueError("activations are all zero; nothing to scale")
    return flat / rms * np.sqrt(flat.shape[1])


def drop_prefix_tokens(tokens: np.ndarray, spec: BackboneSpec) -> np.ndarray:
    """Strip CLS and register tokens, leaving only patch tokens.

    Getting this count wrong does not error — it silently shifts every patch, so the score
    map is offset from the image and localisation quietly fails.
    """
    if tokens.ndim != 3:
        raise ValueError(f"expected (batch, tokens, dim), got {tokens.shape}")
    if tokens.shape[1] <= spec.n_prefix_tokens:
        raise ValueError(
            f"{tokens.shape[1]} tokens is not more than "
            f"{spec.n_prefix_tokens} prefix tokens"
        )
    return tokens[:, spec.n_prefix_tokens :, :]
