"""Block-Sparse Featurizer (BSF) integration for Parallax.

This module wraps the vendored Grassmannian BSF to compute concept blocks
from DINOv3 patch tokens.
"""
from __future__ import annotations

import sys
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import torch

from parallax.backbone import centre_and_scale
from parallax.features import FrameFeatures

# Ensure the vendored package is importable
VENDOR_DIR = Path(__file__).resolve().parents[2] / "vendor"
BSF_DIR = VENDOR_DIR / "block-sparse-featurizer"
if str(BSF_DIR) not in sys.path:
    sys.path.append(str(BSF_DIR))

import bsf


@dataclass(frozen=True)
class BSFConcepts:
    """BSF concept activations for a frame."""
    norms: np.ndarray  # (n_tiles, n_patches, n_groups)
    coords: np.ndarray # (n_tiles, n_patches, n_groups, group_size)
    grid: int
    n_groups: int

    @property
    def n_tiles(self) -> int:
        return self.norms.shape[0]

    def tile_norm_maps(self) -> list[np.ndarray]:
        """Reshape block norms into per-tile square grids of shape (grid, grid, n_groups)."""
        return list(self.norms.reshape(self.n_tiles, self.grid, self.grid, self.n_groups))


class ParallaxBSF:
    """Wraps a trained BSF model to score Parallax frame features."""

    def __init__(self, model_path: Path | str, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = Path(model_path)
        
        state_dict = torch.load(self.model_path, map_location=self.device, weights_only=True)
        # Infer dimensions from B_raw: (n_groups, d, group_size)
        B_raw_shape = state_dict['B_raw'].shape
        n_groups, d, group_size = B_raw_shape
        l0 = 16 # Default L0, can be adjusted or inferred if saved
        
        self.model = bsf.GrassmannianBSF(d=d, n_groups=n_groups, group_size=group_size, l0=l0)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        self.pos_mean = np.load(BSF_DIR / "bsf" / "pos_mean.npy").astype(np.float32)

    @torch.inference_mode()
    def extract_concepts(self, features: FrameFeatures) -> BSFConcepts:
        """Process FrameFeatures through the BSF to get concept block norms and coordinates."""
        # 1. Normalize tokens: centre by pos_mean and scale to RMS=sqrt(d)
        flat_tokens = centre_and_scale(features.tokens, self.pos_mean)
        
        # 2. To tensor
        x_tensor = torch.as_tensor(flat_tokens, dtype=torch.float32, device=self.device)
        
        # 3. Encode to get z of shape (N, n_groups, group_size)
        z = self.model.encode(x_tensor)
        z_np = z.cpu().numpy()
        
        # 4. Compute block norms
        norms = np.linalg.norm(z_np, axis=-1)
        
        n_tiles, n_patches = features.tokens.shape[0], features.tokens.shape[1]
        
        norms_reshaped = norms.reshape(n_tiles, n_patches, self.model.n_groups)
        coords_reshaped = z_np.reshape(n_tiles, n_patches, self.model.n_groups, self.model.group_size)

        return BSFConcepts(
            norms=norms_reshaped,
            coords=coords_reshaped,
            grid=features.grid,
            n_groups=self.model.n_groups
        )
