"""Frozen-backbone patch feature extraction.

Nothing here is trained. Each frame is cut into native-resolution 224px tiles
(``tiling.py``), pushed through a frozen DINOv3/DINOv2 ViT, and the patch tokens are kept.
Those activations are the single input shared by all three downstream heads — linear probe,
BSF, and the SAE baseline — so this pass runs once and is cached.

Two device rules are enforced here rather than left to the caller:

* **dtype comes from device capability, never from ``is_bf16_supported()``.** On a V100
  (SM 7.0) that function returns True while bf16 measures 5.10 TFLOPS against fp16's 56.63
  — an 11x penalty, slower even than fp32, because Volta has no bf16 tensor cores.
* **attention uses PyTorch SDPA**, whose memory-efficient backend supports SM 7.0.
  FlashAttention-2 requires SM 8.0+ and is unavailable on the target hardware.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from parallax.backbone import DEFAULT, BackboneSpec, drop_prefix_tokens, select_dtype
from parallax.tiling import DEFAULT_OVERLAP, DEFAULT_TILE, Tile, cut, plan_tiles

DEFAULT_BATCH = 32


@dataclass(frozen=True)
class FrameFeatures:
    """Patch activations for one frame, plus the geometry needed to map them back."""

    tokens: np.ndarray  # (n_tiles, n_patches, dim), float32
    tiles: tuple[Tile, ...]
    shape: tuple[int, int]  # source frame (height, width)
    grid: int  # patches per tile side
    model_id: str

    @property
    def n_tiles(self) -> int:
        return self.tokens.shape[0]

    def tile_maps(self, patch_scores: np.ndarray) -> list[np.ndarray]:
        """Reshape flat per-patch scores back into per-tile square grids.

        ``patch_scores`` is ``(n_tiles * n_patches,)`` — the layout every head emits.
        """
        expected = self.n_tiles * self.grid**2
        if patch_scores.size != expected:
            raise ValueError(f"expected {expected} patch scores, got {patch_scores.size}")
        return list(patch_scores.reshape(self.n_tiles, self.grid, self.grid))


class BackboneRunner:
    """Loads a frozen backbone once and extracts patch tokens for many frames."""

    def __init__(
        self,
        spec: BackboneSpec = DEFAULT,
        *,
        device: str | None = None,
        tile: int = DEFAULT_TILE,
        overlap: float = DEFAULT_OVERLAP,
        batch_size: int = DEFAULT_BATCH,
    ) -> None:
        import torch
        from transformers import AutoImageProcessor, AutoModel

        self.spec = spec
        self.tile = tile
        self.overlap = overlap
        self.batch_size = batch_size
        self.grid = spec.grid_side(tile)

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        capability = (
            torch.cuda.get_device_capability(self.device)
            if self.device.startswith("cuda")
            else None
        )
        self.dtype = select_dtype(capability)

        self.processor = AutoImageProcessor.from_pretrained(spec.model_id)
        self.model = (
            AutoModel.from_pretrained(spec.model_id, attn_implementation="sdpa")
            .to(self.device, dtype=self.dtype)
            .eval()
        )
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def describe(self) -> str:
        return (
            f"{self.spec.model_id} | device={self.device} dtype={self.dtype} "
            f"| tile={self.tile} grid={self.grid}x{self.grid} "
            f"patches/tile={self.grid ** 2} dim={self.spec.dim}"
        )

    def _forward(self, crops: list[np.ndarray]) -> np.ndarray:
        import torch

        batch = self.processor(images=crops, return_tensors="pt", do_resize=False)
        pixel_values = batch["pixel_values"].to(self.device, dtype=self.dtype)

        with torch.inference_mode():
            hidden = self.model(pixel_values=pixel_values).last_hidden_state

        tokens = drop_prefix_tokens(hidden.float().cpu().numpy(), self.spec)

        expected = self.grid**2
        if tokens.shape[1] != expected:
            raise RuntimeError(
                f"{self.spec.model_id} returned {tokens.shape[1]} patch tokens, "
                f"expected {expected}. Check patch size and prefix-token count."
            )
        return tokens

    def extract(self, image: np.ndarray) -> FrameFeatures:
        """Tile a BGR frame, run the backbone, and return patch tokens per tile."""
        import cv2

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.ndim == 3 else cv2.cvtColor(
            image, cv2.COLOR_GRAY2RGB
        )
        tiles = plan_tiles(rgb.shape[:2], tile=self.tile, overlap=self.overlap)
        crops = cut(rgb, tiles)

        batches = [
            self._forward(crops[i : i + self.batch_size])
            for i in range(0, len(crops), self.batch_size)
        ]
        return FrameFeatures(
            tokens=np.concatenate(batches, axis=0),
            tiles=tiles,
            shape=rgb.shape[:2],
            grid=self.grid,
            model_id=self.spec.model_id,
        )

    def extract_paths(self, paths: list[Path], *, progress: bool = False) -> list[FrameFeatures]:
        import cv2

        out = []
        for index, path in enumerate(paths, 1):
            image = cv2.imread(str(path))
            if image is None:
                raise FileNotFoundError(f"could not read {path}")
            out.append(self.extract(image))
            if progress and (index % 25 == 0 or index == len(paths)):
                print(f"  {index}/{len(paths)} frames", flush=True)
        return out
