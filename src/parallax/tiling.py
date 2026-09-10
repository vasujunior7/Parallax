"""Tile a full-resolution frame into backbone-sized windows, and stitch scores back.

Why this exists: VisA defects occupy 0.075%-0.78% of a frame. Resize a 1404x1070 image to
the backbone's 224px input and the median defect is 6-20px across, which is **under one
14px patch** for four of six rigid classes — there is no patch left to flag. Cropping
224x224 windows at native resolution instead keeps a ~30-55px defect at 2-4 patches, and
keeps the 224 input size that the BSF positional mean is computed for.

Cost is roughly 30 forward passes per frame rather than one.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

DEFAULT_TILE = 224
DEFAULT_OVERLAP = 0.25


@dataclass(frozen=True)
class Tile:
    """A window into the source frame, in source pixel coordinates."""

    x: int
    y: int
    size: int

    @property
    def slices(self) -> tuple[slice, slice]:
        return slice(self.y, self.y + self.size), slice(self.x, self.x + self.size)


def _axis_positions(extent: int, tile: int, stride: int) -> list[int]:
    """Start offsets covering ``extent``, with the final tile flush to the far edge.

    Clamping the last tile rather than emitting a partial one keeps every window exactly
    ``tile`` wide, so the backbone never sees a padded or resized input.
    """
    if extent <= tile:
        return [0]

    positions = list(range(0, extent - tile + 1, stride))
    if positions[-1] != extent - tile:
        positions.append(extent - tile)
    return positions


def plan_tiles(
    shape: tuple[int, int], *, tile: int = DEFAULT_TILE, overlap: float = DEFAULT_OVERLAP
) -> tuple[Tile, ...]:
    """Cover a frame of ``shape`` (height, width) with overlapping windows."""
    if not 0.0 <= overlap < 1.0:
        raise ValueError(f"overlap must be in [0, 1), got {overlap}")

    height, width = shape
    stride = max(1, int(round(tile * (1.0 - overlap))))
    return tuple(
        Tile(x=x, y=y, size=tile)
        for y in _axis_positions(height, tile, stride)
        for x in _axis_positions(width, tile, stride)
    )


def cut(image: np.ndarray, tiles: tuple[Tile, ...]) -> list[np.ndarray]:
    """Crop each window out of the frame. Frames smaller than a tile are padded once."""
    height, width = image.shape[:2]
    size = tiles[0].size

    if height < size or width < size:
        pad_y, pad_x = max(0, size - height), max(0, size - width)
        image = cv2.copyMakeBorder(
            image, 0, pad_y, 0, pad_x, cv2.BORDER_REFLECT_101
        )

    return [image[t.slices] for t in tiles]


def stitch(
    tile_maps: list[np.ndarray], tiles: tuple[Tile, ...], shape: tuple[int, int]
) -> np.ndarray:
    """Average per-tile score maps back onto a full-resolution canvas.

    ``tile_maps`` are patch-resolution scores (e.g. 16x16 for a 224px tile at patch 14).
    Each is upsampled to tile size before placement, so the returned map is at **source
    resolution** and directly comparable to a ground-truth mask. Overlapping regions are
    averaged, which also softens tile-boundary artefacts.
    """
    if len(tile_maps) != len(tiles):
        raise ValueError(f"{len(tile_maps)} score maps for {len(tiles)} tiles")

    height, width = shape
    size = tiles[0].size
    canvas = np.zeros((max(height, size), max(width, size)), dtype=np.float32)
    counts = np.zeros_like(canvas)

    for scores, tile in zip(tile_maps, tiles):
        upsampled = cv2.resize(
            scores.astype(np.float32), (size, size), interpolation=cv2.INTER_LINEAR
        )
        canvas[tile.slices] += upsampled
        counts[tile.slices] += 1.0

    return (canvas / np.maximum(counts, 1.0))[:height, :width]
