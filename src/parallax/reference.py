"""Golden references built from the normal training split.

A golden reference here is not one hand-picked "perfect" image. It is a per-pixel
**median** over many normal captures plus a per-pixel **MAD** (median absolute deviation)
describing how much that pixel normally varies. Deviation is then measured in units of
that variation — a robust z-score — rather than as an absolute grey-level difference.

Why this rather than warping each frame onto a reference image: VisA is captured on a
fixed rig, so frames arrive near-registered. Estimating a homography between two *different
physical instances* of a part does not correct pose, it invents one. Measured on VisA's
rigid classes, feature-based registration made the post-alignment residual worse
(candle 11.2 -> 18.3 mean abs) at inlier ratios of 0.02-0.39, while the statistical
reference below separates real defects from background by 2.2x-4.8x on five of six rigid
classes. Alignment stays available in ``align.py`` for the live-capture path, where the
camera really does move.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np

# Grey levels. In flat regions the MAD collapses toward zero and the z-score explodes:
# without a floor, background z reached 3665 on candle. This bounds sensor noise instead.
MAD_FLOOR = 2.0
MAD_TO_SIGMA = 1.4826  # makes MAD a consistent estimator of sigma for normal data

DEFAULT_Z_THRESHOLD = 4.0
MIN_SOURCES = 8


class ReferenceError(RuntimeError):
    """Raised when a reference cannot be built or does not fit the frame."""


def _load_grayscale(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ReferenceError(f"could not read image: {path}")
    return image


@dataclass(frozen=True)
class GoldenReference:
    """Per-pixel expectation and tolerance for one object class."""

    object_class: str
    median: np.ndarray  # float32, HxW
    mad: np.ndarray  # float32, HxW, floored at MAD_FLOOR
    n_sources: int

    @property
    def shape(self) -> tuple[int, int]:
        return self.median.shape[:2]

    def z_score(self, image: np.ndarray) -> np.ndarray:
        """Absolute deviation from the reference, in units of normal variation."""
        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if gray.shape[:2] != self.shape:
            raise ReferenceError(
                f"frame {gray.shape[:2]} does not match reference {self.shape}"
            )
        return np.abs(gray.astype(np.float32) - self.median) / self.mad

    def defect_mask(
        self, image: np.ndarray, *, z_threshold: float = DEFAULT_Z_THRESHOLD
    ) -> np.ndarray:
        """Binary mask of pixels deviating beyond ``z_threshold``.

        Returns a uint8 mask so it feeds ``inspect.measure`` unchanged.
        """
        return ((self.z_score(image) > z_threshold) * 255).astype(np.uint8)

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            object_class=self.object_class,
            median=self.median,
            mad=self.mad,
            n_sources=self.n_sources,
        )
        return path

    @classmethod
    def load(cls, path: Path) -> "GoldenReference":
        with np.load(path) as data:
            return cls(
                object_class=str(data["object_class"]),
                median=data["median"].astype(np.float32),
                mad=data["mad"].astype(np.float32),
                n_sources=int(data["n_sources"]),
            )


def build_from_arrays(
    images: Sequence[np.ndarray], *, object_class: str
) -> GoldenReference:
    """Build a reference from in-memory grayscale frames."""
    if len(images) < MIN_SOURCES:
        raise ReferenceError(
            f"{object_class}: need at least {MIN_SOURCES} normal frames, got {len(images)}"
        )

    shapes = {image.shape[:2] for image in images}
    if len(shapes) != 1:
        raise ReferenceError(f"{object_class}: inconsistent frame sizes {sorted(shapes)}")

    stack = np.stack(images)  # uint8; ~1.5 MB per 1070x1404 frame
    median = np.median(stack, axis=0).astype(np.float32)
    deviation = np.abs(stack.astype(np.float32) - median)
    mad = np.maximum(np.median(deviation, axis=0) * MAD_TO_SIGMA, MAD_FLOOR)

    return GoldenReference(
        object_class=object_class,
        median=median,
        mad=mad.astype(np.float32),
        n_sources=len(images),
    )


def build(
    paths: Iterable[Path], *, object_class: str, limit: int | None = 100
) -> GoldenReference:
    """Build a reference from image files.

    ``limit`` bounds peak memory: every frame is held at once to take a per-pixel median,
    so 100 frames of a 1070x1404 image is roughly 150 MB.
    """
    selected = list(paths)[:limit] if limit else list(paths)
    return build_from_arrays(
        [_load_grayscale(path) for path in selected], object_class=object_class
    )
