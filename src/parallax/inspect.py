"""Difference an aligned frame against its golden reference and measure what changed.

This is the OpenCV verdict half of the system. It produces a defect mask and per-blob
geometry; it deliberately produces no confidence estimate, because that is the probe
sidecar's job and keeping them separate is what lets the agent weigh one against the other.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

DEFAULT_MIN_AREA_PX = 25.0
DEFAULT_THRESHOLD = 40
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID = (8, 8)
MORPH_KERNEL = (5, 5)


@dataclass(frozen=True)
class Defect:
    """One connected region that differs from the reference."""

    area_px: float
    centroid: tuple[float, float]
    bbox: tuple[int, int, int, int]  # x, y, w, h
    perimeter_px: float
    min_area_rect: tuple[tuple[float, float], tuple[float, float], float]

    def area_mm2(self, mm_per_px: float) -> float:
        """Physical area under a stated per-class scale factor.

        VisA ships no camera intrinsics, so mm_per_px is an assumption supplied by the
        caller, not a calibration result. See proposal.md, Section 2.
        """
        return self.area_px * mm_per_px**2


@dataclass(frozen=True)
class Verdict:
    defects: tuple[Defect, ...]
    mask: np.ndarray

    @property
    def is_defective(self) -> bool:
        return bool(self.defects)

    @property
    def total_area_px(self) -> float:
        return sum(d.area_px for d in self.defects)


def _normalise(image: np.ndarray) -> np.ndarray:
    """Grayscale + CLAHE, so a global lighting shift does not read as a defect."""
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID)
    return clahe.apply(gray)


def defect_mask(
    aligned: np.ndarray, reference: np.ndarray, *, threshold: int = DEFAULT_THRESHOLD
) -> np.ndarray:
    """Absolute difference -> threshold -> morphological open/close."""
    if aligned.shape[:2] != reference.shape[:2]:
        raise ValueError(
            f"aligned frame {aligned.shape[:2]} does not match reference {reference.shape[:2]}"
        )

    difference = cv2.absdiff(_normalise(aligned), _normalise(reference))
    _, binary = cv2.threshold(difference, threshold, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, MORPH_KERNEL)
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    return cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)


def measure(mask: np.ndarray, *, min_area_px: float = DEFAULT_MIN_AREA_PX) -> tuple[Defect, ...]:
    """Contour the mask and convert each surviving blob to geometry."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    defects = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area_px:
            continue

        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            continue

        defects.append(
            Defect(
                area_px=area,
                centroid=(moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]),
                bbox=cv2.boundingRect(contour),
                perimeter_px=cv2.arcLength(contour, closed=True),
                min_area_rect=cv2.minAreaRect(contour),
            )
        )

    return tuple(sorted(defects, key=lambda d: d.area_px, reverse=True))


def inspect(
    aligned: np.ndarray,
    reference: np.ndarray,
    *,
    threshold: int = DEFAULT_THRESHOLD,
    min_area_px: float = DEFAULT_MIN_AREA_PX,
) -> Verdict:
    """Full differencing pass over an already-aligned frame."""
    mask = defect_mask(aligned, reference, threshold=threshold)
    return Verdict(defects=measure(mask, min_area_px=min_area_px), mask=mask)
