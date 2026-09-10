"""Register a captured frame to its golden reference.

Two backends behind one call:

- ``orb``       — classical ORB + BFMatcher. No model weights, always available. This is
                  the baseline the report benchmarks LightGlue against.
- ``lightglue`` — ALIKED/DISK features matched by LightGlue. Robust to the illumination
                  change that breaks classical matching, which is the case that matters
                  for inspection. Needs ONNX weights (see ``LightGlueWeights``).

Both return the same thing, so the rest of the pipeline does not know which ran.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

MIN_MATCHES = 4  # findHomography cannot solve with fewer
RANSAC_REPROJ_THRESHOLD = 5.0

# A homography can be "solved" from garbage correspondences: RANSAC returns a transform
# that is self-consistent among a handful of outliers. On VisA this produced inlier ratios
# of 0.02 with corner displacements over 1500 px on a 1400 px image, and the caller was
# told the alignment succeeded. These gate that.
MIN_INLIER_RATIO = 0.25
MAX_CORNER_SHIFT_FRACTION = 0.25  # of the image diagonal


class AlignmentError(RuntimeError):
    """Raised when the frame cannot be registered to the reference."""


@dataclass(frozen=True)
class LightGlueWeights:
    """Paths to the ONNX models OpenCV needs. It does not bundle these."""

    detector: Path
    matcher: Path

    def missing(self) -> list[Path]:
        return [p for p in (self.detector, self.matcher) if not p.is_file()]

    @classmethod
    def download(cls) -> "LightGlueWeights":
        """Fetch the ALIKED + ALIKED-LightGlue pair OpenCV is tested against."""
        from parallax.models import ensure_models

        paths = ensure_models("aliked", "aliked_lightglue")
        return cls(detector=paths["aliked"], matcher=paths["aliked_lightglue"])


@dataclass(frozen=True)
class Alignment:
    warped: np.ndarray
    homography: np.ndarray
    n_matches: int
    n_inliers: int
    backend: str

    @property
    def inlier_ratio(self) -> float:
        return self.n_inliers / self.n_matches if self.n_matches else 0.0


def _to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _orb_correspondences(
    frame: np.ndarray, reference: np.ndarray, max_features: int
) -> tuple[np.ndarray, np.ndarray]:
    orb = cv2.ORB_create(nfeatures=max_features)
    kp_frame, desc_frame = orb.detectAndCompute(_to_gray(frame), None)
    kp_ref, desc_ref = orb.detectAndCompute(_to_gray(reference), None)

    if desc_frame is None or desc_ref is None:
        raise AlignmentError("no ORB descriptors found in frame or reference")

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(matcher.match(desc_frame, desc_ref), key=lambda m: m.distance)

    src = np.float32([kp_frame[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst = np.float32([kp_ref[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    return src, dst


def _reject_implausible(
    homography: np.ndarray, n_inliers: int, n_matches: int, width: int, height: int
) -> None:
    """Refuse a transform that RANSAC solved but that cannot describe a real pose change.

    Differencing an unregistered frame reports the whole part as defective, so a bad warp
    is worse than no answer at all.
    """
    ratio = n_inliers / n_matches if n_matches else 0.0
    if ratio < MIN_INLIER_RATIO:
        raise AlignmentError(
            f"inlier ratio {ratio:.2f} below {MIN_INLIER_RATIO}: "
            f"{n_inliers}/{n_matches} correspondences agree"
        )

    corners = np.float32(
        [[0, 0], [width, 0], [width, height], [0, height]]
    ).reshape(-1, 1, 2)
    shift = float(
        np.linalg.norm(cv2.perspectiveTransform(corners, homography) - corners, axis=2).max()
    )
    limit = MAX_CORNER_SHIFT_FRACTION * float(np.hypot(width, height))
    if shift > limit:
        raise AlignmentError(
            f"corner displacement {shift:.0f}px exceeds {limit:.0f}px: "
            "the estimated transform is not a plausible pose change"
        )


def _keypoints_to_matrix(keypoints) -> np.ndarray:
    """LightGlue wants an Nx2 float matrix of x,y, not a KeyPoint vector."""
    return np.array([kp.pt for kp in keypoints], dtype=np.float32).reshape(-1, 2)


def _lightglue_correspondences(
    frame: np.ndarray, reference: np.ndarray, weights: LightGlueWeights
) -> tuple[np.ndarray, np.ndarray]:
    detector = cv2.ALIKED_create(str(weights.detector))
    kp_frame, desc_frame = detector.detectAndCompute(frame, None)
    kp_ref, desc_ref = detector.detectAndCompute(reference, None)

    if desc_frame is None or desc_ref is None or not kp_frame or not kp_ref:
        raise AlignmentError("ALIKED found no features in frame or reference")

    matcher = cv2.LightGlueMatcher_create(str(weights.matcher))
    # LightGlue needs spatial context, not just descriptors: without setPairInfo the
    # match call has no keypoint positions to reason about.
    matcher.setPairInfo(
        _keypoints_to_matrix(kp_frame),
        _keypoints_to_matrix(kp_ref),
        (frame.shape[1], frame.shape[0]),
        (reference.shape[1], reference.shape[0]),
    )
    matches = matcher.match(desc_frame, desc_ref)

    src = np.float32([kp_frame[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst = np.float32([kp_ref[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    return src, dst


def align_to_reference(
    frame: np.ndarray,
    reference: np.ndarray,
    *,
    backend: str = "orb",
    weights: LightGlueWeights | None = None,
    max_features: int = 2000,
) -> Alignment:
    """Warp ``frame`` onto ``reference``.

    Raises AlignmentError when registration fails — a failed alignment must stop the
    pipeline, because differencing an unregistered frame reports the whole part as a defect.
    """
    if backend == "orb":
        src, dst = _orb_correspondences(frame, reference, max_features)
    elif backend == "lightglue":
        if weights is None:
            raise AlignmentError("the lightglue backend requires ONNX weights")
        if missing := weights.missing():
            raise AlignmentError(
                f"missing ONNX weights: {', '.join(str(p) for p in missing)}"
            )
        src, dst = _lightglue_correspondences(frame, reference, weights)
    else:
        raise ValueError(f"unknown backend {backend!r}")

    if len(src) < MIN_MATCHES:
        raise AlignmentError(f"only {len(src)} matches, need at least {MIN_MATCHES}")

    homography, mask = cv2.findHomography(
        src, dst, cv2.RANSAC, RANSAC_REPROJ_THRESHOLD
    )
    if homography is None:
        raise AlignmentError("findHomography failed to find a consistent transform")

    height, width = reference.shape[:2]
    n_inliers = int(mask.sum()) if mask is not None else 0
    _reject_implausible(homography, n_inliers, len(src), width, height)

    warped = cv2.warpPerspective(frame, homography, (width, height))

    return Alignment(
        warped=warped,
        homography=homography,
        n_matches=len(src),
        n_inliers=int(mask.sum()) if mask is not None else 0,
        backend=backend,
    )
