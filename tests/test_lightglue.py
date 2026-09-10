"""LightGlue/ALIKED alignment, and the ORB baseline it is measured against.

These tests need the ONNX weights, which are downloaded on first run and cached under
data/models. They skip rather than fail when the weights cannot be fetched, so the suite
still runs offline.
"""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from parallax.align import AlignmentError, LightGlueWeights, align_to_reference

pytestmark = pytest.mark.slow


@pytest.fixture(scope="session")
def weights() -> LightGlueWeights:
    try:
        return LightGlueWeights.download()
    except Exception as exc:  # network, integrity, or disk failure
        pytest.skip(f"ONNX weights unavailable: {exc}")


def _shift_and_rotate(image: np.ndarray, *, angle: float, tx: int, ty: int) -> np.ndarray:
    h, w = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    matrix[0, 2] += tx
    matrix[1, 2] += ty
    return cv2.warpAffine(image, matrix, (w, h))


def _directional_shadow(image: np.ndarray) -> np.ndarray:
    """A hard lighting gradient across the frame — Scenario B's warehouse door."""
    gradient = np.linspace(0.25, 1.0, image.shape[1])[None, :, None]
    return (image.astype(np.float32) * gradient).astype(np.uint8)


def test_lightglue_recovers_pose(reference_board, weights):
    moved = _shift_and_rotate(reference_board, angle=3.0, tx=12, ty=-8)

    result = align_to_reference(moved, reference_board, backend="lightglue", weights=weights)

    assert result.backend == "lightglue"
    assert result.n_inliers > 100
    assert result.inlier_ratio > 0.9


def test_lightglue_keeps_more_correspondences_than_orb_under_shadow(
    reference_board, weights
):
    """The reason LightGlue is in the pipeline at all.

    Classical descriptors degrade sharply when illumination changes; learned ones hold.
    This is the comparison the technical report publishes.
    """
    shadowed = _directional_shadow(
        _shift_and_rotate(reference_board, angle=3.0, tx=12, ty=-8)
    )

    orb = align_to_reference(shadowed, reference_board, backend="orb")
    glue = align_to_reference(
        shadowed, reference_board, backend="lightglue", weights=weights
    )

    assert glue.n_matches > orb.n_matches
    assert glue.inlier_ratio > orb.inlier_ratio


def test_alignment_does_not_remove_the_illumination_difference(reference_board, weights):
    """Good geometry is not enough, which is the whole premise of the project.

    After a correct warp the residual is still dominated by the lighting change. No
    alignment backend can separate 'shadow' from 'defect' — that judgement belongs to the
    confidence sidecar, not to OpenCV.
    """
    shadowed = _directional_shadow(reference_board)

    result = align_to_reference(
        shadowed, reference_board, backend="lightglue", weights=weights
    )

    interior = (slice(60, -60), slice(60, -60))
    residual = cv2.absdiff(result.warped, reference_board)[interior].mean()
    assert result.inlier_ratio > 0.9, "alignment itself should still succeed"
    assert residual > 5.0, "the lighting difference must survive alignment"


def test_missing_weights_raise_rather_than_silently_falling_back(reference_board, tmp_path):
    absent = LightGlueWeights(detector=tmp_path / "no.onnx", matcher=tmp_path / "no2.onnx")

    with pytest.raises(AlignmentError, match="missing ONNX weights"):
        align_to_reference(
            reference_board, reference_board, backend="lightglue", weights=absent
        )
