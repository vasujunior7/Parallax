"""Week-1 acceptance: OpenCV 5 compliance, registration, and differencing."""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from parallax import align_to_reference, assert_compliant, banner, inspect
from parallax.align import AlignmentError
from parallax.compliance import missing_symbols, opencv_version

RNG = np.random.default_rng(20260909)


def _shift_and_rotate(image: np.ndarray, *, angle: float, tx: int, ty: int) -> np.ndarray:
    """Simulate a part sitting slightly off-pose under the camera."""
    h, w = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    matrix[0, 2] += tx
    matrix[1, 2] += ty
    return cv2.warpAffine(image, matrix, (w, h))


class TestCompliance:
    def test_runtime_is_opencv_5(self):
        assert opencv_version()[0] >= 5

    def test_lightglue_aliked_disk_all_present(self):
        assert missing_symbols() == []

    def test_assert_compliant_passes(self):
        assert_compliant()

    def test_banner_names_the_version_and_features(self):
        text = banner()
        assert cv2.__version__ in text
        assert "LightGlueMatcher" in text


class TestAlignment:
    def test_recovers_a_known_pose_offset(self, reference_board):
        moved = _shift_and_rotate(reference_board, angle=3.0, tx=12, ty=-8)

        result = align_to_reference(moved, reference_board)

        assert result.n_inliers >= 4
        assert result.inlier_ratio > 0.5
        assert result.homography.shape == (3, 3)

    def test_alignment_makes_the_frame_match_the_reference(self, reference_board):
        moved = _shift_and_rotate(reference_board, angle=3.0, tx=12, ty=-8)

        aligned = align_to_reference(moved, reference_board).warped

        # Compare interiors only: warping leaves undefined border regions.
        interior = (slice(60, -60), slice(60, -60))
        before = cv2.absdiff(moved, reference_board)[interior].mean()
        after = cv2.absdiff(aligned, reference_board)[interior].mean()
        assert after < before / 2

    def test_unmatchable_frame_raises_rather_than_guessing(self, reference_board):
        blank = np.zeros_like(reference_board)
        with pytest.raises(AlignmentError):
            align_to_reference(blank, reference_board)

    def test_lightglue_backend_reports_missing_weights(self, reference_board):
        with pytest.raises(AlignmentError, match="ONNX weights"):
            align_to_reference(reference_board, reference_board, backend="lightglue")

    def test_rejects_a_transform_solved_from_garbage_correspondences(self, reference_board):
        """Regression: RANSAC can solve a self-consistent homography from outliers.

        On VisA this returned inlier ratios of 0.02 with corner shifts over 1500px and
        reported success. Differencing against such a warp marks the whole part defective,
        so the alignment must refuse rather than answer.
        """
        noise = RNG.integers(0, 255, reference_board.shape, dtype=np.uint8)

        with pytest.raises(AlignmentError, match="inlier ratio|corner displacement"):
            align_to_reference(noise, reference_board)


class TestInspection:
    def test_identical_boards_report_no_defect(self, reference_board):
        verdict = inspect(reference_board, reference_board)
        assert not verdict.is_defective

    def test_missing_component_is_found_at_the_right_place(
        self, reference_board, defective_board, defect_location
    ):
        verdict = inspect(defective_board, reference_board)

        assert verdict.is_defective
        x, y, w, h = defect_location
        expected = (x + w / 2, y + h / 2)
        cx, cy = verdict.defects[0].centroid
        assert abs(cx - expected[0]) < 15
        assert abs(cy - expected[1]) < 15

    def test_defect_area_is_close_to_the_injected_area(
        self, reference_board, defective_board, defect_location
    ):
        _, _, w, h = defect_location
        verdict = inspect(defective_board, reference_board)
        assert verdict.defects[0].area_px == pytest.approx(w * h, rel=0.5)

    def test_mild_lighting_change_is_absorbed(self, reference_board):
        """CLAHE normalisation handles a modest gain without inventing defects."""
        brighter = cv2.convertScaleAbs(reference_board, alpha=1.15, beta=0)
        verdict = inspect(brighter, reference_board)
        assert not verdict.is_defective

    def test_strong_lighting_change_produces_false_defects(self, reference_board):
        """The limitation the confidence sidecar exists to catch.

        Beyond roughly +-15% gain, differencing against a golden reference reports
        illumination as damage. Nothing in the OpenCV stage can tell these apart from real
        defects — that judgement is the probe's job, and this test pins the boundary so a
        later change to the differencing stage cannot quietly move it.
        """
        much_brighter = cv2.convertScaleAbs(reference_board, alpha=1.30, beta=0)
        verdict = inspect(much_brighter, reference_board)
        assert verdict.is_defective, "expected illumination to be misread as defects"

    def test_end_to_end_offset_pose_with_a_defect(
        self, reference_board, defective_board, defect_location
    ):
        moved = _shift_and_rotate(defective_board, angle=2.0, tx=10, ty=6)

        aligned = align_to_reference(moved, reference_board).warped
        verdict = inspect(aligned, reference_board)

        assert verdict.is_defective
        x, y, w, h = defect_location
        cx, cy = verdict.defects[0].centroid
        assert abs(cx - (x + w / 2)) < 25
        assert abs(cy - (y + h / 2)) < 25
