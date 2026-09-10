"""Golden reference construction and z-score deviation."""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from parallax.inspect import measure
from parallax.reference import (
    MAD_FLOOR,
    GoldenReference,
    ReferenceError,
    build_from_arrays,
)

RNG = np.random.default_rng(20260909)
FRAME = (120, 160)


def _normal_frames(n: int = 24, *, noise: float = 3.0) -> list[np.ndarray]:
    """A repeatable scene plus sensor noise — what the training split looks like."""
    base = np.full(FRAME, 60, dtype=np.uint8)
    cv2.rectangle(base, (40, 30), (110, 80), 200, thickness=-1)
    cv2.circle(base, (30, 95), 10, 150, thickness=-1)

    frames = []
    for _ in range(n):
        noisy = base.astype(np.float32) + RNG.normal(0, noise, FRAME)
        frames.append(np.clip(noisy, 0, 255).astype(np.uint8))
    return frames


@pytest.fixture
def reference() -> GoldenReference:
    return build_from_arrays(_normal_frames(), object_class="synthetic")


class TestBuild:
    def test_records_source_count_and_shape(self, reference):
        assert reference.n_sources == 24
        assert reference.shape == FRAME

    def test_median_recovers_the_underlying_scene(self, reference):
        # Component interior is 200, substrate is 60; noise must not move them far.
        assert reference.median[50, 70] == pytest.approx(200, abs=3)
        assert reference.median[10, 10] == pytest.approx(60, abs=3)

    def test_mad_never_drops_below_the_floor(self, reference):
        assert reference.mad.min() >= MAD_FLOOR

    def test_too_few_frames_is_refused(self):
        with pytest.raises(ReferenceError, match="at least"):
            build_from_arrays(_normal_frames(3), object_class="synthetic")

    def test_inconsistent_sizes_are_refused(self):
        frames = _normal_frames(10)
        frames[4] = cv2.resize(frames[4], (80, 60))
        with pytest.raises(ReferenceError, match="inconsistent frame sizes"):
            build_from_arrays(frames, object_class="synthetic")


class TestZScore:
    def test_a_normal_frame_stays_near_zero(self, reference):
        z = reference.z_score(_normal_frames(1)[0])
        assert z.mean() < 1.5

    def test_a_defect_stands_out_above_background(self, reference):
        defective = _normal_frames(1)[0].copy()
        cv2.rectangle(defective, (60, 45), (85, 65), 0, thickness=-1)

        z = reference.z_score(defective)
        inside = z[45:65, 60:85].mean()
        outside = np.delete(z.ravel(), np.s_[:0])  # whole frame as a coarse background
        assert inside > 4 * outside.mean()

    def test_wrong_frame_size_is_refused(self, reference):
        with pytest.raises(ReferenceError, match="does not match reference"):
            reference.z_score(np.zeros((10, 10), dtype=np.uint8))

    def test_mask_feeds_the_existing_measure_stage(self, reference):
        defective = _normal_frames(1)[0].copy()
        cv2.rectangle(defective, (60, 45), (85, 65), 0, thickness=-1)

        defects = measure(reference.defect_mask(defective), min_area_px=25)

        assert defects
        cx, cy = defects[0].centroid
        assert 55 < cx < 90 and 40 < cy < 70

    def test_clean_frame_yields_no_measured_defect(self, reference):
        defects = measure(reference.defect_mask(_normal_frames(1)[0]), min_area_px=25)
        assert not defects


class TestPersistence:
    def test_round_trips_through_disk(self, reference, tmp_path):
        path = reference.save(tmp_path / "ref.npz")
        loaded = GoldenReference.load(path)

        assert loaded.object_class == reference.object_class
        assert loaded.n_sources == reference.n_sources
        np.testing.assert_allclose(loaded.median, reference.median)
        np.testing.assert_allclose(loaded.mad, reference.mad)
