"""Tiling geometry and score stitching. No model needed."""
from __future__ import annotations

import numpy as np
import pytest

from parallax.tiling import Tile, cut, plan_tiles, stitch

VISA_SHAPE = (1070, 1404)  # pcb1, height x width


class TestPlanTiles:
    def test_every_tile_is_inside_the_frame(self):
        for tile in plan_tiles(VISA_SHAPE):
            assert 0 <= tile.y and tile.y + tile.size <= VISA_SHAPE[0]
            assert 0 <= tile.x and tile.x + tile.size <= VISA_SHAPE[1]

    def test_tiles_cover_the_whole_frame(self):
        covered = np.zeros(VISA_SHAPE, dtype=bool)
        for tile in plan_tiles(VISA_SHAPE):
            covered[tile.slices] = True
        assert covered.all(), "tiling left gaps"

    def test_last_tile_is_flush_with_the_far_edge(self):
        tiles = plan_tiles(VISA_SHAPE)
        assert max(t.x + t.size for t in tiles) == VISA_SHAPE[1]
        assert max(t.y + t.size for t in tiles) == VISA_SHAPE[0]

    def test_frame_smaller_than_a_tile_yields_one_tile(self):
        assert len(plan_tiles((100, 120))) == 1

    def test_overlap_must_be_a_fraction(self):
        with pytest.raises(ValueError, match="overlap"):
            plan_tiles(VISA_SHAPE, overlap=1.0)

    def test_more_overlap_means_more_tiles(self):
        assert len(plan_tiles(VISA_SHAPE, overlap=0.5)) > len(
            plan_tiles(VISA_SHAPE, overlap=0.0)
        )


class TestCut:
    def test_all_crops_are_exactly_tile_sized(self):
        image = np.zeros((*VISA_SHAPE, 3), dtype=np.uint8)
        tiles = plan_tiles(VISA_SHAPE)

        for crop in cut(image, tiles):
            assert crop.shape[:2] == (224, 224), "backbone must never see a resized input"

    def test_small_frame_is_padded_not_resized(self):
        image = np.zeros((100, 120, 3), dtype=np.uint8)
        tiles = plan_tiles((100, 120))

        crops = cut(image, tiles)

        assert len(crops) == 1
        assert crops[0].shape[:2] == (224, 224)

    def test_crop_content_matches_its_source_window(self):
        rng = np.random.default_rng(0)
        image = rng.integers(0, 255, (*VISA_SHAPE, 3), dtype=np.uint8)
        tiles = plan_tiles(VISA_SHAPE)

        crops = cut(image, tiles)

        np.testing.assert_array_equal(crops[3], image[tiles[3].slices])


class TestStitch:
    def test_output_is_source_resolution_not_patch_resolution(self):
        tiles = plan_tiles(VISA_SHAPE)
        maps = [np.ones((16, 16), dtype=np.float32) for _ in tiles]

        assert stitch(maps, tiles, VISA_SHAPE).shape == VISA_SHAPE

    def test_uniform_scores_survive_stitching(self):
        tiles = plan_tiles(VISA_SHAPE)
        maps = [np.full((16, 16), 3.0, dtype=np.float32) for _ in tiles]

        result = stitch(maps, tiles, VISA_SHAPE)

        np.testing.assert_allclose(result, 3.0, rtol=1e-5)

    def test_a_hot_tile_lands_in_the_right_place(self):
        tiles = plan_tiles(VISA_SHAPE)
        target = tiles[5]
        maps = [np.zeros((16, 16), dtype=np.float32) for _ in tiles]
        maps[5] = np.full((16, 16), 10.0, dtype=np.float32)

        result = stitch(maps, tiles, VISA_SHAPE)

        centre = result[target.y + 112, target.x + 112]
        assert centre > 1.0, "hot tile did not appear at its own coordinates"

    def test_mismatched_counts_are_refused(self):
        tiles = plan_tiles(VISA_SHAPE)
        with pytest.raises(ValueError, match="score maps"):
            stitch([np.zeros((16, 16), dtype=np.float32)], tiles, VISA_SHAPE)


class TestDefectSurvivesTiling:
    """The reason tiling exists.

    A VisA defect is ~0.19% of the frame. Downscaling the whole frame to 224 leaves it
    under one 14px patch; cropping at native resolution keeps it several patches wide.
    """

    def test_defect_covers_multiple_patches_after_tiling(self):
        defect_fraction = 0.00185  # pcb1 median
        height, width = VISA_SHAPE
        defect_side = (defect_fraction * height * width) ** 0.5

        patches_if_resized = (defect_side * 224 / max(height, width)) / 14
        patches_if_tiled = defect_side / 14

        assert patches_if_resized < 1.0, "resizing should lose the defect"
        assert patches_if_tiled > 2.0, "tiling should preserve it"
