"""Backbone config, dtype selection, and the BSF input convention. No model needed."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from parallax.backbone import (
    BACKBONES,
    DEFAULT,
    DINOV2,
    DINOV3,
    centre_and_scale,
    drop_prefix_tokens,
    select_dtype,
)


class TestSpecs:
    def test_both_backbones_share_a_feature_dimension(self):
        """Why the swap is cheap: the featurizer sees the same d either way."""
        assert DINOV2.dim == DINOV3.dim == 768

    def test_both_drop_the_same_number_of_prefix_tokens(self):
        """dinov2-with-registers matches DINOv3's CLS + 4 registers layout."""
        assert DINOV2.n_prefix_tokens == DINOV3.n_prefix_tokens == 5

    def test_patch_sizes_differ_so_grids_differ(self):
        assert DINOV2.patch_size == 14
        assert DINOV3.patch_size == 16
        assert DINOV2.n_patches(224) == 256
        assert DINOV3.n_patches(224) == 196

    def test_dinov3_positional_mean_shape_matches_its_grid(self):
        """bsf/pos_mean.npy is (196, 768) — only valid at this grid and dim."""
        assert (DINOV3.n_patches(224), DINOV3.dim) == (196, 768)

    def test_non_divisible_input_is_refused(self):
        with pytest.raises(ValueError, match="multiple of patch size"):
            DINOV3.grid_side(518)

    def test_dinov3_is_the_default(self):
        """Matches the setting BSF was validated in, so pos_mean.npy applies directly."""
        assert DEFAULT is DINOV3

    def test_only_dinov3_is_gated(self):
        assert DINOV3.gated and not DINOV2.gated
        assert set(BACKBONES) == {"dinov2", "dinov3"}


class TestSelectDtype:
    def test_volta_gets_fp16_not_bf16(self):
        """The DGX V100s are SM 7.0. bfloat16 there is a crash, not a slowdown."""
        assert select_dtype((7, 0)) is torch.float16

    def test_ampere_and_later_get_bf16(self):
        assert select_dtype((8, 0)) is torch.bfloat16
        assert select_dtype((8, 9)) is torch.bfloat16  # this dev laptop

    def test_cpu_stays_fp32(self):
        assert select_dtype(None) is torch.float32


class TestCentreAndScale:
    def test_output_is_flattened_for_the_featurizer(self):
        activations = np.random.default_rng(0).normal(size=(4, 196, 768))
        assert centre_and_scale(activations).shape == (4 * 196, 768)

    def test_mean_squared_norm_equals_d(self):
        """The convention the BSF sparsity thresholds assume."""
        activations = np.random.default_rng(0).normal(size=(4, 196, 768))

        scaled = centre_and_scale(activations)

        assert (scaled**2).sum(axis=1).mean() == pytest.approx(768, rel=1e-4)

    def test_position_mean_is_subtracted(self):
        activations = np.full((3, 196, 768), 5.0)
        position_mean = np.full((196, 768), 5.0)

        with pytest.raises(ValueError, match="all zero"):
            centre_and_scale(activations, position_mean)

    def test_mismatched_position_mean_is_refused(self):
        """Guards the DINOv2/DINOv3 grid mismatch: 256 patches vs a (196, 768) mean."""
        activations = np.random.default_rng(0).normal(size=(2, 256, 768))
        dinov3_mean = np.zeros((196, 768))

        with pytest.raises(ValueError, match="does not match"):
            centre_and_scale(activations, dinov3_mean)

    def test_wrong_rank_is_refused(self):
        with pytest.raises(ValueError, match="expected"):
            centre_and_scale(np.zeros((196, 768)))


class TestDropPrefixTokens:
    def test_leaves_exactly_the_patch_tokens(self):
        tokens = np.zeros((2, 5 + 196, 768))
        assert drop_prefix_tokens(tokens, DINOV3).shape == (2, 196, 768)

    def test_keeps_patch_content_intact(self):
        tokens = np.arange(2 * (5 + 196) * 4, dtype=np.float32).reshape(2, 201, 4)
        np.testing.assert_array_equal(drop_prefix_tokens(tokens, DINOV3), tokens[:, 5:, :])

    def test_too_few_tokens_is_refused(self):
        with pytest.raises(ValueError, match="prefix tokens"):
            drop_prefix_tokens(np.zeros((1, 3, 768)), DINOV3)
