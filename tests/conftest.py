"""Synthetic PCB fixtures.

VisA is not vendored (10k images), so the pipeline is tested against a generated board.
The board needs real texture or ORB has nothing to match on, which is why components are
drawn rather than using flat noise.
"""
from __future__ import annotations

import cv2
import numpy as np
import pytest

BOARD_SIZE = (480, 640)  # h, w
SEED = 20260909
DEFECT_BOX = (300, 220, 40, 30)  # x, y, w, h


@pytest.fixture
def reference_board() -> np.ndarray:
    """A 'golden' board: dark substrate with bright rectangular and round components."""
    rng = np.random.default_rng(SEED)
    board = np.full((*BOARD_SIZE, 3), 30, dtype=np.uint8)

    for _ in range(40):
        x = int(rng.integers(20, BOARD_SIZE[1] - 60))
        y = int(rng.integers(20, BOARD_SIZE[0] - 60))
        w = int(rng.integers(12, 45))
        h = int(rng.integers(8, 30))
        colour = tuple(int(c) for c in rng.integers(120, 255, size=3))
        cv2.rectangle(board, (x, y), (x + w, y + h), colour, thickness=-1)

    for _ in range(25):
        centre = (
            int(rng.integers(20, BOARD_SIZE[1] - 20)),
            int(rng.integers(20, BOARD_SIZE[0] - 20)),
        )
        cv2.circle(board, centre, int(rng.integers(4, 11)), (200, 200, 200), thickness=-1)

    # Drawn last, so the component the defect removes is guaranteed present and bright.
    # Without this the "defect" can land on bare substrate, where blacking it out changes
    # the image by less than the detection threshold and nothing is found.
    x, y, w, h = DEFECT_BOX
    cv2.rectangle(board, (x, y), (x + w, y + h), (230, 230, 230), thickness=-1)

    return board


@pytest.fixture
def defect_location() -> tuple[int, int, int, int]:
    """x, y, w, h of the injected defect."""
    return DEFECT_BOX


@pytest.fixture
def defective_board(reference_board: np.ndarray, defect_location) -> np.ndarray:
    """The golden board with one component blacked out — a missing-component defect."""
    x, y, w, h = defect_location
    board = reference_board.copy()
    cv2.rectangle(board, (x, y), (x + w, y + h), (0, 0, 0), thickness=-1)
    return board
