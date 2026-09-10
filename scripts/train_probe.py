"""Fit the confidence probe per rigid VisA class and score it against ground truth.

Run on the GPU box:
    CUDA_VISIBLE_DEVICES=7 ./.venv/bin/python scripts/train_probe.py

Trains only on normal frames. Scored with the same largest-blob metric used for the OpenCV
baseline in scripts/build_references.py, so the two numbers are directly comparable.
"""
from __future__ import annotations

import argparse
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from parallax.features import BackboneRunner
from parallax.probe import LinearProbe, fit
from parallax.tiling import stitch
from parallax.visa import RIGID_CLASSES, ingest

PROBE_DIR = Path("data/probes")
MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))


def _largest_blob(mask: np.ndarray) -> np.ndarray:
    cleaned = cv2.morphologyEx(
        cv2.morphologyEx((mask * 255).astype(np.uint8), cv2.MORPH_OPEN, MORPH_KERNEL),
        cv2.MORPH_CLOSE,
        MORPH_KERNEL,
    )
    count, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned)
    if count <= 1:
        return np.zeros(mask.shape, dtype=bool)
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == largest


def score_map(runner: BackboneRunner, probe: LinearProbe, image: np.ndarray) -> np.ndarray:
    """Per-pixel Mahalanobis score at source resolution."""
    features = runner.extract(image)
    flat = features.tokens.reshape(-1, features.tokens.shape[-1])
    return stitch(features.tile_maps(probe.score(flat)), features.tiles, features.shape)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=int, default=40, help="normal frames per class for fitting")
    parser.add_argument("--test", type=int, default=25, help="test frames per class per label")
    parser.add_argument("--quantile", type=float, default=0.99, help="score threshold quantile")
    args = parser.parse_args()

    samples = ingest(rigid_only=True)
    train, test = defaultdict(list), defaultdict(list)
    for sample in samples:
        (train if sample.split == "train" else test)[sample.object_class].append(sample)

    runner = BackboneRunner()
    print(runner.describe(), flush=True)
    print()

    header = f"{'class':10s} {'fit_px':>9s} {'hit':>5s} {'IoU':>6s} {'m_def':>7s} {'m_bg':>7s} {'ratio':>6s} {'AUROC':>6s}"
    print(header)
    print("-" * len(header))

    for object_class in RIGID_CLASSES:
        started = time.perf_counter()

        normals = [s.image for s in train[object_class][: args.train]]
        pooled = np.concatenate(
            [f.tokens.reshape(-1, f.tokens.shape[-1]) for f in runner.extract_paths(normals)]
        )
        probe = fit(pooled, model_id=runner.spec.model_id)
        probe.save(PROBE_DIR / f"{object_class}.npz")

        anomalies = [s for s in test[object_class] if s.is_anomalous][: args.test]
        clean = [s for s in test[object_class] if not s.is_anomalous][: args.test]

        # Threshold from normal test frames, so it never sees a defect.
        clean_scores = [score_map(runner, probe, cv2.imread(str(s.image))) for s in clean]
        threshold = float(np.quantile(np.concatenate([m.ravel() for m in clean_scores]), args.quantile))

        hits, ious, in_scores, out_scores, image_labels, image_maxima = 0, [], [], [], [], []
        for sample in anomalies:
            scores = score_map(runner, probe, cv2.imread(str(sample.image)))
            truth = cv2.imread(str(sample.mask), cv2.IMREAD_GRAYSCALE) > 0
            if truth.sum() == 0:
                continue

            predicted = _largest_blob(scores > threshold)
            union = (predicted | truth).sum()
            ious.append(float((predicted & truth).sum() / union) if union else 0.0)
            hits += bool((predicted & truth).any())
            in_scores.append(float(scores[truth].mean()))
            out_scores.append(float(scores[~truth].mean()))
            image_labels.append(1)
            image_maxima.append(float(scores.max()))

        for scores in clean_scores:
            image_labels.append(0)
            image_maxima.append(float(scores.max()))

        # Image-level AUROC via rank statistic — comparable to published VisA numbers.
        order = np.argsort(image_maxima)
        ranks = np.empty(len(order), dtype=np.float64)
        ranks[order] = np.arange(1, len(order) + 1)
        labels = np.array(image_labels)
        n_pos, n_neg = int(labels.sum()), int((1 - labels).sum())
        auroc = (
            (ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
            if n_pos and n_neg
            else float("nan")
        )

        mean_in, mean_out = float(np.mean(in_scores)), float(np.mean(out_scores))
        print(
            f"{object_class:10s} {probe.n_samples:9d} {hits / len(ious):5.2f} "
            f"{np.mean(ious):6.3f} {mean_in:7.1f} {mean_out:7.1f} "
            f"{mean_in / mean_out:6.2f} {auroc:6.3f}   ({time.perf_counter() - started:.0f}s)",
            flush=True,
        )

    print(f"\nprobes written to {PROBE_DIR}/")


if __name__ == "__main__":
    main()
