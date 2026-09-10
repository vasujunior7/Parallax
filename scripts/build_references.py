"""Build a golden reference per rigid VisA class and score it against ground truth.

Run: python scripts/build_references.py

Reports, per class, how well the z-score deviation map localises real defects:
  hit rate  - fraction of anomalies whose largest detected blob overlaps the true mask
  IoU       - overlap between the thresholded deviation map and the annotated mask
Normal test images are scored too, because a detector that fires on everything gets a
perfect hit rate and is useless.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from parallax.reference import DEFAULT_Z_THRESHOLD, GoldenReference, build
from parallax.visa import RIGID_CLASSES, ingest

REFERENCE_DIR = Path("data/references")
N_TRAIN = 80
N_TEST = 40


MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))


def _largest_blob(mask: np.ndarray) -> np.ndarray:
    """Clean the mask and keep only its largest connected component."""
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


def score_class(reference: GoldenReference, anomalies, normals) -> dict[str, float]:
    hits, ious, z_in, z_out = 0, [], [], []

    for sample in anomalies:
        image = cv2.imread(str(sample.image), cv2.IMREAD_GRAYSCALE)
        truth = cv2.imread(str(sample.mask), cv2.IMREAD_GRAYSCALE) > 0
        if truth.sum() == 0:
            continue

        z = reference.z_score(image)
        predicted = _largest_blob(z > DEFAULT_Z_THRESHOLD)

        z_in.append(float(z[truth].mean()))
        z_out.append(float(z[~truth].mean()))
        union = (predicted | truth).sum()
        ious.append(float((predicted & truth).sum() / union) if union else 0.0)
        # Deliberately scored against the single largest blob, i.e. the region the pipeline
        # would actually report. Asking whether *any* flagged pixel touches the mask scores
        # ~1.00 for free, because the threshold already flags 2-3% of the frame.
        hits += bool((predicted & truth).any())

    false_alarm = [
        float((reference.z_score(cv2.imread(str(s.image), cv2.IMREAD_GRAYSCALE))
               > DEFAULT_Z_THRESHOLD).mean())
        for s in normals
    ]

    return {
        "n": len(ious),
        "hit_rate": hits / len(ious) if ious else 0.0,
        "iou": float(np.mean(ious)) if ious else 0.0,
        "z_defect": float(np.mean(z_in)) if z_in else 0.0,
        "z_bg": float(np.mean(z_out)) if z_out else 0.0,
        "normal_flagged_px": float(np.mean(false_alarm)) if false_alarm else 0.0,
    }


def main() -> None:
    samples = ingest(rigid_only=True)
    train, test = defaultdict(list), defaultdict(list)
    for sample in samples:
        (train if sample.split == "train" else test)[sample.object_class].append(sample)

    header = f"{'class':10s} {'n':>3s} {'hit':>6s} {'IoU':>6s} {'z_def':>6s} {'z_bg':>5s} {'FP%':>6s}"
    print(header)
    print("-" * len(header))

    for object_class in RIGID_CLASSES:
        reference = build(
            (s.image for s in train[object_class]),
            object_class=object_class,
            limit=N_TRAIN,
        )
        reference.save(REFERENCE_DIR / f"{object_class}.npz")

        anomalies = [s for s in test[object_class] if s.is_anomalous][:N_TEST]
        normals = [s for s in test[object_class] if not s.is_anomalous][:N_TEST]
        r = score_class(reference, anomalies, normals)

        print(
            f"{object_class:10s} {r['n']:3d} {r['hit_rate']:6.2f} {r['iou']:6.3f} "
            f"{r['z_defect']:6.2f} {r['z_bg']:5.2f} {100 * r['normal_flagged_px']:5.1f}%"
        )

    print(f"\nreferences written to {REFERENCE_DIR}/")


if __name__ == "__main__":
    main()
