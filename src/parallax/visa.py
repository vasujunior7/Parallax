"""VisA ingest.

VisA (Visual Anomaly, Amazon Science) — 10,821 images over 12 object classes with
pixel-level anomaly masks, CC BY 4.0. Chosen over MVTec AD because MVTec's CC BY-NC-SA
terms conflict with the licence competition entrants grant the organisers.

The archive is 1.8 GiB and lives outside the repository, under data/ (gitignored).
"""
from __future__ import annotations

import csv
import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

VISA_URL = "https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar"
VISA_TAR_BYTES = 1_929_840_640

DEFAULT_DATA_DIR = Path("data/visa")

# Homography registration against a golden reference assumes a rigid, near-planar part.
# That holds for these classes and does not hold for the deformable ones, so the
# align-and-difference pipeline is scoped to the rigid subset and the split is explicit
# rather than buried in a config file. See proposal.md, Section 2.
RIGID_CLASSES = ("pcb1", "pcb2", "pcb3", "pcb4", "capsules", "candle")
DEFORMABLE_CLASSES = (
    "cashew",
    "chewinggum",
    "fryum",
    "pipe_fryum",
    "macaroni1",
    "macaroni2",
)
ALL_CLASSES = RIGID_CLASSES + DEFORMABLE_CLASSES


@dataclass(frozen=True)
class Sample:
    """One VisA image, with its mask when the image is anomalous."""

    object_class: str
    split: str  # "train" | "test"
    label: str  # "normal" | "anomaly"
    image: Path
    mask: Path | None

    @property
    def is_anomalous(self) -> bool:
        return self.label != "normal"

    @property
    def is_rigid(self) -> bool:
        return self.object_class in RIGID_CLASSES


def _report(done: int, total: int) -> None:
    pct = 100 * done / total if total else 0
    print(f"\r  {done / 1048576:8.1f} / {total / 1048576:.1f} MiB ({pct:5.1f}%)", end="")


def download(data_dir: Path = DEFAULT_DATA_DIR, *, url: str = VISA_URL) -> Path:
    """Fetch the VisA tar if it is not already present at the expected size."""
    data_dir.mkdir(parents=True, exist_ok=True)
    archive = data_dir / Path(url).name

    if archive.is_file() and archive.stat().st_size == VISA_TAR_BYTES:
        return archive

    with urllib.request.urlopen(url) as response:  # noqa: S310 - pinned https URL
        total = int(response.headers.get("Content-Length", 0))
        done = 0
        with archive.open("wb") as handle:
            while chunk := response.read(1 << 20):
                handle.write(chunk)
                done += len(chunk)
                _report(done, total)
    print()
    return archive


def extract(archive: Path, data_dir: Path = DEFAULT_DATA_DIR) -> Path:
    """Extract the archive and return the dataset root.

    The tar has no wrapping directory: the twelve class folders and split_csv/ sit at the
    archive root, so the dataset root is data_dir itself. Uses the 'data' filter to refuse
    path-traversal members.
    """
    if (data_dir / ALL_CLASSES[0]).is_dir():
        return data_dir

    with tarfile.open(archive) as tar:
        tar.extractall(data_dir, filter="data")
    return data_dir


def split_csv_path(root: Path, *, setup: str = "1cls") -> Path:
    """The official split, which ships inside the archive under split_csv/."""
    return root / "split_csv" / f"{setup}.csv"


def load_split(split_csv: Path, root: Path, *, rigid_only: bool = False) -> tuple[Sample, ...]:
    """Read the split CSV into Samples with resolved paths."""
    samples = []
    with split_csv.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            object_class = row["object"]
            if rigid_only and object_class not in RIGID_CLASSES:
                continue
            mask = row.get("mask") or ""
            samples.append(
                Sample(
                    object_class=object_class,
                    split=row["split"],
                    label=row["label"],
                    image=root / row["image"],
                    mask=root / mask if mask else None,
                )
            )
    return tuple(samples)


def verify(samples: tuple[Sample, ...], *, check: int = 50) -> None:
    """Fail loudly if indexed paths do not exist on disk.

    Counting CSV rows succeeds even when the dataset root is wrong, which silently yields
    thousands of dangling paths. This spot-checks resolution instead of trusting it.
    """
    if not samples:
        raise FileNotFoundError("split file produced no samples")

    step = max(1, len(samples) // check)
    for sample in samples[::step]:
        if not sample.image.is_file():
            raise FileNotFoundError(f"indexed image is missing on disk: {sample.image}")
        if sample.mask is not None and not sample.mask.is_file():
            raise FileNotFoundError(f"indexed mask is missing on disk: {sample.mask}")


def ingest(data_dir: Path = DEFAULT_DATA_DIR, *, rigid_only: bool = False) -> tuple[Sample, ...]:
    """Download, extract, and index VisA. Safe to re-run; each step is idempotent."""
    archive = download(data_dir)
    root = extract(archive, data_dir)
    samples = load_split(split_csv_path(root), root, rigid_only=rigid_only)
    verify(samples)
    return samples


if __name__ == "__main__":
    samples = ingest()
    anomalous = sum(s.is_anomalous for s in samples)
    rigid = sum(s.is_rigid for s in samples)
    print(f"samples    : {len(samples)}")
    print(f"anomalous  : {anomalous}")
    print(f"rigid      : {rigid}  (align-and-difference scope)")
    print(f"deformable : {len(samples) - rigid}  (stated limitation)")
