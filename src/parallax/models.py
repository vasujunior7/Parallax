"""ONNX weights for the OpenCV 5 feature stack.

OpenCV ships the ALIKED/DISK/LightGlue *code* but not the *weights*, and they are not in
opencv_zoo. These URLs and SHA-1 digests are lifted from OpenCV's own test manifest
(``opencv_extra/testdata/dnn/download_models.py``), so these are the exact artifacts the
library is tested against rather than arbitrary third-party exports.

Digests are verified on every fetch. A model that fails its digest is deleted, not used.
"""
from __future__ import annotations

import hashlib
import urllib.request
from dataclasses import dataclass
from pathlib import Path

DEFAULT_MODEL_DIR = Path("data/models")
CHUNK_BYTES = 1 << 20


@dataclass(frozen=True)
class ModelSpec:
    name: str
    filename: str
    url: str
    sha1: str


# Source: opencv_extra @ 5.x, testdata/dnn/download_models.py
MODELS: dict[str, ModelSpec] = {
    "aliked": ModelSpec(
        name="ALIKED (ONNX)",
        filename="aliked-n16rot-top1k-640.onnx",
        url="https://raw.githubusercontent.com/YangGuanyuhan/lightglue_opencv_project/main/model/aliked-n16rot-top1k-640.onnx",
        sha1="41faa7bf5d7eb68a2851471ba03aa20c9db30e4c",
    ),
    "aliked_lightglue": ModelSpec(
        name="ALIKED LightGlue (ONNX)",
        filename="aliked_lightglue.onnx",
        url="https://raw.githubusercontent.com/YangGuanyuhan/lightglue_opencv_project/main/model/aliked_lightglue.onnx",
        sha1="02723aa521990e57fe33d90b67977590c460e351",
    ),
    "disk": ModelSpec(
        name="DISK (ONNX)",
        filename="disk.onnx",
        url="https://github.com/fabio-sim/LightGlue-ONNX/releases/download/v0.1.0/disk.onnx",
        sha1="5f6a9069aed0af7302b67dcfb6d24b0d46707aec",
    ),
}


class ModelIntegrityError(RuntimeError):
    """Raised when a downloaded model does not match its expected digest."""


def sha1_of(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(spec: ModelSpec, model_dir: Path = DEFAULT_MODEL_DIR) -> Path:
    """Download ``spec`` if absent, verify its digest, and return the local path."""
    model_dir.mkdir(parents=True, exist_ok=True)
    destination = model_dir / spec.filename

    if destination.is_file() and sha1_of(destination) == spec.sha1:
        return destination

    urllib.request.urlretrieve(spec.url, destination)  # noqa: S310 - pinned https URLs

    actual = sha1_of(destination)
    if actual != spec.sha1:
        destination.unlink(missing_ok=True)
        raise ModelIntegrityError(
            f"{spec.name}: expected sha1 {spec.sha1}, got {actual}. Model discarded."
        )
    return destination


def ensure_models(*keys: str, model_dir: Path = DEFAULT_MODEL_DIR) -> dict[str, Path]:
    """Fetch the named models (default: all) and return {key: path}."""
    wanted = keys or tuple(MODELS)
    return {key: fetch(MODELS[key], model_dir) for key in wanted}


if __name__ == "__main__":
    for key, path in ensure_models().items():
        print(f"{key:18s} {path}  ({path.stat().st_size / 1048576:.1f} MiB, sha1 ok)")
