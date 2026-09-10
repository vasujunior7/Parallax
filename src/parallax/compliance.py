"""Runtime version compliance.

The submission claims OpenCV 5 is a functional dependency, not a nominal one: the
alignment stage uses LightGlue with ALIKED/DISK features, which do not exist in 4.x.
This module proves that at runtime. `banner()` output goes in the CI log and the demo
video; `assert_compliant()` fails the process rather than letting a 4.x install run a
silently degraded pipeline.
"""
from __future__ import annotations

import cv2

MIN_MAJOR = 5

# Symbols that exist only in OpenCV 5. Their presence is the compliance evidence.
OPENCV5_SYMBOLS = ("LightGlueMatcher", "ALIKED", "DISK")


class ComplianceError(RuntimeError):
    """Raised when the runtime OpenCV cannot support the claimed pipeline."""


def opencv_version() -> tuple[int, int, int]:
    major, minor, patch = cv2.__version__.split(".")[:3]
    return int(major), int(minor), int(patch)


def missing_symbols() -> list[str]:
    return [name for name in OPENCV5_SYMBOLS if not hasattr(cv2, name)]


def assert_compliant() -> None:
    """Raise ComplianceError unless this really is OpenCV 5 with the features we claim."""
    major = opencv_version()[0]
    if major < MIN_MAJOR:
        raise ComplianceError(
            f"OpenCV {cv2.__version__} found, {MIN_MAJOR}.x required. "
            "The alignment stage needs LightGlue/ALIKED, which 4.x does not provide."
        )

    missing = missing_symbols()
    if missing:
        raise ComplianceError(
            f"OpenCV {cv2.__version__} is missing {', '.join(missing)}. "
            "This build cannot run the claimed alignment stage."
        )


def banner() -> str:
    """One-line provenance string for the CI log and the demo video."""
    present = [name for name in OPENCV5_SYMBOLS if hasattr(cv2, name)]
    status = ", ".join(present) if present else "none"
    return f"OpenCV {cv2.__version__} | OpenCV 5 features present: {status}"


if __name__ == "__main__":
    print(banner())
    assert_compliant()
    print("compliance: OK")
