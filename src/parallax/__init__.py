"""Parallax — inspection that refuses to answer when it should not."""

from parallax.align import Alignment, AlignmentError, align_to_reference
from parallax.compliance import ComplianceError, assert_compliant, banner
from parallax.inspect import Defect, Verdict, defect_mask, inspect, measure
from parallax.reference import GoldenReference, ReferenceError, build_from_arrays

__all__ = [
    "Alignment",
    "AlignmentError",
    "align_to_reference",
    "ComplianceError",
    "assert_compliant",
    "banner",
    "Defect",
    "Verdict",
    "defect_mask",
    "inspect",
    "measure",
    "GoldenReference",
    "ReferenceError",
    "build_from_arrays",
]
