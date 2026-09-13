"""Parallax — inspection that refuses to answer when it should not."""

from parallax.agent import (
    AgentConfig,
    AgentResult,
    AgentState,
    Decision,
    DEFAULT_CONFIG,
    decide,
)
from parallax.align import Alignment, AlignmentError, align_to_reference
from parallax.compliance import ComplianceError, assert_compliant, banner
from parallax.inspect import Defect, Verdict, defect_mask, inspect, measure
from parallax.reference import GoldenReference, ReferenceError, build_from_arrays
from parallax.trace import TraceStore

__all__ = [
    # agent
    "AgentConfig",
    "AgentResult",
    "AgentState",
    "Decision",
    "DEFAULT_CONFIG",
    "decide",
    # align
    "Alignment",
    "AlignmentError",
    "align_to_reference",
    # compliance
    "ComplianceError",
    "assert_compliant",
    "banner",
    # inspect
    "Defect",
    "Verdict",
    "defect_mask",
    "inspect",
    "measure",
    # reference
    "GoldenReference",
    "ReferenceError",
    "build_from_arrays",
    # trace
    "TraceStore",
]
