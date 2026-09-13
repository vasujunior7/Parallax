"""Parallax agent loop — deterministic policy engine.

Receives the outputs of three heads (OpenCV verdict, Linear Probe OOD score, BSF block
residual) and maps them to one of three decisions: ACCEPT, RELOOK, or ESCALATE.

The agent is a **pure function** of its inputs plus a config dataclass.  There is no LLM,
no sampling, and no state mutation at decision time.  All thresholds have calibrated defaults
derived from the Stage 4 measured score distributions on VisA rigid classes.

Decision logic (evaluated in priority order):
  1. ESCALATE  — vision defect AND OOD score is high  (both signals agree)
  2. ESCALATE  — vision defect AND BSF residual is high
  3. RELOOK    — OOD uncertain, vision clean, retries remain
  4. RELOOK    — BSF uncertain, vision clean, retries remain
  5. ESCALATE  — retries exhausted but still uncertain
  6. ACCEPT    — everything within normal bounds

Example usage::

    from parallax.agent import AgentConfig, AgentState, decide, Decision
    from parallax.inspect import Verdict

    verdict = inspect(aligned, reference)
    state   = AgentState(
        verdict           = verdict,
        ood_score         = probe_score,
        bsf_residual      = bsf_mean_error,
        bsf_top_block_norm= top_norm,
        retry_count       = 0,
        object_class      = "pcb1",
    )
    result = decide(state)
    if result.decision == Decision.ESCALATE:
        ...
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

from parallax.inspect import Verdict


# ---------------------------------------------------------------------------
# Public API: Decision
# ---------------------------------------------------------------------------

class Decision(Enum):
    ACCEPT   = auto()
    RELOOK   = auto()
    ESCALATE = auto()


# ---------------------------------------------------------------------------
# AgentConfig — all tunable thresholds in one place
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AgentConfig:
    """Thresholds governing the three-decision policy.

    All thresholds are calibrated from the Stage 4 VisA rigid-class distributions.
    They are deliberately conservative: a false escalation costs a human a few seconds;
    a missed defect can cost a product recall.

    Attributes
    ----------
    ood_relook_threshold:
        Linear Probe Mahalanobis score above which we request a second capture.
        Default: 8.0  (~75th pct of normal image scores across the six rigid classes).
    ood_escalate_threshold:
        OOD score above which a vision-confirmed defect triggers an ESCALATE immediately.
        Default: 14.0  (~90th pct of anomaly image scores).
    bsf_relook_threshold:
        BSF mean patch reconstruction error above which we request a re-look.
        Default: 0.15  (normal mean ≈ 0.10, this is +50 %).
    bsf_escalate_threshold:
        BSF residual above which a vision-confirmed defect triggers an ESCALATE.
        Default: 0.25  (above the 90th pct of anomaly patch errors).
    max_retries:
        Maximum RELOOK attempts before a still-uncertain frame is force-escalated.
        Default: 2.
    """
    ood_relook_threshold:   float = 8.0
    ood_escalate_threshold: float = 14.0
    bsf_relook_threshold:   float = 0.15
    bsf_escalate_threshold: float = 0.25
    max_retries:            int   = 2

    def __post_init__(self) -> None:
        if self.ood_relook_threshold >= self.ood_escalate_threshold:
            raise ValueError(
                f"ood_relook_threshold ({self.ood_relook_threshold}) must be "
                f"< ood_escalate_threshold ({self.ood_escalate_threshold})"
            )
        if self.bsf_relook_threshold >= self.bsf_escalate_threshold:
            raise ValueError(
                f"bsf_relook_threshold ({self.bsf_relook_threshold}) must be "
                f"< bsf_escalate_threshold ({self.bsf_escalate_threshold})"
            )
        if self.max_retries < 0:
            raise ValueError(f"max_retries must be >= 0, got {self.max_retries}")


DEFAULT_CONFIG = AgentConfig()


# ---------------------------------------------------------------------------
# AgentState — snapshot of all head outputs for one frame
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AgentState:
    """All signals the policy needs to make one decision.

    Attributes
    ----------
    verdict:
        OpenCV inspection result (defect blobs + binary mask).
    ood_score:
        Linear Probe image-level Mahalanobis distance.  Larger = more OOD.
    bsf_residual:
        BSF mean patch reconstruction error (MSE) across all patches in the frame.
        Larger = the model cannot reconstruct this frame from its concept dictionary.
    bsf_top_block_norm:
        Maximum block norm across all patches.  The loudest concept activation.
        Diagnostic only — not used in the main policy, surfaced in the trace.
    retry_count:
        How many RELOOK decisions have already been issued for this frame.
    object_class:
        Which VisA (or live) class this frame belongs to.  Informational; per-class
        threshold overrides are the caller's responsibility (pass a non-default config).
    frame_id:
        Optional stable identifier for the frame (file path, timestamp, etc.).
        Used as a trace key.
    """
    verdict:            Verdict
    ood_score:          float
    bsf_residual:       float
    bsf_top_block_norm: float
    retry_count:        int
    object_class:       str
    frame_id:           Optional[str] = None


# ---------------------------------------------------------------------------
# AgentResult — decision + provenance
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AgentResult:
    """The output of one call to ``decide()``.

    Attributes
    ----------
    decision:
        One of Decision.ACCEPT, RELOOK, or ESCALATE.
    reason:
        Human-readable string explaining which rule fired.  Suitable for trace logs
        and the escalation UI.
    rule_index:
        1-indexed rule number that fired (matches the docstring numbering).
    state:
        The full input snapshot that produced this decision.
    config:
        The config used.  Stored so traces can be replayed with different thresholds.
    decision_id:
        UUID4 string assigned at decision time.  Unique audit key.
    timestamp_utc:
        Unix timestamp (float seconds since epoch, UTC) when decide() was called.
    """
    decision:    Decision
    reason:      str
    rule_index:  int
    state:       AgentState
    config:      AgentConfig
    decision_id: str  = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp_utc: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Core policy function
# ---------------------------------------------------------------------------

def decide(
    state:  AgentState,
    config: AgentConfig = DEFAULT_CONFIG,
) -> AgentResult:
    """Apply the deterministic policy to one frame's signals.

    The six rules are evaluated in priority order.  The first rule that matches
    determines the decision.  See module docstring for rule definitions.

    Parameters
    ----------
    state:
        Snapshot of all head outputs for this frame.
    config:
        Threshold configuration.  Defaults to :data:`DEFAULT_CONFIG`.

    Returns
    -------
    AgentResult
        Decision, the reason string, the matching rule index, and full provenance.
    """
    def _result(decision: Decision, reason: str, rule_index: int) -> AgentResult:
        return AgentResult(
            decision   = decision,
            reason     = reason,
            rule_index = rule_index,
            state      = state,
            config     = config,
        )

    v   = state.verdict
    ood = state.ood_score
    bsf = state.bsf_residual
    n   = state.retry_count
    cls = state.object_class

    # Rule 1 — vision defect + OOD both high → escalate immediately
    if v.is_defective and ood >= config.ood_escalate_threshold:
        return _result(
            Decision.ESCALATE,
            f"[{cls}] Vision found {len(v.defects)} defect(s) "
            f"(area={v.total_area_px:.0f} px) AND OOD score {ood:.2f} >= "
            f"ood_escalate_threshold {config.ood_escalate_threshold:.2f}. "
            "Both signals agree: escalating to human review.",
            rule_index=1,
        )

    # Rule 2 — vision defect + BSF residual high → escalate
    if v.is_defective and bsf >= config.bsf_escalate_threshold:
        return _result(
            Decision.ESCALATE,
            f"[{cls}] Vision found {len(v.defects)} defect(s) "
            f"(area={v.total_area_px:.0f} px) AND BSF residual {bsf:.4f} >= "
            f"bsf_escalate_threshold {config.bsf_escalate_threshold:.4f}. "
            "BSF reconstruction confirms the anomaly: escalating.",
            rule_index=2,
        )

    # Rule 3 — OOD uncertain, vision clean, retries remain → relook
    if ood >= config.ood_relook_threshold and n < config.max_retries:
        return _result(
            Decision.RELOOK,
            f"[{cls}] Vision is clean but OOD score {ood:.2f} >= "
            f"ood_relook_threshold {config.ood_relook_threshold:.2f}. "
            f"Requesting re-capture (attempt {n + 1}/{config.max_retries}).",
            rule_index=3,
        )

    # Rule 4 — BSF uncertain, vision clean, retries remain → relook
    if bsf >= config.bsf_relook_threshold and n < config.max_retries:
        return _result(
            Decision.RELOOK,
            f"[{cls}] Vision is clean but BSF residual {bsf:.4f} >= "
            f"bsf_relook_threshold {config.bsf_relook_threshold:.4f}. "
            f"Requesting re-capture (attempt {n + 1}/{config.max_retries}).",
            rule_index=4,
        )

    # Rule 5 — retries exhausted but still uncertain → force escalate
    if n >= config.max_retries and (
        ood >= config.ood_relook_threshold or bsf >= config.bsf_relook_threshold
    ):
        return _result(
            Decision.ESCALATE,
            f"[{cls}] {n} re-look(s) exhausted and signal is still uncertain "
            f"(OOD={ood:.2f}, BSF={bsf:.4f}). Force-escalating to human review.",
            rule_index=5,
        )

    # Rule 6 — all signals within normal bounds → accept
    return _result(
        Decision.ACCEPT,
        f"[{cls}] Vision clean, OOD={ood:.2f} < {config.ood_relook_threshold:.2f}, "
        f"BSF={bsf:.4f} < {config.bsf_relook_threshold:.4f}. Accepted.",
        rule_index=6,
    )
