"""Tests for the agent loop (agent.py) and trace store (trace.py).

All tests run without GPU, without VisA data, and without any heavy imports.
They use synthetic Verdict objects built from a fixed-seed reference board.

Coverage:
  1. Policy correctness — all 6 rules fire as expected
  2. Threshold boundaries — exact threshold values trigger correctly (< vs >=)
  3. Retry saturation — RELOOK flips to ESCALATE at max_retries
  4. Config validation — bad threshold ordering raises ValueError
  5. Reason strings — each rule's reason mentions expected tokens
  6. TraceStore round-trip — write one row, read it back, verify every column
  7. TraceStore replay — frame_id lookup returns correct rows
  8. TraceStore rates — escalation_rate / accept_rate computed correctly
  9. TraceStore counts — decision_counts() matches manual totals
 10. Context manager — with-block closes store cleanly
 11. Multiple classes — class filter in rate functions
 12. Decision enum — Decision has exactly the three expected members
 13. AgentResult fields — decision_id is unique UUID4, timestamp is recent
"""
from __future__ import annotations

import time
import uuid
import math

import numpy as np
import pytest

from parallax.agent import (
    AgentConfig,
    AgentResult,
    AgentState,
    Decision,
    DEFAULT_CONFIG,
    decide,
)
from parallax.inspect import Defect, Verdict
from parallax.trace import TraceStore


# ---------------------------------------------------------------------------
# Helpers — build synthetic Verdicts without GPU
# ---------------------------------------------------------------------------

def _clean_verdict() -> Verdict:
    """A Verdict with no defects."""
    mask = np.zeros((64, 64), dtype=np.uint8)
    return Verdict(defects=(), mask=mask)


def _defective_verdict(area: float = 500.0) -> Verdict:
    """A Verdict with one synthetic defect blob."""
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[10:30, 10:35] = 255
    defect = Defect(
        area_px=area,
        centroid=(22.0, 20.0),
        bbox=(10, 10, 25, 20),
        perimeter_px=90.0,
        min_area_rect=((22.0, 20.0), (25.0, 20.0), 0.0),
    )
    return Verdict(defects=(defect,), mask=mask)


def _state(
    *,
    defective: bool = False,
    ood: float = 0.0,
    bsf: float = 0.0,
    top_norm: float = 0.0,
    retries: int = 0,
    cls: str = "pcb1",
    frame_id: str | None = None,
) -> AgentState:
    verdict = _defective_verdict() if defective else _clean_verdict()
    return AgentState(
        verdict=verdict,
        ood_score=ood,
        bsf_residual=bsf,
        bsf_top_block_norm=top_norm,
        retry_count=retries,
        object_class=cls,
        frame_id=frame_id,
    )


CFG = AgentConfig()  # default config


# ---------------------------------------------------------------------------
# 1. Policy correctness — all 6 rules
# ---------------------------------------------------------------------------

def test_rule1_escalate_vision_and_ood():
    """Defect + ood >= ood_escalate_threshold → ESCALATE (rule 1)."""
    r = decide(_state(defective=True, ood=CFG.ood_escalate_threshold))
    assert r.decision == Decision.ESCALATE
    assert r.rule_index == 1


def test_rule2_escalate_vision_and_bsf():
    """Defect + bsf >= bsf_escalate_threshold → ESCALATE (rule 2), OOD is below."""
    r = decide(_state(defective=True, ood=0.0, bsf=CFG.bsf_escalate_threshold))
    assert r.decision == Decision.ESCALATE
    assert r.rule_index == 2


def test_rule3_relook_ood_uncertain():
    """OOD >= relook threshold, no vision defect, retries remain → RELOOK (rule 3)."""
    r = decide(_state(defective=False, ood=CFG.ood_relook_threshold, retries=0))
    assert r.decision == Decision.RELOOK
    assert r.rule_index == 3


def test_rule4_relook_bsf_uncertain():
    """BSF >= relook threshold, no vision defect, OOD clean, retries remain → RELOOK (rule 4)."""
    r = decide(_state(defective=False, ood=0.0, bsf=CFG.bsf_relook_threshold, retries=0))
    assert r.decision == Decision.RELOOK
    assert r.rule_index == 4


def test_rule5_escalate_retries_exhausted():
    """Retries exhausted, OOD still high → ESCALATE (rule 5)."""
    r = decide(_state(defective=False, ood=CFG.ood_relook_threshold, retries=CFG.max_retries))
    assert r.decision == Decision.ESCALATE
    assert r.rule_index == 5


def test_rule5_escalate_retries_exhausted_via_bsf():
    """Retries exhausted, BSF still high → ESCALATE (rule 5)."""
    r = decide(_state(defective=False, bsf=CFG.bsf_relook_threshold, retries=CFG.max_retries))
    assert r.decision == Decision.ESCALATE
    assert r.rule_index == 5


def test_rule6_accept_all_clean():
    """Everything within bounds → ACCEPT (rule 6)."""
    r = decide(_state(defective=False, ood=0.0, bsf=0.0))
    assert r.decision == Decision.ACCEPT
    assert r.rule_index == 6


def test_rule6_accept_max_retries_but_clean():
    """Retries exhausted but both signals are clean → ACCEPT (rule 6, not rule 5)."""
    r = decide(_state(defective=False, ood=0.0, bsf=0.0, retries=CFG.max_retries))
    assert r.decision == Decision.ACCEPT
    assert r.rule_index == 6


# ---------------------------------------------------------------------------
# 2. Threshold boundaries — < vs >=
# ---------------------------------------------------------------------------

def test_ood_just_below_relook_threshold_accepts():
    ood = CFG.ood_relook_threshold - 0.001
    r = decide(_state(defective=False, ood=ood))
    assert r.decision == Decision.ACCEPT


def test_ood_at_relook_threshold_triggers_relook():
    r = decide(_state(defective=False, ood=CFG.ood_relook_threshold))
    assert r.decision == Decision.RELOOK


def test_ood_just_below_escalate_with_defect_uses_rule2():
    """OOD is between relook and escalate thresholds; defect present; BSF is at escalate."""
    ood = CFG.ood_escalate_threshold - 0.001
    bsf = CFG.bsf_escalate_threshold
    r = decide(_state(defective=True, ood=ood, bsf=bsf))
    # Rule 1 doesn't fire (OOD below escalate), rule 2 fires (BSF at escalate)
    assert r.decision == Decision.ESCALATE
    assert r.rule_index == 2


def test_bsf_just_below_relook_threshold_accepts():
    bsf = CFG.bsf_relook_threshold - 0.001
    r = decide(_state(defective=False, bsf=bsf))
    assert r.decision == Decision.ACCEPT


def test_bsf_at_relook_threshold_triggers_relook():
    r = decide(_state(defective=False, bsf=CFG.bsf_relook_threshold))
    assert r.decision == Decision.RELOOK


# ---------------------------------------------------------------------------
# 3. Retry saturation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("retries", [0, 1])
def test_relook_while_retries_remain(retries):
    cfg = AgentConfig(max_retries=2)
    r = decide(_state(defective=False, ood=cfg.ood_relook_threshold, retries=retries), cfg)
    assert r.decision == Decision.RELOOK


def test_escalate_when_max_retries_reached():
    cfg = AgentConfig(max_retries=2)
    r = decide(_state(defective=False, ood=cfg.ood_relook_threshold, retries=2), cfg)
    assert r.decision == Decision.ESCALATE


def test_zero_max_retries_skips_relook():
    """max_retries=0: uncertain OOD with no defect → rule 3 doesn't fire (0 < 0 is False),
    rule 4 doesn't fire, rule 5 fires if uncertain."""
    cfg = AgentConfig(max_retries=0)
    r = decide(_state(defective=False, ood=cfg.ood_relook_threshold, retries=0), cfg)
    # retries=0, max_retries=0: rules 3 & 4 skip (n < max_retries is False)
    # rule 5: n >= max_retries → True; ood >= relook → True → ESCALATE
    assert r.decision == Decision.ESCALATE
    assert r.rule_index == 5


# ---------------------------------------------------------------------------
# 4. Config validation
# ---------------------------------------------------------------------------

def test_config_invalid_ood_thresholds():
    with pytest.raises(ValueError, match="ood_relook_threshold"):
        AgentConfig(ood_relook_threshold=15.0, ood_escalate_threshold=10.0)


def test_config_equal_ood_thresholds_raises():
    with pytest.raises(ValueError):
        AgentConfig(ood_relook_threshold=10.0, ood_escalate_threshold=10.0)


def test_config_invalid_bsf_thresholds():
    with pytest.raises(ValueError, match="bsf_relook_threshold"):
        AgentConfig(bsf_relook_threshold=0.5, bsf_escalate_threshold=0.2)


def test_config_negative_max_retries():
    with pytest.raises(ValueError, match="max_retries"):
        AgentConfig(max_retries=-1)


def test_config_zero_max_retries_is_valid():
    cfg = AgentConfig(max_retries=0)
    assert cfg.max_retries == 0


# ---------------------------------------------------------------------------
# 5. Reason strings
# ---------------------------------------------------------------------------

def test_reason_contains_class_name():
    r = decide(_state(cls="candle", defective=True, ood=CFG.ood_escalate_threshold))
    assert "candle" in r.reason


def test_reason_mentions_ood_score_rule1():
    r = decide(_state(defective=True, ood=CFG.ood_escalate_threshold))
    assert str(round(CFG.ood_escalate_threshold, 2)) in r.reason or "OOD" in r.reason


def test_reason_mentions_retry_count_rule3():
    r = decide(_state(defective=False, ood=CFG.ood_relook_threshold, retries=1))
    assert "2" in r.reason or "1" in r.reason  # attempt 2/max or similar


def test_reason_accept_mentions_clean():
    r = decide(_state(defective=False))
    assert r.reason  # non-empty
    assert "Accepted" in r.reason or "clean" in r.reason


# ---------------------------------------------------------------------------
# 6. AgentResult fields
# ---------------------------------------------------------------------------

def test_decision_id_is_valid_uuid():
    r = decide(_state())
    uuid.UUID(r.decision_id)  # raises if not valid UUID


def test_two_decisions_have_different_ids():
    r1 = decide(_state())
    r2 = decide(_state())
    assert r1.decision_id != r2.decision_id


def test_timestamp_is_recent():
    before = time.time()
    r = decide(_state())
    after = time.time()
    assert before <= r.timestamp_utc <= after


def test_result_carries_state_and_config():
    s = _state(cls="pcb4", ood=5.0)
    r = decide(s, CFG)
    assert r.state is s
    assert r.config is CFG


# ---------------------------------------------------------------------------
# 7. Decision enum
# ---------------------------------------------------------------------------

def test_decision_enum_members():
    names = {d.name for d in Decision}
    assert names == {"ACCEPT", "RELOOK", "ESCALATE"}


# ---------------------------------------------------------------------------
# 8. TraceStore — round-trip
# ---------------------------------------------------------------------------

def test_trace_store_roundtrip():
    with TraceStore(":memory:") as store:
        r = decide(_state(cls="candle", ood=1.0, bsf=0.05, top_norm=2.3, retries=0))
        store.log(r, frame_id="candle/test/img_001.JPG")

        rows = store.all_rows()
        assert len(rows) == 1
        row = rows[0]

        assert row["decision_id"]  == r.decision_id
        assert row["frame_id"]     == "candle/test/img_001.JPG"
        assert row["object_class"] == "candle"
        assert row["decision"]     == r.decision.name
        assert row["rule_index"]   == r.rule_index
        assert math.isclose(row["ood_score"],      1.0,  rel_tol=1e-6)
        assert math.isclose(row["bsf_residual"],   0.05, rel_tol=1e-6)
        assert math.isclose(row["bsf_top_block_norm"], 2.3, rel_tol=1e-6)
        assert row["retry_count"]  == 0


def test_trace_store_frame_id_from_state():
    """frame_id falls back to state.frame_id when not passed to log()."""
    s = _state(frame_id="pcb2/test/img_005.JPG")
    r = decide(s)
    with TraceStore(":memory:") as store:
        store.log(r)  # no frame_id kwarg
        rows = store.all_rows()
        assert rows[0]["frame_id"] == "pcb2/test/img_005.JPG"


# ---------------------------------------------------------------------------
# 9. TraceStore replay
# ---------------------------------------------------------------------------

def test_trace_store_replay_by_frame_id():
    with TraceStore(":memory:") as store:
        fid = "pcb1/test/img_042.JPG"
        for ood in [1.0, 9.0, 15.0]:
            r = decide(_state(ood=ood, defective=ood >= 15.0))
            store.log(r, frame_id=fid)
        # also log a decoy from a different frame
        store.log(decide(_state()), frame_id="pcb1/test/img_099.JPG")

        rows = store.replay(fid)
        assert len(rows) == 3
        assert all(row["frame_id"] == fid for row in rows)


def test_replay_empty_for_unknown_frame():
    with TraceStore(":memory:") as store:
        assert store.replay("nonexistent") == []


# ---------------------------------------------------------------------------
# 10. TraceStore rates
# ---------------------------------------------------------------------------

def test_escalation_rate_empty_store():
    with TraceStore(":memory:") as store:
        assert store.escalation_rate() == 0.0


def test_escalation_rate_correct():
    with TraceStore(":memory:") as store:
        # 2 ESCALATE, 1 ACCEPT, 1 RELOOK → rate = 0.5
        store.log(decide(_state(defective=True, ood=CFG.ood_escalate_threshold)), "f1")
        store.log(decide(_state(defective=True, ood=CFG.ood_escalate_threshold)), "f2")
        store.log(decide(_state(defective=False, ood=0.0)),                       "f3")
        store.log(decide(_state(defective=False, ood=CFG.ood_relook_threshold)),   "f4")

        rate = store.escalation_rate()
        assert math.isclose(rate, 0.5, rel_tol=1e-6)


def test_accept_rate_correct():
    with TraceStore(":memory:") as store:
        store.log(decide(_state(defective=False, ood=0.0)), "f1")
        store.log(decide(_state(defective=False, ood=0.0)), "f2")
        store.log(decide(_state(defective=True, ood=CFG.ood_escalate_threshold)), "f3")

        assert math.isclose(store.accept_rate(), 2 / 3, rel_tol=1e-6)


def test_class_filtered_escalation_rate():
    with TraceStore(":memory:") as store:
        store.log(decide(_state(cls="pcb1", defective=True, ood=CFG.ood_escalate_threshold)), "f1")
        store.log(decide(_state(cls="pcb1", defective=False, ood=0.0)),                       "f2")
        store.log(decide(_state(cls="candle", defective=True, ood=CFG.ood_escalate_threshold)), "f3")
        store.log(decide(_state(cls="candle", defective=True, ood=CFG.ood_escalate_threshold)), "f4")

        assert math.isclose(store.escalation_rate("pcb1"),  0.5, rel_tol=1e-6)
        assert math.isclose(store.escalation_rate("candle"), 1.0, rel_tol=1e-6)


# ---------------------------------------------------------------------------
# 11. TraceStore decision_counts
# ---------------------------------------------------------------------------

def test_decision_counts():
    with TraceStore(":memory:") as store:
        store.log(decide(_state(defective=False, ood=0.0)), "f1")
        store.log(decide(_state(defective=False, ood=0.0)), "f2")
        store.log(decide(_state(defective=False, ood=CFG.ood_relook_threshold)), "f3")
        store.log(decide(_state(defective=True,  ood=CFG.ood_escalate_threshold)), "f4")

        counts = store.decision_counts()
        assert counts.get("ACCEPT", 0)   == 2
        assert counts.get("RELOOK", 0)   == 1
        assert counts.get("ESCALATE", 0) == 1


# ---------------------------------------------------------------------------
# 12. Context manager
# ---------------------------------------------------------------------------

def test_context_manager_closes_store():
    with TraceStore(":memory:") as store:
        store.log(decide(_state()), "f1")
    # After __exit__, further calls should raise (connection closed)
    with pytest.raises(Exception):
        store.all_rows()


# ---------------------------------------------------------------------------
# 13. Vision fields in trace
# ---------------------------------------------------------------------------

def test_trace_vision_fields_defective():
    with TraceStore(":memory:") as store:
        r = decide(_state(defective=True, ood=CFG.ood_escalate_threshold))
        store.log(r, "f1")
        row = store.all_rows()[0]
        assert row["vision_is_defective"] == 1
        assert row["vision_n_defects"]    == 1
        assert row["vision_total_area_px"] > 0


def test_trace_vision_fields_clean():
    with TraceStore(":memory:") as store:
        r = decide(_state(defective=False))
        store.log(r, "f1")
        row = store.all_rows()[0]
        assert row["vision_is_defective"] == 0
        assert row["vision_n_defects"]    == 0
        assert row["vision_total_area_px"] == 0.0
