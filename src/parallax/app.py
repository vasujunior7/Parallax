"""Parallax FastAPI intake service.

Provides a single HTTP endpoint that accepts a frame (multipart upload or
base64 JSON), runs all three heads (OpenCV verdict, linear probe OOD score,
BSF confidence), applies the agent policy, and returns a structured decision.

All decisions are persisted to the SQLite trace store for later analysis
and for the escalation review queue UI.

Usage::

    # Start the server (development)
    uv run uvicorn parallax.app:app --host 0.0.0.0 --port 8000 --reload

    # Upload a frame
    curl -X POST http://localhost:8000/inspect \\
         -F "frame=@path/to/image.jpg" \\
         -F "object_class=pcb1"

Endpoints
---------
POST /inspect
    Run the full pipeline on an uploaded frame.  Returns a decision JSON.

GET /escalations
    Return the most recent ESCALATE decisions (for the review UI).

GET /health
    Liveness check.

GET /
    Serve the escalation review queue UI (static HTML).
"""
from __future__ import annotations

import base64
import io
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# ── FastAPI ──────────────────────────────────────────────────────────────────
try:
    from fastapi import FastAPI, File, Form, HTTPException, UploadFile
    from fastapi.responses import HTMLResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "FastAPI and uvicorn are required for the intake service.\n"
        "Install with: uv add fastapi uvicorn python-multipart"
    ) from e

# ── Parallax internals ───────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from parallax.agent import AgentConfig, AgentState, Decision, decide
from parallax.backbone import centre_and_scale
from parallax.bsf import ParallaxBSF
from parallax.features import BackboneRunner
from parallax.inspect import inspect as cv_inspect
from parallax.probe import LinearProbe
from parallax.reference import GoldenReference
from parallax.trace import TraceStore

log = logging.getLogger("parallax.app")
logging.basicConfig(level=logging.INFO)

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR    = PROJECT_ROOT / "data"
PROBE_DIR   = DATA_DIR / "probes"
BSF_DIR     = DATA_DIR / "bsf"
REF_DIR     = DATA_DIR / "references"
POS_MEAN    = PROJECT_ROOT / "vendor" / "block-sparse-featurizer" / "bsf" / "pos_mean.npy"
TRACE_DB    = PROJECT_ROOT / "logs" / "agent" / "intake_trace.db"

# ── Lazy-loaded singletons ────────────────────────────────────────────────────
_runner:   Optional[BackboneRunner]  = None
_pos_mean: Optional[np.ndarray]     = None
_probes:   dict[str, LinearProbe]   = {}
_bsf:      dict[str, ParallaxBSF]   = {}
_refs:     dict[str, GoldenReference] = {}
_store:    Optional[TraceStore]     = None
_config:   AgentConfig              = AgentConfig(
    # Calibrated from VisA pcb1–pcb4 score distributions (mean_normal ≈ 2316,
    # half_separation ≈ 189): relook @ 1.5σ, escalate @ 2.5σ.
    ood_relook_threshold   = 2600.0,
    ood_escalate_threshold = 2790.0,
    bsf_relook_threshold   = 0.18,
    bsf_escalate_threshold = 0.28,
)


def _get_runner() -> BackboneRunner:
    global _runner
    if _runner is None:
        log.info("Loading backbone runner (first request may be slow)…")
        _runner = BackboneRunner()
    return _runner


def _get_pos_mean() -> np.ndarray:
    global _pos_mean
    if _pos_mean is None:
        _pos_mean = np.load(str(POS_MEAN)).astype(np.float32)
    return _pos_mean


def _get_probe(object_class: str) -> LinearProbe:
    if object_class not in _probes:
        path = PROBE_DIR / f"{object_class}.npz"
        if not path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Probe not found for class '{object_class}'. "
                       f"Run scripts/train_probe.py first.",
            )
        _probes[object_class] = LinearProbe.load(path)
    return _probes[object_class]


def _get_bsf(object_class: str) -> ParallaxBSF:
    if object_class not in _bsf:
        path = BSF_DIR / f"{object_class}.pt"
        if not path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"BSF weights not found for class '{object_class}'. "
                       f"Run scripts/train_bsf.py first.",
            )
        _bsf[object_class] = ParallaxBSF(path)
    return _bsf[object_class]


def _get_ref(object_class: str) -> GoldenReference:
    if object_class not in _refs:
        path = REF_DIR / f"{object_class}.npz"
        if not path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Golden reference not found for class '{object_class}'. "
                       f"Run scripts/build_references.py first.",
            )
        _refs[object_class] = GoldenReference.load(path)
    return _refs[object_class]


def _get_store() -> TraceStore:
    global _store
    if _store is None:
        TRACE_DB.parent.mkdir(parents=True, exist_ok=True)
        target_db = TRACE_DB
        if not target_db.exists():
            runs = sorted(TRACE_DB.parent.glob("run_*.db"))
            if runs:
                target_db = runs[-1]
        _store = TraceStore(target_db)
    return _store


def _decode_image(data: bytes) -> np.ndarray:
    """Decode raw image bytes (JPEG, PNG, …) to a BGR numpy array."""
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=422, detail="Could not decode uploaded image.")
    return img


def _run_pipeline(img: np.ndarray, object_class: str, frame_id: Optional[str] = None) -> dict:
    """Run all three heads + agent decision for one frame."""
    runner   = _get_runner()
    pos_mean = _get_pos_mean()
    probe    = _get_probe(object_class)
    bsf_mdl  = _get_bsf(object_class)
    ref      = _get_ref(object_class)

    # 1. DINOv3 activations
    feats = runner.extract(img)

    # 2. OpenCV verdict
    ref_img = ref.median.astype(np.uint8)
    verdict = cv_inspect(img, ref_img)

    # 3. Probe OOD score
    flat        = feats.tokens.reshape(-1, feats.tokens.shape[-1])
    patch_scores = probe.score(flat)
    ood_score    = float(patch_scores.max())

    # 4. BSF residual + block coord
    from parallax.bsf import BSFConcepts
    concepts    = bsf_mdl.extract_concepts(feats)
    flat_norms  = concepts.norms.reshape(-1, concepts.n_groups)
    flat_coords = concepts.coords.reshape(-1, concepts.n_groups, -1)
    idx_p, idx_g = np.unravel_index(flat_norms.argmax(), flat_norms.shape)
    top_norm    = float(flat_norms[idx_p, idx_g])
    top_coord   = flat_coords[idx_p, idx_g].tolist()

    # Residual: mean MSE across all patches
    import torch
    x     = torch.as_tensor(
        centre_and_scale(feats.tokens, pos_mean), dtype=torch.float32, device=bsf_mdl.device
    )
    z     = bsf_mdl.model.encode(x)
    B     = bsf_mdl.model.B_raw
    x_hat = torch.einsum("bgs,gds->bd", z, B)
    bsf_err = float((x - x_hat).pow(2).mean().item())

    # 5. Agent decision
    state = AgentState(
        verdict             = verdict,
        ood_score           = ood_score,
        bsf_residual        = bsf_err,
        bsf_top_block_norm  = top_norm,
        bsf_top_block_coord = top_coord,
        retry_count         = 0,
        object_class        = object_class,
        frame_id            = frame_id,
    )
    result = decide(state, _config)
    _get_store().log(result)

    return {
        "decision":           result.decision.name,
        "reason":             result.reason,
        "rule_index":         result.rule_index,
        "decision_id":        result.decision_id,
        "object_class":       object_class,
        "vision_is_defective": verdict.is_defective,
        "vision_n_defects":    len(verdict.defects),
        "vision_total_area_px": verdict.total_area_px,
        "ood_score":          round(ood_score, 2),
        "bsf_residual":       round(bsf_err, 5),
        "bsf_top_block_norm": round(top_norm, 4),
        "bsf_top_block_coord": top_coord,
        "config": {
            "ood_relook_threshold":   _config.ood_relook_threshold,
            "ood_escalate_threshold": _config.ood_escalate_threshold,
            "bsf_relook_threshold":   _config.bsf_relook_threshold,
            "bsf_escalate_threshold": _config.bsf_escalate_threshold,
        },
    }


# ── Application ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="Parallax Inspection Service",
    description=(
        "Industrial visual inspection with self-calibrated escalation. "
        "POST a frame to /inspect; GET /escalations for the human review queue."
    ),
    version="0.1.0",
)


@app.get("/health")
def health() -> dict:
    """Liveness check."""
    return {"status": "ok", "service": "parallax"}


@app.post("/inspect")
async def inspect_frame(
    frame: UploadFile = File(..., description="Image file (JPEG, PNG, etc.)"),
    object_class: str = Form(..., description="VisA class name, e.g. 'pcb1'"),
    frame_id: Optional[str] = Form(None, description="Optional stable frame identifier"),
) -> JSONResponse:
    """Run the full Parallax pipeline on an uploaded frame.

    Returns the agent decision (ACCEPT / RELOOK / ESCALATE), the reason
    string, and all raw scores.
    """
    data = await frame.read()
    img  = _decode_image(data)
    fid  = frame_id or frame.filename
    result = _run_pipeline(img, object_class, frame_id=fid)
    status = 200 if result["decision"] != "ESCALATE" else 202
    return JSONResponse(content=result, status_code=status)


@app.get("/escalations")
def escalations(
    limit: int = 50,
    object_class: Optional[str] = None,
) -> JSONResponse:
    """Return the most recent decisions for the review queue, enriched with calibrated policy and BSF reasoning."""
    rows = _get_store().escalations_for_ui(limit=limit, object_class=object_class)
    from parallax.inspect import Verdict, Defect
    from parallax.agent import AgentState, decide
    for r in rows:
        if r["vision_is_defective"]:
            d = Defect(
                area_px=float(r["vision_total_area_px"]),
                centroid=(0.0, 0.0),
                bbox=(0, 0, 10, 10),
                perimeter_px=0.0,
                min_area_rect=((0.0, 0.0), (10.0, 10.0), 0.0),
            )
            v = Verdict(defects=(d,), mask=np.zeros((1, 1), dtype=np.uint8))
        else:
            v = Verdict(defects=(), mask=np.zeros((1, 1), dtype=np.uint8))
        st = AgentState(
            verdict=v,
            ood_score=r["ood_score"],
            bsf_residual=r["bsf_residual"],
            bsf_top_block_norm=r["bsf_top_block_norm"],
            retry_count=r["retry_count"],
            object_class=r["object_class"],
            frame_id=r["frame_id"],
            bsf_top_block_coord=r.get("bsf_top_block_coord"),
        )
        res = decide(st, _config)
        r["calibrated_decision"] = res.decision.name
        r["calibrated_reason"]   = res.reason
        
        # Explicit BSF explanation
        if r["bsf_residual"] >= _config.bsf_escalate_threshold:
            r["bsf_diagnosis"] = f"Anomalous concept activation: reconstruction error {r['bsf_residual']:.4f} >= threshold {_config.bsf_escalate_threshold:.4f}."
        elif r["bsf_residual"] >= _config.bsf_relook_threshold:
            r["bsf_diagnosis"] = f"Uncertain concept match: residual {r['bsf_residual']:.4f} lies in ambiguity band [{_config.bsf_relook_threshold:.4f}, {_config.bsf_escalate_threshold:.4f}]."
        else:
            r["bsf_diagnosis"] = f"Normal concept subspace: residual {r['bsf_residual']:.4f} fits normal manifold (< {_config.bsf_relook_threshold:.4f})."

    return JSONResponse(content={"escalations": rows, "count": len(rows)})


@app.get("/", response_class=HTMLResponse)
def review_queue() -> HTMLResponse:
    """Serve the escalation review queue UI."""
    ui_path = Path(__file__).parent / "ui.html"
    if ui_path.exists():
        return HTMLResponse(content=ui_path.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>Parallax Review Queue</h1><p>UI not found — see ui.html</p>")
