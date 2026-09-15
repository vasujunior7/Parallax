"""Run the Parallax agent loop over the full VisA rigid test split.

For each test frame and each rigid class:
  1. OpenCV verdict:   load golden reference, z-score diff, run inspect()
  2. Linear Probe OOD: re-fit probe from train normals (if not cached), score frame
  3. BSF residual:     load trained BSF weights, compute mean patch reconstruction error
  4. Agent decision:   call decide() with all three signals
  5. Trace:           log to SQLite for Stage 11 threshold sweep

Outputs
-------
  logs/agent/
    run_<timestamp>.db        SQLite trace (all decisions + all input scores)
    summary.json              per-class + aggregate metrics
    escalation_report.md      human-readable confusion table
    plots/
      <class>_score_scatter.png
      escalation_curve.png    sensitivity vs false-escalation as OOD threshold varies

Run:
    uv run python scripts/run_agent.py
    uv run python scripts/run_agent.py --classes pcb1 candle  # subset
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import sys
import time
from pathlib import Path

if sys.platform == "win32":
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "buffer"):
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from parallax.agent import AgentConfig, AgentState, Decision, decide
from parallax.backbone import centre_and_scale
from parallax.bsf import ParallaxBSF
from parallax.features import BackboneRunner
from parallax.inspect import inspect as cv_inspect, DEFAULT_THRESHOLD
from parallax.probe import LinearProbe, fit as fit_probe
from parallax.reference import GoldenReference
from parallax.tiling import stitch
from parallax.trace import TraceStore
from parallax.visa import RIGID_CLASSES, ingest

# vendor BSF pos_mean
VENDOR_BSF   = PROJECT_ROOT / "vendor" / "block-sparse-featurizer"
POS_MEAN_PATH = VENDOR_BSF / "bsf" / "pos_mean.npy"

# paths
AGENT_LOGS  = PROJECT_ROOT / "logs" / "agent"
AGENT_PLOTS = AGENT_LOGS / "plots"
BSF_WEIGHTS = PROJECT_ROOT / "data" / "bsf"
PROBE_DIR   = PROJECT_ROOT / "data" / "probes"
REF_DIR     = PROJECT_ROOT / "data" / "references"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("run_agent")


# ==============================================================================
# Helpers
# ==============================================================================

def load_or_fit_probe(
    object_class: str,
    all_samples,
    runner: BackboneRunner,
    n_train: int = 80,
) -> LinearProbe:
    """Load a cached probe or fit from scratch if not on disk."""
    probe_path = PROBE_DIR / f"{object_class}.npz"
    if probe_path.exists():
        log.info(f"  [Probe] Loaded from {probe_path}")
        return LinearProbe.load(probe_path)

    log.info(f"  [Probe] Fitting from scratch (no cache at {probe_path})")
    train_normals = [
        s.image for s in all_samples
        if s.object_class == object_class and s.split == "train"
    ][:n_train]

    feats = runner.extract_paths(train_normals, progress=True)
    pooled = np.concatenate(
        [f.tokens.reshape(-1, f.tokens.shape[-1]) for f in feats]
    )
    probe = fit_probe(pooled, model_id=runner.spec.model_id)
    probe_path.parent.mkdir(parents=True, exist_ok=True)
    probe.save(probe_path)
    log.info(f"  [Probe] Saved to {probe_path}")
    return probe


@torch.inference_mode()
def bsf_residual(
    bsf_model: ParallaxBSF,
    feats,
    pos_mean: np.ndarray,
    batch: int = 4096,
) -> float:
    """Mean patch MSE reconstruction error for one frame (image-level BSF score)."""
    flat = centre_and_scale(feats.tokens, pos_mean)   # (n_tiles*n_pat, d)
    x = torch.as_tensor(flat, dtype=torch.float32, device=bsf_model.device)

    errors = []
    for i in range(0, len(x), batch):
        xb = x[i : i + batch]
        # BSF encode → decode → MSE
        z   = bsf_model.model.encode(xb)                  # (B, n_groups, group_size)
        # reconstruct: decoder is B_raw (n_groups, d, group_size)
        B   = bsf_model.model.B_raw                        # (n_groups, d, group_size)
        # z: (B, n_groups, gs)  B: (n_groups, d, gs)
        # x_hat[b] = sum_g B[g] @ z[b,g]
        x_hat = torch.einsum("bgs,gds->bd", z, B)
        mse   = (xb - x_hat).pow(2).mean(dim=1)           # (B,)
        errors.append(mse.cpu().numpy())

    return float(np.concatenate(errors).mean())


def probe_image_score(probe: LinearProbe, feats) -> float:
    """Max patch Mahalanobis score = image-level OOD score (PaDiM convention)."""
    flat = feats.tokens.reshape(-1, feats.tokens.shape[-1])
    patch_scores = probe.score(flat)
    return float(patch_scores.max())


def bsf_top_norm_and_coord(
    bsf_model: ParallaxBSF, feats, pos_mean: np.ndarray
) -> tuple[float, list[float]]:
    """Maximum block norm across all patches + the coordinate of that loudest block.

    Returns
    -------
    top_norm  : float — the highest block norm in the frame
    top_coord : list[float] — normalised coordinate vector inside that block
                (length = group_size, e.g. 3 for GrassmannianBSF)
    """
    concepts = bsf_model.extract_concepts(feats)  # BSFConcepts
    # norms: (n_tiles, n_patches, n_groups)  coords: (..., group_size)
    flat_norms  = concepts.norms.reshape(-1, concepts.n_groups)   # (N, n_groups)
    flat_coords = concepts.coords.reshape(-1, concepts.n_groups, -1)  # (N, n_groups, gs)
    idx_patch, idx_group = np.unravel_index(flat_norms.argmax(), flat_norms.shape)
    top_norm  = float(flat_norms[idx_patch, idx_group])
    top_coord = flat_coords[idx_patch, idx_group].tolist()
    return top_norm, top_coord


# ==============================================================================
# Per-class pipeline
# ==============================================================================

def run_class(
    object_class: str,
    all_samples,
    runner: BackboneRunner,
    bsf_mdl: ParallaxBSF,
    probe: LinearProbe,
    reference: GoldenReference,
    pos_mean: np.ndarray,
    store: TraceStore,
    config: AgentConfig,
    args: argparse.Namespace,
) -> dict:
    test_samples = [
        s for s in all_samples
        if s.object_class == object_class and s.split == "test"
    ][:args.max_test]

    log.info(f"  Test frames: {len(test_samples)}")

    class_rows: list[dict] = {
        "ACCEPT": 0, "RELOOK": 0, "ESCALATE": 0,
        "TP": 0, "FP": 0, "TN": 0, "FN": 0,
    }
    ood_scores_normal, ood_scores_anomaly = [], []
    bsf_scores_normal, bsf_scores_anomaly = [], []

    # Re-look parameter schedule: (clahe_clip_limit, crop_fraction, threshold_divisor)
    # Each retry tightens crop and boosts contrast — visual evidence changes the verdict.
    RELOOK_PARAMS = [
        (2.0, 1.00, 1.0),   # attempt 0: original frame, default CLAHE
        (3.5, 0.85, 1.2),   # attempt 1: tighter crop, higher contrast, lower threshold
        (4.5, 0.70, 1.5),   # attempt 2: tightest crop, max contrast, most sensitive diff
    ]

    for sample in test_samples:
        img_orig = cv2.imread(str(sample.image))
        if img_orig is None:
            log.warning(f"    Cannot read {sample.image}, skipping")
            continue

        retry_count = 0
        ood = bsf_err = top_norm = 0.0
        top_coord: list = []
        result = None

        while True:
            clahe_clip, crop_frac, thr_div = RELOOK_PARAMS[
                min(retry_count, len(RELOOK_PARAMS) - 1)
            ]

            # Apply crop for re-look attempts
            if crop_frac < 1.0:
                h, w = img_orig.shape[:2]
                cy, cx = h // 2, w // 2
                nh, nw = int(h * crop_frac), int(w * crop_frac)
                y0, x0 = cy - nh // 2, cx - nw // 2
                img_frame = img_orig[y0:y0 + nh, x0:x0 + nw]
            else:
                img_frame = img_orig

            # ── three heads ──────────────────────────────────────────────────
            feats = runner.extract(img_frame)

            # 1. OpenCV verdict (threshold varies by retry — vision can change)
            ref_img   = reference.median.astype(np.uint8)
            if crop_frac < 1.0:
                h, w = ref_img.shape[:2]
                cy, cx = h // 2, w // 2
                nh, nw = int(h * crop_frac), int(w * crop_frac)
                y0, x0 = cy - nh // 2, cx - nw // 2
                ref_crop = ref_img[y0:y0 + nh, x0:x0 + nw]
            else:
                ref_crop = ref_img

            verdict = cv_inspect(
                img_frame, ref_crop,
                threshold=max(1, int(40 / thr_div)),
            )

            # 2. Probe OOD score
            ood = probe_image_score(probe, feats)

            # 3. BSF residual + block coord (the UI needs coord to explain escalations)
            bsf_err            = bsf_residual(bsf_mdl, feats, pos_mean)
            top_norm, top_coord = bsf_top_norm_and_coord(bsf_mdl, feats, pos_mean)

            # ── agent decision ───────────────────────────────────────────────
            state = AgentState(
                verdict              = verdict,
                ood_score            = ood,
                bsf_residual         = bsf_err,
                bsf_top_block_norm   = top_norm,
                bsf_top_block_coord  = top_coord,
                retry_count          = retry_count,
                object_class         = object_class,
                frame_id             = str(sample.image),
            )
            result = decide(state, config)
            store.log(result)

            if result.decision == Decision.RELOOK and retry_count < config.max_retries:
                class_rows["RELOOK"] = class_rows.get("RELOOK", 0) + 1
                retry_count += 1
                log.debug(
                    f"    RE-LOOK {retry_count}: crop={crop_frac:.0%} "
                    f"clahe={clahe_clip} thr_div={thr_div}"
                )
                continue  # re-run all three heads with new params

            # Terminal decision: ACCEPT or ESCALATE
            d_name = result.decision.name
            class_rows[d_name] = class_rows.get(d_name, 0) + 1
            break

        # confusion: ground-truth label vs ESCALATE
        is_anomalous = sample.is_anomalous
        escalated    = result.decision == Decision.ESCALATE
        if is_anomalous and escalated:
            class_rows["TP"] += 1
        elif is_anomalous and not escalated:
            class_rows["FN"] += 1
        elif not is_anomalous and escalated:
            class_rows["FP"] += 1
        else:
            class_rows["TN"] += 1

        if is_anomalous:
            ood_scores_anomaly.append(ood)
            bsf_scores_anomaly.append(bsf_err)
        else:
            ood_scores_normal.append(ood)
            bsf_scores_normal.append(bsf_err)

    total = sum(class_rows[k] for k in ("ACCEPT", "RELOOK", "ESCALATE"))
    tp, fp, tn, fn = class_rows["TP"], class_rows["FP"], class_rows["TN"], class_rows["FN"]
    sensitivity = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    false_esc   = fp / max(fp + tn, 1)

    log.info(
        f"  ACCEPT={class_rows['ACCEPT']}  RELOOK={class_rows['RELOOK']}  ESCALATE={class_rows['ESCALATE']}"
    )
    log.info(
        f"  Sensitivity(anomaly->ESC)={sensitivity:.3f}  "
        f"Specificity(normal->ACC)={specificity:.3f}  "
        f"FalseEsc={false_esc:.3f}"
    )

    return {
        "class":       object_class,
        "n_test":      total,
        "accept":      class_rows["ACCEPT"],
        "relook":      class_rows["RELOOK"],
        "escalate":    class_rows["ESCALATE"],
        "TP": tp, "FP": fp, "TN": tn, "FN": fn,
        "sensitivity": round(sensitivity, 4),
        "specificity": round(specificity, 4),
        "false_esc_rate": round(false_esc, 4),
        "ood_normal_mean":   round(float(np.mean(ood_scores_normal))  if ood_scores_normal  else 0, 3),
        "ood_anomaly_mean":  round(float(np.mean(ood_scores_anomaly)) if ood_scores_anomaly else 0, 3),
        "bsf_normal_mean":   round(float(np.mean(bsf_scores_normal))  if bsf_scores_normal  else 0, 4),
        "bsf_anomaly_mean":  round(float(np.mean(bsf_scores_anomaly)) if bsf_scores_anomaly else 0, 4),
        "_ood_normal":   ood_scores_normal,
        "_ood_anomaly":  ood_scores_anomaly,
        "_bsf_normal":   bsf_scores_normal,
        "_bsf_anomaly":  bsf_scores_anomaly,
    }


# ==============================================================================
# Plotting
# ==============================================================================

def plot_score_scatter(metrics: dict, out_dir: Path) -> None:
    cls = metrics["class"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"Agent score distributions — {cls}", fontsize=13, fontweight="bold")

    for ax, norm, anml, label, title in [
        (axes[0], metrics["_ood_normal"], metrics["_ood_anomaly"], "OOD score", "Probe OOD"),
        (axes[1], metrics["_bsf_normal"], metrics["_bsf_anomaly"], "BSF residual", "BSF residual"),
    ]:
        ax.hist(norm,  bins=30, alpha=0.6, color="steelblue", label="Normal")
        ax.hist(anml, bins=30, alpha=0.6, color="tomato",    label="Anomaly")
        ax.set_title(title); ax.set_xlabel(label); ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    p = out_dir / f"{cls}_score_scatter.png"
    plt.savefig(p, dpi=150); plt.close(fig)
    log.info(f"  Saved score plot -> {p}")


def plot_escalation_curve(store: TraceStore, config: AgentConfig, out_dir: Path) -> None:
    """Sweep the OOD threshold and plot sensitivity vs false-escalation rate."""
    rows = store.all_rows()
    if not rows:
        return

    ood_vals = np.array([r["ood_score"] for r in rows])
    is_anml  = np.array(["anomaly" in (r.get("frame_id") or "").lower() for r in rows], dtype=bool)
    if is_anml.sum() == 0:
        is_anml = np.array([r["vision_is_defective"] for r in rows], dtype=bool)

    thresholds = np.percentile(ood_vals, np.linspace(0, 100, 200))
    sens_list, fpr_list = [], []
    for t in thresholds:
        escalated = ood_vals >= t
        tp = (escalated & is_anml).sum()
        fn = (~escalated & is_anml).sum()
        fp = (escalated & ~is_anml).sum()
        tn = (~escalated & ~is_anml).sum()
        sens_list.append(tp / max(tp + fn, 1))
        fpr_list.append(fp / max(fp + tn, 1))

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr_list, sens_list, "b-o", ms=3, label="OOD threshold sweep")
    # mark the current operating point
    curr_esc = np.array([r["decision"] == "ESCALATE" for r in rows])
    tp_c = (curr_esc & is_anml).sum()
    fn_c = (~curr_esc & is_anml).sum()
    fp_c = (curr_esc & ~is_anml).sum()
    tn_c = (~curr_esc & ~is_anml).sum()
    ax.plot(fp_c / max(fp_c+tn_c, 1), tp_c / max(tp_c+fn_c, 1),
            "r*", ms=14, label=f"Current config (OOD>={config.ood_relook_threshold})")
    ax.set_xlabel("False Escalation Rate"); ax.set_ylabel("Sensitivity (anomaly detected)")
    ax.set_title("Escalation trade-off curve (OOD threshold sweep)")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    p = out_dir / "escalation_curve.png"
    plt.savefig(p, dpi=150); plt.close(fig)
    log.info(f"Saved escalation curve -> {p}")


# ==============================================================================
# Report
# ==============================================================================

def generate_report(all_metrics: list[dict], out_dir: Path, config: AgentConfig) -> None:
    total_tp = sum(m["TP"] for m in all_metrics)
    total_fp = sum(m["FP"] for m in all_metrics)
    total_tn = sum(m["TN"] for m in all_metrics)
    total_fn = sum(m["FN"] for m in all_metrics)
    agg_sens = total_tp / max(total_tp + total_fn, 1)
    agg_spec = total_tn / max(total_tn + total_fp, 1)
    agg_fer  = total_fp / max(total_fp + total_tn, 1)

    def fmt(v) -> str:
        return "—" if v is None else f"{v:.3f}"

    lines = [
        "# Agent Loop — Escalation Report",
        "",
        f"*Generated {time.strftime('%Y-%m-%d %H:%M')}*",
        "",
        "## Config",
        "",
        f"| Param | Value |",
        f"|---|---|",
        f"| ood_relook_threshold   | {config.ood_relook_threshold} |",
        f"| ood_escalate_threshold | {config.ood_escalate_threshold} |",
        f"| bsf_relook_threshold   | {config.bsf_relook_threshold} |",
        f"| bsf_escalate_threshold | {config.bsf_escalate_threshold} |",
        f"| max_retries            | {config.max_retries} |",
        "",
        "## Per-class decisions",
        "",
        "| Class | Frames | ACCEPT | RELOOK | ESCALATE | Sensitivity | Specificity | FalseEsc |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for m in all_metrics:
        lines.append(
            f"| {m['class']} | {m['n_test']} | {m['accept']} | {m['relook']} | {m['escalate']} "
            f"| {fmt(m['sensitivity'])} | {fmt(m['specificity'])} | {fmt(m['false_esc_rate'])} |"
        )
    lines += [
        f"| **All** | {sum(m['n_test'] for m in all_metrics)} "
        f"| {sum(m['accept'] for m in all_metrics)} "
        f"| {sum(m['relook'] for m in all_metrics)} "
        f"| {sum(m['escalate'] for m in all_metrics)} "
        f"| **{agg_sens:.3f}** | **{agg_spec:.3f}** | **{agg_fer:.3f}** |",
        "",
        "## Signal means",
        "",
        "| Class | OOD normal | OOD anomaly | BSF normal | BSF anomaly |",
        "|---|---|---|---|---|",
    ]
    for m in all_metrics:
        lines.append(
            f"| {m['class']} | {m['ood_normal_mean']} | {m['ood_anomaly_mean']} "
            f"| {m['bsf_normal_mean']} | {m['bsf_anomaly_mean']} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        f"- **Sensitivity**: {agg_sens:.3f} — fraction of anomaly frames that triggered ESCALATE.",
        f"- **Specificity**: {agg_spec:.3f} — fraction of normal frames correctly NOT escalated.",
        f"- **False escalation rate**: {agg_fer:.3f} — fraction of normal frames wrongly escalated.",
        "- See `plots/escalation_curve.png` for the full threshold sweep.",
    ]

    md_path = out_dir / "escalation_report.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    log.info(f"Saved report -> {md_path}")

    # stdout
    print()
    print("=" * 90)
    print(f"{'Class':<12}  {'Frames':>6}  {'ACC':>5}  {'REL':>5}  {'ESC':>5}  "
          f"{'Sens':>6}  {'Spec':>6}  {'FalseESC':>8}")
    print("-" * 90)
    for m in all_metrics:
        print(
            f"{m['class']:<12}  {m['n_test']:>6}  {m['accept']:>5}  {m['relook']:>5}  "
            f"{m['escalate']:>5}  {m['sensitivity']:>6.3f}  {m['specificity']:>6.3f}  "
            f"{m['false_esc_rate']:>8.3f}"
        )
    print("=" * 90)
    print(f"{'ALL':<12}  {sum(m['n_test'] for m in all_metrics):>6}  "
          f"{sum(m['accept'] for m in all_metrics):>5}  "
          f"{sum(m['relook'] for m in all_metrics):>5}  "
          f"{sum(m['escalate'] for m in all_metrics):>5}  "
          f"{agg_sens:>6.3f}  {agg_spec:>6.3f}  {agg_fer:>8.3f}")
    print(f"\n  Report -> {md_path}")


# ==============================================================================
# Entry point
# ==============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Run Parallax agent over VisA test split.")
    parser.add_argument("--classes",         nargs="+", default=list(RIGID_CLASSES))
    parser.add_argument("--max-test",        type=int, default=200,
                        help="max test frames per class (default: all)")
    parser.add_argument("--ood-relook",      type=float, default=8.0)
    parser.add_argument("--ood-escalate",    type=float, default=14.0)
    parser.add_argument("--bsf-relook",      type=float, default=0.15)
    parser.add_argument("--bsf-escalate",    type=float, default=0.25)
    parser.add_argument("--max-retries",     type=int,   default=2)
    args = parser.parse_args()

    for d in (AGENT_LOGS, AGENT_PLOTS):
        d.mkdir(parents=True, exist_ok=True)

    fh = logging.FileHandler(AGENT_LOGS / "run.log", mode="a", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-7s  %(message)s", datefmt="%H:%M:%S"))
    logging.getLogger().addHandler(fh)

    config = AgentConfig(
        ood_relook_threshold   = args.ood_relook,
        ood_escalate_threshold = args.ood_escalate,
        bsf_relook_threshold   = args.bsf_relook,
        bsf_escalate_threshold = args.bsf_escalate,
        max_retries            = args.max_retries,
    )

    ts = time.strftime("%Y%m%d_%H%M%S")
    db_path = AGENT_LOGS / f"run_{ts}.db"

    log.info("=" * 60)
    log.info("Parallax agent run  classes=%s", args.classes)
    log.info("Config: OOD>=%.1f->relook  OOD>=%.1f->esc  BSF>=%.3f->relook  BSF>=%.3f->esc  retries=%d",
             config.ood_relook_threshold, config.ood_escalate_threshold,
             config.bsf_relook_threshold, config.bsf_escalate_threshold,
             config.max_retries)
    log.info("=" * 60)

    all_samples = ingest(rigid_only=True)
    log.info(f"{len(all_samples)} samples indexed")

    log.info("Loading backbone runner...")
    runner = BackboneRunner()
    log.info(f"  {runner.describe()}")

    pos_mean = np.load(POS_MEAN_PATH).astype(np.float32)

    all_metrics = []

    with TraceStore(db_path) as store:
        for object_class in args.classes:
            log.info(f"{'='*60}")
            log.info(f"CLASS: {object_class}")
            log.info(f"{'='*60}")

            # Load models
            bsf_path = BSF_WEIGHTS / f"{object_class}.pt"
            if not bsf_path.exists():
                log.error(f"  BSF weights not found: {bsf_path} — skipping class")
                continue

            ref_path = REF_DIR / f"{object_class}.npz"
            if not ref_path.exists():
                log.error(f"  Golden reference not found: {ref_path} — skipping class")
                continue

            log.info(f"  Loading BSF from {bsf_path}")
            bsf_mdl = ParallaxBSF(bsf_path)

            log.info(f"  Loading golden reference from {ref_path}")
            reference = GoldenReference.load(ref_path)

            probe = load_or_fit_probe(object_class, all_samples, runner)

            metrics = run_class(
                object_class, all_samples, runner, bsf_mdl, probe, reference,
                pos_mean, store, config, args,
            )
            all_metrics.append(metrics)

            plot_score_scatter(metrics, AGENT_PLOTS)

            # save per-class metrics (strip private lists)
            clean = {k: v for k, v in metrics.items() if not k.startswith("_")}
            with (AGENT_LOGS / f"{object_class}_metrics.json").open("w") as f:
                json.dump(clean, f, indent=2)

        # aggregate
        summary_clean = [{k: v for k, v in m.items() if not k.startswith("_")} for m in all_metrics]
        with (AGENT_LOGS / "summary.json").open("w") as f:
            json.dump(summary_clean, f, indent=2)

        plot_escalation_curve(store, config, AGENT_PLOTS)
        generate_report(all_metrics, AGENT_LOGS, config)

    log.info(f"Trace DB -> {db_path}")
    log.info("Done.")


if __name__ == "__main__":
    main()
