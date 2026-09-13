"""Train the SAE baseline and compare against BSF and Linear Probe.

This script is the **measurement** for the architecture decision document claim:
  "SAEs dilute manifold structure across many atoms; BSF captures it in a block."
We train a BatchTopKSAE (from vendor/sae-manifold) at matched capacity to the BSF
and measure:
  - Reconstruction-based anomaly detection (same MSE metric used for BSF)
  - Subspace capture: how many atoms does each model need to explain 95% of the
    normal manifold variance? (geometric greedy from find_support_greedy)

All evaluation mirrors train_bsf.py exactly so numbers are directly comparable.

Run:
    uv run python scripts/train_sae.py
    uv run python scripts/train_sae.py --epochs 5 --train 10  # quick test
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
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -- project imports -----------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from parallax.backbone import centre_and_scale
from parallax.features import BackboneRunner
from parallax.visa import RIGID_CLASSES, ingest

# -- vendor SAE ----------------------------------------------------------------
VENDOR_SAE = PROJECT_ROOT / "vendor" / "sae-manifold"
if str(VENDOR_SAE) not in sys.path:
    sys.path.insert(0, str(VENDOR_SAE))

from saes import BatchTopKSAE, get_decoder           # noqa: E402
# NOTE: we do NOT import find_support_greedy from vendor/sae-manifold because
# its _detect_elbow uses np.cross on 2D vectors, which was removed in NumPy 2.x.
# We implement a compatible version below.

# -- vendor BSF (for pos_mean) -------------------------------------------------
VENDOR_BSF = PROJECT_ROOT / "vendor" / "block-sparse-featurizer"
POS_MEAN_PATH = VENDOR_BSF / "bsf" / "pos_mean.npy"

# -- sklearn metrics -----------------------------------------------------------
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
)

# -- output directories --------------------------------------------------------
SAE_WEIGHTS = PROJECT_ROOT / "data" / "sae"
SAE_LOGS    = PROJECT_ROOT / "logs" / "sae"
SAE_PLOTS   = SAE_LOGS / "plots"
BSF_LOGS    = PROJECT_ROOT / "logs" / "bsf"

MAX_TRAIN_PATCHES = 500_000

# -- logging -------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("train_sae")


# ==============================================================================
# SAE training loop
# ==============================================================================

def train_sae(
    model: BatchTopKSAE,
    X: np.ndarray,
    epochs: int = 40,
    batch_size: int = 2048,
    lr: float = 2e-4,
    device: str = "cpu",
    log_every: int = 5,
) -> tuple[BatchTopKSAE, dict]:
    """Train a BatchTopKSAE on (N, d) activations with Adam."""
    model = model.to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)

    X_t = torch.as_tensor(X, dtype=torch.float32).to(device)
    n = len(X_t)

    history: dict[str, list] = {
        "epoch": [], "loss": [], "recon_loss": [], "l0": [],
    }

    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n, device=device)
        epoch_loss  = 0.0
        epoch_recon = 0.0
        epoch_l0    = 0.0
        n_batches   = 0

        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            xb = X_t[idx]

            x_hat, z = model(xb)
            recon  = F.mse_loss(x_hat, xb)
            l1_pen = 1e-4 * z.abs().mean()   # light auxiliary penalty; TopK already enforces sparsity
            loss   = recon + l1_pen

            optimiser.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()

            # Keep decoder columns unit-normed (standard SAE convention)
            with torch.no_grad():
                w = model.decoder.weight       # (d_in, d_sae)
                norms = w.norm(dim=0, keepdim=True).clamp(min=1e-8)
                model.decoder.weight.copy_(w / norms)

            epoch_loss  += loss.item()
            epoch_recon += recon.item()
            epoch_l0    += (z > 0).float().sum(1).mean().item()
            n_batches   += 1

        avg_loss  = epoch_loss  / max(n_batches, 1)
        avg_recon = epoch_recon / max(n_batches, 1)
        avg_l0    = epoch_l0    / max(n_batches, 1)

        history["epoch"].append(ep)
        history["loss"].append(round(avg_loss,  6))
        history["recon_loss"].append(round(avg_recon, 6))
        history["l0"].append(round(avg_l0, 2))

        if ep % log_every == 0 or ep == 1 or ep == epochs:
            log.info(
                f"  epoch {ep:3d}/{epochs}  loss={avg_loss:.5f}  "
                f"recon={avg_recon:.5f}  L0={avg_l0:.1f}"
            )

    model.eval()
    return model, history


# ==============================================================================
# Helpers (mirror train_bsf.py so activation scale is identical)
# ==============================================================================

def centre_and_scale_chunked(
    features_list: list,
    pos_mean: np.ndarray,
    max_patches: int = MAX_TRAIN_PATCHES,
    rms_sample: int = 50_000,
) -> np.ndarray:
    chunks16 = []
    for feat in features_list:
        c = feat.tokens.astype(np.float32) - pos_mean
        chunks16.append(c.reshape(-1, c.shape[-1]).astype(np.float16))
    flat16 = np.concatenate(chunks16, axis=0)
    n_total = len(flat16)

    idx_rms    = np.random.choice(n_total, min(rms_sample, n_total), replace=False)
    sample_f32 = flat16[idx_rms].astype(np.float32)
    rms        = float(np.sqrt((sample_f32 ** 2).sum(axis=1).mean()))
    scale      = np.sqrt(sample_f32.shape[1]) / max(rms, 1e-8)

    if n_total > max_patches:
        log.info(f"  Subsampling {n_total:,} -> {max_patches:,} patches for RAM budget")
        idx_sub = np.random.choice(n_total, max_patches, replace=False)
        flat16  = flat16[idx_sub]

    return flat16.astype(np.float32) * scale


def best_f1(y_true: np.ndarray, y_score: np.ndarray) -> float:
    prec, rec, _ = precision_recall_curve(y_true, y_score)
    denom = prec + rec
    f1 = np.where(denom > 0, 2 * prec * rec / denom, 0.0)
    return float(f1.max())


def aupro(
    gt_masks: list[np.ndarray],
    pred_maps: list[np.ndarray],
    max_fpr: float = 0.3,
) -> float:
    """Memory-efficient AUPRO (identical to train_bsf.py)."""
    from skimage.measure import label as cc_label

    regions_sorted = []
    normal_samples = []

    for mask, pred in zip(gt_masks, pred_maps):
        if mask.sum() > 0:
            lbl = cc_label(mask > 0)
            for r in range(1, int(lbl.max()) + 1):
                r_scores = pred[lbl == r]
                if len(r_scores) > 0:
                    regions_sorted.append(np.sort(r_scores))
        norm_pix = pred[mask == 0]
        if len(norm_pix) > 5000:
            norm_pix = np.random.choice(norm_pix, 5000, replace=False)
        if len(norm_pix) > 0:
            normal_samples.append(norm_pix)

    if not regions_sorted or not normal_samples:
        return 0.0

    normal_scores_sorted = np.sort(np.concatenate(normal_samples))
    n_normal  = len(normal_scores_sorted)
    thresholds = np.percentile(normal_scores_sorted, np.linspace(0, 100, 300))

    pro_list, fpr_list = [], []
    for t in thresholds:
        fp  = n_normal - np.searchsorted(normal_scores_sorted, t)
        fpr = fp / max(n_normal, 1)
        overlaps = [(len(r) - np.searchsorted(r, t)) / len(r) for r in regions_sorted]
        fpr_list.append(fpr)
        pro_list.append(float(np.mean(overlaps)) if overlaps else 0.0)

    fpr_arr = np.array(fpr_list)
    pro_arr = np.array(pro_list)
    order   = np.argsort(fpr_arr)
    fpr_arr, pro_arr = fpr_arr[order], pro_arr[order]

    mask_fpr = fpr_arr <= max_fpr
    if mask_fpr.sum() < 2:
        return 0.0
    return float(np.trapezoid(pro_arr[mask_fpr], fpr_arr[mask_fpr]) / max_fpr)


@torch.inference_mode()
def compute_recon_error(
    model: BatchTopKSAE,
    tokens_flat: np.ndarray,
    device: str,
    batch_size: int = 4096,
) -> np.ndarray:
    errors = []
    x = torch.as_tensor(tokens_flat, dtype=torch.float32)
    for start in range(0, len(x), batch_size):
        xb = x[start : start + batch_size].to(device)
        x_hat, _ = model(xb)
        mse = (xb - x_hat).pow(2).mean(dim=-1).cpu().numpy()
        errors.append(mse)
    return np.concatenate(errors)


def _detect_elbow(curve: np.ndarray, min_k: int = 1) -> int:
    """Maximum-distance-to-chord elbow detector.

    NumPy-2.x-safe: uses explicit scalar cross-product instead of np.cross
    on 2-D vectors (which was removed in NumPy 2.0).
    """
    n = len(curve)
    if n <= min_k + 1:
        return n - 1
    x = np.arange(n, dtype=float)
    y = np.asarray(curve, dtype=float)
    p0 = np.array([x[0], y[0]])
    p1 = np.array([x[-1], y[-1]])
    line_vec = p1 - p0
    line_len = np.linalg.norm(line_vec)
    if line_len < 1e-10:
        return n - 1
    line_unit = line_vec / line_len   # 2-D unit vector [ux, uy]
    # Vector from p0 to each point: shape (n, 2)
    diff = np.column_stack([x, y]) - p0
    # 2-D cross product z-component: ux*dy - uy*dx
    dists = np.abs(line_unit[0] * diff[:, 1] - line_unit[1] * diff[:, 0])
    dists[:min_k] = -1
    return int(np.argmax(dists))


def _find_support_greedy_local(
    activations: np.ndarray,
    decoder: np.ndarray,
    max_k: int = 100,
    var_threshold: float = 0.95,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Greedy subspace pursuit over decoder directions (NumPy-2.x-safe local copy).

    Mirrors vendor/sae-manifold/subspace_capture.py::find_support_greedy but
    replaces the broken np.cross call with a scalar formula.
    """
    X = np.asarray(activations, dtype=np.float32)
    X = X - X.mean(0)
    total_ss = (X ** 2).sum()
    if total_ss < 1e-10:
        return np.array([], dtype=int), np.array([]), 0

    candidates  = np.arange(decoder.shape[0])
    D_cand      = decoder[candidates]
    d_norms_sq  = (D_cand ** 2).sum(1)
    alive       = d_norms_sq > 1e-10

    selected_local:  list[int] = []
    selected_global: list[int] = []
    var_curve:       list[float] = []
    residual = X.copy()

    for _ in range(max_k):
        projections = residual @ D_cand.T
        scores = (projections ** 2).sum(0) / d_norms_sq.clip(1e-10)
        scores[~alive] = -np.inf
        for i in selected_local:
            scores[i] = -np.inf
        best = int(np.argmax(scores))
        if scores[best] <= 0:
            break
        selected_local.append(best)
        selected_global.append(int(candidates[best]))
        D_sel = decoder[selected_global]
        _, s, Vt = np.linalg.svd(D_sel, full_matrices=False)
        basis    = Vt[s > 1e-8]
        residual = X - (X @ basis.T) @ basis
        explained = 1.0 - (residual ** 2).sum() / total_ss
        var_curve.append(float(explained))
        if explained >= var_threshold:
            break

    vc = np.array(var_curve)
    elbow_k = (_detect_elbow(vc, min_k=1) + 1 if len(vc) > 2 else len(vc))
    return np.array(selected_global), vc, elbow_k


def compute_subspace_capture(
    model: BatchTopKSAE,
    normal_X: np.ndarray,
    device: str,
    max_k: int = 64,
    subsample: int = 10_000,
) -> dict:
    """
    Geometric subspace capture via greedy decoder-direction selection.

    Returns: k_for_95pct, elbow_k, var_at_k16, and the full variance curve.
    """
    if len(normal_X) > subsample:
        idx   = np.random.choice(len(normal_X), subsample, replace=False)
        X_sub = normal_X[idx]
    else:
        X_sub = normal_X

    decoder = get_decoder(model)   # (d_sae, d_in)

    _, var_curve, elbow_k = _find_support_greedy_local(
        X_sub, decoder, max_k=max_k, var_threshold=0.95
    )

    arr  = np.array(var_curve)
    hits = np.where(arr >= 0.95)[0]
    k_for_95pct = int(hits[0] + 1) if len(hits) > 0 else max_k + 1
    var_at_k16  = float(arr[15]) if len(arr) >= 16 else (float(arr[-1]) if len(arr) > 0 else 0.0)

    return {
        "k_for_95pct": k_for_95pct,
        "elbow_k":     int(elbow_k),
        "var_at_k16":  round(var_at_k16, 4),
        "curve":       [round(float(v), 4) for v in arr],
    }


# ==============================================================================
# Plots
# ==============================================================================

def plot_training_curves(history: dict, object_class: str, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle(f"SAE Training -- {object_class}", fontsize=13, fontweight="bold")
    epochs = history["epoch"]

    axes[0].plot(epochs, history["loss"], "b-o", ms=4)
    axes[0].set_title("Total Loss"); axes[0].set_xlabel("Epoch"); axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history["recon_loss"], "g-o", ms=4)
    axes[1].set_title("Reconstruction MSE"); axes[1].set_xlabel("Epoch"); axes[1].grid(True, alpha=0.3)

    axes[2].plot(epochs, history["l0"], "r-o", ms=4)
    axes[2].set_title("Mean Active Features (L0)")
    axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("L0"); axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    p = out_dir / f"{object_class}_sae_training_curves.png"
    plt.savefig(p, dpi=150); plt.close(fig)
    log.info(f"  Saved training curves -> {p}")


def plot_score_distributions(
    normal_scores: np.ndarray, anomaly_scores: np.ndarray,
    object_class: str, out_dir: Path
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(normal_scores,  bins=50, alpha=0.6, label="Normal",  color="steelblue")
    ax.hist(anomaly_scores, bins=50, alpha=0.6, label="Anomaly", color="tomato")
    ax.set_title(f"SAE Score Distribution -- {object_class}")
    ax.set_xlabel("Mean Reconstruction Error"); ax.set_ylabel("Count")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    p = out_dir / f"{object_class}_sae_score_distribution.png"
    plt.savefig(p, dpi=150); plt.close(fig)
    log.info(f"  Saved score distribution -> {p}")


def plot_subspace_capture(curve: list, object_class: str, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ks = list(range(1, len(curve) + 1))
    ax.plot(ks, curve, "b-o", ms=4, label="SAE (greedy)")
    ax.axhline(0.95, color="red", linestyle="--", alpha=0.7, label="95% threshold")
    ax.set_title(f"Subspace Capture -- {object_class}")
    ax.set_xlabel("Decoder atoms selected (k)"); ax.set_ylabel("Variance explained")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    p = out_dir / f"{object_class}_subspace_capture.png"
    plt.savefig(p, dpi=150); plt.close(fig)
    log.info(f"  Saved subspace capture -> {p}")


# ==============================================================================
# Core per-class pipeline
# ==============================================================================

def run_class(
    object_class: str,
    all_samples,
    runner: BackboneRunner,
    pos_mean: np.ndarray,
    args: argparse.Namespace,
) -> dict:
    log.info(f"{'='*60}")
    log.info(f"CLASS: {object_class}")
    log.info(f"{'='*60}")

    train_normals = [s for s in all_samples
                     if s.object_class == object_class and s.split == "train"][:args.train]
    test_samples  = [s for s in all_samples
                     if s.object_class == object_class and s.split == "test"]

    log.info(f"  Train normals : {len(train_normals)}")
    log.info(f"  Test samples  : {len(test_samples)}")

    device     = runner.device
    weight_path = SAE_WEIGHTS / f"{object_class}.pt"
    d_in       = runner.spec.dim
    d_sae      = args.n_groups * args.group_size
    k          = args.l0

    subspace: dict = {}

    if getattr(args, "resume", False) and weight_path.is_file():
        log.info(f"  [Resume] Loading weights from {weight_path}...")
        model = BatchTopKSAE(d_in=d_in, d_sae=d_sae, k=k)
        model.load_state_dict(torch.load(weight_path, map_location=device, weights_only=True))
        model = model.to(device)
        model.eval()
        train_time = 0.0
        # compute subspace capture on a small subset of test normals
        log.info("  Computing subspace capture on test normals (resume mode)...")
        normal_test = [s for s in test_samples if not s.is_anomalous][:20]
        if normal_test:
            nf = runner.extract_paths([s.image for s in normal_test], progress=False)
            X_norm = centre_and_scale_chunked(nf, pos_mean, max_patches=50_000)
            subspace = compute_subspace_capture(model, X_norm, device)
        else:
            subspace = {"k_for_95pct": None, "elbow_k": None, "var_at_k16": None, "curve": []}
    else:
        # -- extract training tokens ------------------------------------------
        log.info("  Extracting training features...")
        t0 = time.perf_counter()
        train_features = runner.extract_paths(
            [s.image for s in train_normals], progress=True)
        log.info(f"  Feature extraction: {time.perf_counter()-t0:.1f}s")

        X = centre_and_scale_chunked(train_features, pos_mean)
        log.info(f"  Training tensor: {X.shape}  ({X.shape[0]:,} patches x {X.shape[1]} dims)")
        log.info(f"  SAE config: d_sae={d_sae} (n_groups={args.n_groups} x group_size={args.group_size}), k={k}")

        # -- train SAE --------------------------------------------------------
        model = BatchTopKSAE(d_in=d_in, d_sae=d_sae, k=k)
        t0 = time.perf_counter()
        model, history = train_sae(
            model, X,
            epochs=args.epochs,
            batch_size=args.batch_size,
            device=device,
            log_every=max(1, args.epochs // 10),
        )
        train_time = time.perf_counter() - t0
        log.info(f"  Training complete in {train_time:.1f}s")

        torch.save(model.state_dict(), weight_path)
        log.info(f"  Saved weights -> {weight_path}")

        log_path = SAE_LOGS / f"{object_class}_history.jsonl"
        with log_path.open("w") as fh:
            for i, ep in enumerate(history["epoch"]):
                fh.write(json.dumps({
                    "epoch": ep, "loss": history["loss"][i],
                    "recon_loss": history["recon_loss"][i], "l0": history["l0"][i],
                }) + "\n")

        plot_training_curves(history, object_class, SAE_PLOTS)

        # -- subspace capture on training normals -----------------------------
        log.info("  Computing subspace capture...")
        subspace = compute_subspace_capture(model, X, device)
        log.info(
            f"  Subspace: k_for_95pct={subspace['k_for_95pct']}  "
            f"elbow={subspace['elbow_k']}  var@k16={subspace['var_at_k16']}"
        )
        plot_subspace_capture(subspace["curve"], object_class, SAE_PLOTS)

    # -- evaluate on test split -----------------------------------------------
    log.info("  Evaluating on test split...")
    img_labels, img_scores = [], []
    all_pix_labels, all_pix_scores = [], []
    all_gt_masks, all_pred_maps = [], []

    for sample in test_samples:
        raw_img = cv2.imread(str(sample.image))
        if raw_img is None:
            log.warning(f"Could not read {sample.image}, skipping")
            continue
        feats   = runner.extract(raw_img)
        n_tiles = feats.tokens.shape[0]

        flat = centre_and_scale(feats.tokens, pos_mean)
        err  = compute_recon_error(model, flat, device)

        img_score = float(err.mean())
        img_labels.append(int(sample.is_anomalous))
        img_scores.append(img_score)

        grid = feats.grid
        patch_maps = [err[i * grid**2 : (i+1) * grid**2].reshape(grid, grid)
                      for i in range(n_tiles)]

        from parallax.tiling import stitch
        score_map = stitch(patch_maps, feats.tiles, feats.shape)

        if sample.mask is not None and sample.is_anomalous:
            gt_mask = cv2.imread(str(sample.mask), cv2.IMREAD_GRAYSCALE)
            if gt_mask is not None:
                gt_mask_bin = (gt_mask > 0).astype(np.uint8)
                sm_resized  = cv2.resize(
                    score_map.astype(np.float32),
                    (gt_mask_bin.shape[1], gt_mask_bin.shape[0]),
                    interpolation=cv2.INTER_LINEAR,
                )
                all_pix_labels.append(gt_mask_bin.ravel())
                all_pix_scores.append(sm_resized.ravel())
                all_gt_masks.append(gt_mask_bin)
                all_pred_maps.append(sm_resized)

    img_labels = np.array(img_labels)
    img_scores = np.array(img_scores)

    metrics: dict = {
        "class":          object_class,
        "model":          "sae",
        "d_sae":          d_sae,
        "k":              k,
        "train_normals":  len(train_normals),
        "test_samples":   len(test_samples),
        "train_time_s":   round(train_time, 1),
        "img_auroc":      None,
        "img_ap":         None,
        "img_f1_best":    None,
        "pix_auroc":      None,
        "pix_ap":         None,
        "pix_f1_best":    None,
        "aupro":          None,
        "subspace_k95":   subspace.get("k_for_95pct"),
        "subspace_elbow": subspace.get("elbow_k"),
        "var_at_k16":     subspace.get("var_at_k16"),
    }

    if len(set(img_labels)) == 2:
        metrics["img_auroc"]   = round(float(roc_auc_score(img_labels, img_scores)), 4)
        metrics["img_ap"]      = round(float(average_precision_score(img_labels, img_scores)), 4)
        metrics["img_f1_best"] = round(best_f1(img_labels, img_scores), 4)
        plot_score_distributions(
            img_scores[img_labels == 0], img_scores[img_labels == 1],
            object_class, SAE_PLOTS
        )

    if all_pix_labels:
        pix_labels = np.concatenate(all_pix_labels)
        pix_scores = np.concatenate(all_pix_scores)
        if len(set(pix_labels)) == 2:
            metrics["pix_auroc"]   = round(float(roc_auc_score(pix_labels, pix_scores)), 4)
            metrics["pix_ap"]      = round(float(average_precision_score(pix_labels, pix_scores)), 4)
            metrics["pix_f1_best"] = round(best_f1(pix_labels, pix_scores), 4)
        metrics["aupro"] = round(aupro(all_gt_masks, all_pred_maps), 4)

    log.info(f"  img AUROC={metrics['img_auroc']}  img AP={metrics['img_ap']}")
    log.info(f"  pix AUROC={metrics['pix_auroc']}  AUPRO={metrics['aupro']}")
    log.info(
        f"  subspace k@95%={metrics['subspace_k95']}  "
        f"elbow={metrics['subspace_elbow']}  var@k16={metrics['var_at_k16']}"
    )

    metric_path = SAE_LOGS / f"{object_class}_metrics.json"
    with metric_path.open("w") as fh:
        json.dump(metrics, fh, indent=2)
    log.info(f"  Saved metrics -> {metric_path}")

    return metrics


# ==============================================================================
# Comparison report
# ==============================================================================

def load_probe_aurocs() -> dict[str, float]:
    """Read Linear Probe AUROCs from the probe training log."""
    probe_log = PROJECT_ROOT / "logs" / "probe_train.log"
    aurocs: dict[str, float] = {}
    if not probe_log.exists():
        return aurocs
    for line in probe_log.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 8 and parts[0] in RIGID_CLASSES:
            try:
                aurocs[parts[0]] = float(parts[7])
            except (ValueError, IndexError):
                pass
    return aurocs


def generate_comparison_report(
    sae_metrics: list[dict],
    bsf_summary_path: Path,
    probe_aurocs: dict[str, float],
) -> None:
    """Write logs/stage4_comparison.md and .json."""
    bsf_by_class: dict[str, dict] = {}
    if bsf_summary_path.exists():
        with bsf_summary_path.open() as f:
            for m in json.load(f):
                bsf_by_class[m["class"]] = m

    sae_by_class = {m["class"]: m for m in sae_metrics}

    rows = []
    for cls in RIGID_CLASSES:
        bsf = bsf_by_class.get(cls, {})
        sae = sae_by_class.get(cls, {})
        rows.append({
            "class":         cls,
            "probe_auroc":   probe_aurocs.get(cls),
            "bsf_img_auroc": bsf.get("img_auroc"),
            "sae_img_auroc": sae.get("img_auroc"),
            "bsf_pix_auroc": bsf.get("pix_auroc"),
            "sae_pix_auroc": sae.get("pix_auroc"),
            "bsf_aupro":     bsf.get("aupro"),
            "sae_aupro":     sae.get("aupro"),
            "sae_k95":       sae.get("subspace_k95"),
            "sae_var_k16":   sae.get("var_at_k16"),
        })

    json_path = PROJECT_ROOT / "logs" / "stage4_comparison.json"
    with json_path.open("w") as f:
        json.dump(rows, f, indent=2)

    def fmt(v) -> str:
        return "—" if v is None else f"{v:.4f}"

    def winner(a, b) -> str:
        if a is None or b is None:
            return ""
        return " **BSF**" if a > b else (" **SAE**" if b > a else "")

    bsf_img_wins = sum(1 for r in rows
                       if r["bsf_img_auroc"] and r["sae_img_auroc"]
                       and r["bsf_img_auroc"] > r["sae_img_auroc"])
    sae_img_wins = sum(1 for r in rows
                       if r["bsf_img_auroc"] and r["sae_img_auroc"]
                       and r["sae_img_auroc"] > r["bsf_img_auroc"])
    bsf_pix_wins = sum(1 for r in rows
                       if r["bsf_pix_auroc"] and r["sae_pix_auroc"]
                       and r["bsf_pix_auroc"] > r["sae_pix_auroc"])
    sae_pix_wins = sum(1 for r in rows
                       if r["bsf_pix_auroc"] and r["sae_pix_auroc"]
                       and r["sae_pix_auroc"] > r["bsf_pix_auroc"])

    lines = [
        "# Stage 4 — Probe Comparison Report",
        "",
        f"*Generated {time.strftime('%Y-%m-%d %H:%M')}*",
        "",
        "## Models",
        "",
        "| Model | Description |",
        "|---|---|",
        "| **Linear Probe** | Mahalanobis distance on DINOv3 activations — the guaranteed floor |",
        "| **GrassmannianBSF** | Block-sparse featurizer, 512 groups x 3D, L0=16 |",
        "| **BatchTopKSAE** | SAE baseline, d_sae=1536, k=16 (BSF-matched capacity) |",
        "",
        "## Image-level AUROC",
        "",
        "| Class | Linear Probe | GrassmannianBSF | BatchTopKSAE | Winner |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        w = winner(r["bsf_img_auroc"], r["sae_img_auroc"])
        lines.append(
            f"| {r['class']} | {fmt(r['probe_auroc'])} | {fmt(r['bsf_img_auroc'])} "
            f"| {fmt(r['sae_img_auroc'])} |{w} |"
        )
    lines += [
        f"| **Wins** | — | **{bsf_img_wins}** | **{sae_img_wins}** | |",
        "",
        "## Pixel-level AUROC",
        "",
        "| Class | GrassmannianBSF | BatchTopKSAE | Winner |",
        "|---|---|---|---|",
    ]
    for r in rows:
        w = winner(r["bsf_pix_auroc"], r["sae_pix_auroc"])
        lines.append(
            f"| {r['class']} | {fmt(r['bsf_pix_auroc'])} | {fmt(r['sae_pix_auroc'])} |{w} |"
        )
    lines += [
        f"| **Wins** | **{bsf_pix_wins}** | **{sae_pix_wins}** | |",
        "",
        "## AUPRO",
        "",
        "| Class | GrassmannianBSF | BatchTopKSAE |",
        "|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['class']} | {fmt(r['bsf_aupro'])} | {fmt(r['sae_aupro'])} |"
        )
    lines += [
        "",
        "## Subspace capture (SAE)",
        "",
        "> `k_for_95pct`: decoder atoms needed to explain 95% of normal-patch variance.",
        "> BSF always uses exactly 3 dimensions per concept (group_size=3).",
        "> So the comparable BSF count is `k_for_95pct / 3` blocks.",
        "> A large `k_for_95pct` confirms the SAE manifold-dilution finding from the paper.",
        "",
        "| Class | k @ 95% | Var @ k=16 |",
        "|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['class']} | {r['sae_k95'] if r['sae_k95'] is not None else '—'} "
            f"| {fmt(r['sae_var_k16'])} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        f"- **Image AUROC**: BSF wins {bsf_img_wins}/{len(rows)} classes over SAE.",
        f"- **Pixel AUROC**: BSF wins {bsf_pix_wins}/{len(rows)} classes over SAE.",
        "- **Why we keep BSF even on ties**: unlike an SAE, BSF outputs a *block coordinate*",
        "  (a vector inside each concept's subspace), enabling the escalation UI to explain",
        "  *where within a concept* the image sits — e.g. 'cracked end of weld-seam manifold'",
        "  vs just 'weld-seam feature fired'.",
        "- **Subspace capture** measures the companion-paper dilution claim quantitatively:",
        "  if `k_for_95pct` is large, the SAE scatters the normal manifold across many atoms",
        "  where BSF captures each concept in a 3D block.",
    ]

    md_path = PROJECT_ROOT / "logs" / "stage4_comparison.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    log.info(f"Saved comparison report -> {md_path}")

    # stdout summary
    print()
    print("=" * 105)
    print(
        f"{'Class':<14}  {'Probe':<7}  {'BSF_img':<9}  {'SAE_img':<9}  "
        f"{'BSF_pix':<9}  {'SAE_pix':<9}  {'BSF_pro':<8}  {'SAE_pro':<8}  {'k@95%':<6}"
    )
    print("-" * 105)
    for r in rows:
        print(
            f"{r['class']:<14}  "
            f"{str(r['probe_auroc']):<7}  "
            f"{str(r['bsf_img_auroc']):<9}  "
            f"{str(r['sae_img_auroc']):<9}  "
            f"{str(r['bsf_pix_auroc']):<9}  "
            f"{str(r['sae_pix_auroc']):<9}  "
            f"{str(r['bsf_aupro']):<8}  "
            f"{str(r['sae_aupro']):<8}  "
            f"{str(r['sae_k95']):<6}"
        )
    print("=" * 105)
    print(f"  BSF img-AUROC wins: {bsf_img_wins}/{len(rows)}   SAE: {sae_img_wins}/{len(rows)}")
    print(f"  BSF pix-AUROC wins: {bsf_pix_wins}/{len(rows)}   SAE: {sae_pix_wins}/{len(rows)}")
    print(f"\n  Full report -> {md_path}")


# ==============================================================================
# Entry point
# ==============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train SAE baseline and compare against BSF on VisA rigid classes.")
    parser.add_argument("--train",       type=int, default=80)
    parser.add_argument("--epochs",      type=int, default=40)
    parser.add_argument("--batch-size",  type=int, default=2048)
    parser.add_argument("--n-groups",    type=int, default=512,
                        help="BSF n_groups — d_sae = n_groups * group_size")
    parser.add_argument("--group-size",  type=int, default=3,
                        help="BSF group_size — d_sae = n_groups * group_size")
    parser.add_argument("--l0",          type=int, default=16,
                        help="SAE k (matches BSF l0)")
    parser.add_argument("--classes",     nargs="+", default=list(RIGID_CLASSES))
    parser.add_argument("--resume",      action="store_true",
                        help="skip training if weights exist")
    args = parser.parse_args()

    for d in (SAE_WEIGHTS, SAE_LOGS, SAE_PLOTS):
        d.mkdir(parents=True, exist_ok=True)

    fh = logging.FileHandler(SAE_LOGS / "full_training.log", mode="a", encoding="utf-8")
    fh.setFormatter(logging.Formatter(
        "%(asctime)s  %(levelname)-7s  %(message)s", datefmt="%H:%M:%S"))
    logging.getLogger().addHandler(fh)

    log.info("=" * 60)
    log.info("Starting SAE run  classes=%s  resume=%s", args.classes, args.resume)
    log.info("d_sae=%d  k=%d", args.n_groups * args.group_size, args.l0)
    log.info("=" * 60)

    all_samples = ingest(rigid_only=True)
    log.info(f"  {len(all_samples)} samples indexed")

    runner = BackboneRunner()
    log.info(f"  {runner.describe()}")

    pos_mean = np.load(POS_MEAN_PATH).astype(np.float32)

    all_metrics = []
    for object_class in args.classes:
        m = run_class(object_class, all_samples, runner, pos_mean, args)
        all_metrics.append(m)

    # aggregate SAE summary
    existing: dict[str, dict] = {}
    for mf in SAE_LOGS.glob("*_metrics.json"):
        try:
            with mf.open() as f:
                c = json.load(f)
                if isinstance(c, dict) and "class" in c:
                    existing[c["class"]] = c
        except Exception:
            pass
    for m in all_metrics:
        existing[m["class"]] = m
    merged = [existing[c] for c in RIGID_CLASSES if c in existing]

    sae_summary = SAE_LOGS / "summary.json"
    with sae_summary.open("w") as f:
        json.dump(merged, f, indent=2)
    log.info(f"Saved SAE summary -> {sae_summary}")

    probe_aurocs = load_probe_aurocs()
    generate_comparison_report(merged, BSF_LOGS / "summary.json", probe_aurocs)


if __name__ == "__main__":
    main()
