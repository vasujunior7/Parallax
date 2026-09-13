"""Train the Block-Sparse Featurizer (BSF) per rigid VisA class.

For every rigid class this script will:
  1. Extract DINOv3 patch tokens from *normal* training images.
  2. Normalise them (centre + RMS scale) per the BSF paper convention.
  3. Train a GrassmannianBSF (default 40 epochs).
  4. Log per-epoch loss / R2 / L0 / dead-blocks to a JSON-Lines file.
  5. Plot and save four training curves (loss, R2, L0, dead-blocks).
  6. Run a full anomaly-score evaluation on the test split:
       - Image-level  AUROC, AP (average precision), F1-best
       - Pixel-level  AUROC, AP, F1-best
       - Per-class AUPRO  (area under per-region overlap curve)
       - Per-patch mean reconstruction error heat-map (saved as PNG)
  7. Dump all metrics as JSON.
  8. Print a final summary table.

Run:
  uv run python scripts/train_bsf.py
  uv run python scripts/train_bsf.py --epochs 20 --train 50  # quick test
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path

if sys.platform == "win32":
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "buffer"):
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")          # no display needed
import matplotlib.pyplot as plt

# ── project imports ────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from parallax.backbone import centre_and_scale
from parallax.features import BackboneRunner
from parallax.visa import RIGID_CLASSES, ingest

# ── vendor BSF ─────────────────────────────────────────────────────────────────
VENDOR_BSF = PROJECT_ROOT / "vendor" / "block-sparse-featurizer"
if str(VENDOR_BSF) not in sys.path:
    sys.path.append(str(VENDOR_BSF))

import bsf                                 # noqa: E402
from bsf.train import recon_r2, l0_dead   # noqa: E402

# ── sklearn metrics ─────────────────────────────────────────────────────────────
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
)

# ── output directories ─────────────────────────────────────────────────────────
BSF_WEIGHTS   = PROJECT_ROOT / "data"  / "bsf"
BSF_LOGS      = PROJECT_ROOT / "logs"  / "bsf"
BSF_PLOTS     = BSF_LOGS / "plots"
BSF_HEATMAPS  = BSF_LOGS / "heatmaps"

POS_MEAN_PATH = VENDOR_BSF / "bsf" / "pos_mean.npy"

# ── logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("train_bsf")

# Maximum patches to feed into BSF training (caps RAM at ~1.5 GB float32)
MAX_TRAIN_PATCHES = 500_000


# ──────────────────────────────────────────────────────────────────────────────
# Memory-efficient normalisation
# ──────────────────────────────────────────────────────────────────────────────

def centre_and_scale_chunked(
    features_list: list,
    pos_mean: np.ndarray,
    max_patches: int = MAX_TRAIN_PATCHES,
    rms_sample: int = 50_000,
) -> np.ndarray:
    """Centre & scale tokens without materialising a full float32 copy of everything.

    Strategy:
      1. Subtract pos_mean per feature and store as **float16** (halves RAM).
      2. Compute RMS from a random subsample of 50k patches (negligible RAM).
      3. Randomly subsample down to `max_patches` to cap float32 training array.
      4. Return float32 scaled result (~1.5 GB at 500k patches × 768 dims).
    """
    # ── step 1: subtract pos_mean, store float16 ─────────────────────────────
    chunks16 = []
    for feat in features_list:
        c = feat.tokens.astype(np.float32) - pos_mean   # (tiles, 196, 768)
        chunks16.append(c.reshape(-1, c.shape[-1]).astype(np.float16))
    flat16 = np.concatenate(chunks16, axis=0)           # (N, 768) float16
    n_total = len(flat16)

    # ── step 2: compute RMS on a subsample ───────────────────────────────────
    idx_rms = np.random.choice(n_total, min(rms_sample, n_total), replace=False)
    sample_f32 = flat16[idx_rms].astype(np.float32)
    rms = float(np.sqrt((sample_f32 ** 2).sum(axis=1).mean()))
    scale = np.sqrt(sample_f32.shape[1]) / max(rms, 1e-8)

    # ── step 3: subsample if too many patches ────────────────────────────────
    if n_total > max_patches:
        log.info(f"  Subsampling {n_total:,} → {max_patches:,} patches for RAM budget")
        idx_sub = np.random.choice(n_total, max_patches, replace=False)
        flat16 = flat16[idx_sub]

    # ── step 4: float32 + scale ───────────────────────────────────────────────
    return flat16.astype(np.float32) * scale


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def best_f1(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Find F1 at the optimal threshold."""
    prec, rec, _ = precision_recall_curve(y_true, y_score)
    denom = prec + rec
    f1 = np.where(denom > 0, 2 * prec * rec / denom, 0.0)
    return float(f1.max())


def aupro(gt_masks: list[np.ndarray], pred_maps: list[np.ndarray], max_fpr: float = 0.3) -> float:
    """Area Under the Per-Region Overlap curve (truncated at max_fpr).

    Iterates over 2D anomalous connected components per image, computes
    per-component TPR at each threshold, and averages — then integrates
    the mean-TPR vs FPR curve up to `max_fpr`.
    """
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

        # Subsample normal pixels from each anomalous frame to cap memory
        norm_pix = pred[mask == 0]
        if len(norm_pix) > 5000:
            norm_pix = np.random.choice(norm_pix, 5000, replace=False)
        if len(norm_pix) > 0:
            normal_samples.append(norm_pix)

    if not regions_sorted or not normal_samples:
        return 0.0

    normal_scores_sorted = np.sort(np.concatenate(normal_samples))
    n_normal = len(normal_scores_sorted)

    thresholds = np.percentile(normal_scores_sorted, np.linspace(0, 100, 300))

    pro_list, fpr_list = [], []
    for t in thresholds:
        fp = n_normal - np.searchsorted(normal_scores_sorted, t)
        fpr = fp / max(n_normal, 1)

        overlaps = [(len(r) - np.searchsorted(r, t)) / len(r) for r in regions_sorted]
        mean_pro = float(np.mean(overlaps)) if overlaps else 0.0

        fpr_list.append(fpr)
        pro_list.append(mean_pro)

    fpr_arr = np.array(fpr_list)
    pro_arr = np.array(pro_list)
    order = np.argsort(fpr_arr)
    fpr_arr, pro_arr = fpr_arr[order], pro_arr[order]

    mask_fpr = fpr_arr <= max_fpr
    if mask_fpr.sum() < 2:
        return 0.0
    return float(np.trapezoid(pro_arr[mask_fpr], fpr_arr[mask_fpr]) / max_fpr)


def plot_training_curves(history: dict, object_class: str, out_dir: Path) -> None:
    """Save a 2×2 grid of training-curve plots for one class."""
    epochs = history["epoch"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"BSF Training — {object_class}", fontsize=14, fontweight="bold")

    # ── Loss ─────────────────────────────────────────────────
    ax = axes[0, 0]
    ax.plot(epochs, history["loss"], "b-o", ms=4)
    ax.set_title("Training Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)

    # ── R² ───────────────────────────────────────────────────
    ax = axes[0, 1]
    ax.plot(epochs, history["r2"], "g-o", ms=4)
    ax.set_title("Reconstruction R²")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("R²")
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)

    # ── L0 (mean active blocks) ───────────────────────────────
    ax = axes[1, 0]
    ax.plot(epochs, history["l0"], "r-o", ms=4)
    ax.set_title("Mean Active Blocks (L0)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("L0")
    ax.grid(True, alpha=0.3)

    # ── Dead blocks ───────────────────────────────────────────
    ax = axes[1, 1]
    ax.plot(epochs, history["dead"], "m-o", ms=4)
    ax.set_title("Dead Blocks (never fired)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("# Dead")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = out_dir / f"{object_class}_training_curves.png"
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    log.info(f"  Saved training curves → {save_path}")


def plot_score_distributions(
    normal_scores: np.ndarray, anomaly_scores: np.ndarray,
    object_class: str, out_dir: Path
) -> None:
    """Histogram of image-level anomaly scores for normal vs anomaly."""
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = 50
    ax.hist(normal_scores,  bins=bins, alpha=0.6, label="Normal",  color="steelblue")
    ax.hist(anomaly_scores, bins=bins, alpha=0.6, label="Anomaly", color="tomato")
    ax.set_title(f"Image-level Score Distribution — {object_class}")
    ax.set_xlabel("Mean Reconstruction Error")
    ax.set_ylabel("Count")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = out_dir / f"{object_class}_score_distribution.png"
    plt.savefig(path, dpi=150)
    plt.close(fig)
    log.info(f"  Saved score distribution → {path}")


def plot_roc_pr(
    img_labels: np.ndarray, img_scores: np.ndarray,
    pix_labels: np.ndarray, pix_scores: np.ndarray,
    object_class: str, out_dir: Path
) -> None:
    """ROC and PR curves at image and pixel level."""
    from sklearn.metrics import roc_curve

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"ROC & PR Curves — {object_class}", fontsize=13, fontweight="bold")

    # image-level
    for level, labels, scores, color in [
        ("Image", img_labels, img_scores, "dodgerblue"),
        ("Pixel", pix_labels, pix_scores, "tomato"),
    ]:
        fpr, tpr, _ = roc_curve(labels, scores)
        auroc = roc_auc_score(labels, scores)
        axes[0].plot(fpr, tpr, color=color, label=f"{level} AUROC={auroc:.3f}")

        prec, rec, _ = precision_recall_curve(labels, scores)
        ap = average_precision_score(labels, scores)
        axes[1].plot(rec, prec, color=color, label=f"{level} AP={ap:.3f}")

    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.3)
    axes[0].set_title("ROC Curve")
    axes[0].set_xlabel("FPR")
    axes[0].set_ylabel("TPR")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title("Precision-Recall Curve")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = out_dir / f"{object_class}_roc_pr.png"
    plt.savefig(path, dpi=150)
    plt.close(fig)
    log.info(f"  Saved ROC/PR curves → {path}")


def save_heatmap(
    score_map: np.ndarray, image_path: Path,
    object_class: str, img_name: str, out_dir: Path
) -> None:
    """Overlay reconstruction-error heatmap on the source image."""
    img = cv2.imread(str(image_path))
    if img is None:
        return

    h, w = img.shape[:2]
    sm = (score_map - score_map.min()) / ((score_map.max() - score_map.min()) + 1e-8)
    heatmap = cv2.applyColorMap((sm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.resize(heatmap, (w, h))
    overlay = cv2.addWeighted(img, 0.5, heatmap, 0.5, 0)

    class_dir = out_dir / object_class
    class_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(class_dir / f"{img_name}_heatmap.jpg"), overlay)


# ──────────────────────────────────────────────────────────────────────────────
# Core per-class pipeline
# ──────────────────────────────────────────────────────────────────────────────

@torch.inference_mode()
def compute_recon_error(
    model: bsf.BSF,
    tokens_flat: np.ndarray,
    device: str,
    batch_size: int = 4096,
) -> np.ndarray:
    """Reconstruction MSE per patch, returned as a flat (N,) numpy array."""
    errors = []
    x = torch.as_tensor(tokens_flat, dtype=torch.float32)
    for start in range(0, len(x), batch_size):
        xb = x[start : start + batch_size].to(device)
        x_hat, _ = model(xb)
        mse = (xb - x_hat).pow(2).mean(dim=-1).cpu().numpy()
        errors.append(mse)
    return np.concatenate(errors)


def run_class(
    object_class: str,
    all_samples,
    runner: BackboneRunner,
    pos_mean: np.ndarray,
    args: argparse.Namespace,
) -> dict:
    """Full pipeline for one VisA class. Returns metrics dict."""
    log.info(f"{'='*60}")
    log.info(f"CLASS: {object_class}")
    log.info(f"{'='*60}")

    # ── split samples ──────────────────────────────────────────────────────────
    train_normals = [s for s in all_samples
                     if s.object_class == object_class and s.split == "train"][:args.train]
    test_samples  = [s for s in all_samples
                     if s.object_class == object_class and s.split == "test"]

    log.info(f"  Train normals : {len(train_normals)}")
    log.info(f"  Test samples  : {len(test_samples)}")

    device = runner.device
    weight_path = BSF_WEIGHTS / f"{object_class}.pt"
    log_path = BSF_LOGS / f"{object_class}_history.jsonl"

    if getattr(args, "resume", False) and weight_path.is_file():
        log.info(f"  [Resume] Found existing weights at {weight_path}, loading and skipping training...")
        model = bsf.GrassmannianBSF(
            d=runner.spec.dim, n_groups=args.n_groups, group_size=3, l0=16
        ).to(device)
        model.load_state_dict(torch.load(weight_path, map_location=device, weights_only=True))
        model.eval()
        train_time = 0.0
        final_r2, final_l0, final_dead = 0.0, 0.0, 0
        if log_path.exists():
            try:
                lines = log_path.read_text().strip().splitlines()
                if lines:
                    last = json.loads(lines[-1])
                    final_r2 = float(last.get("r2", 0.0))
                    final_l0 = float(last.get("l0", 0.0))
                    final_dead = int(last.get("dead", 0))
            except Exception as e:
                log.warning(f"Could not read previous history stats: {e}")
    else:
        # ── extract training tokens ────────────────────────────────────────────────
        log.info("  Extracting training features...")
        t0 = time.perf_counter()
        train_features = runner.extract_paths(
            [s.image for s in train_normals], progress=True)
        log.info(f"  Feature extraction: {time.perf_counter()-t0:.1f}s")

        pooled_features = train_features
        X = centre_and_scale_chunked(pooled_features, pos_mean)
        log.info(f"  Training tensor: {X.shape}  ({X.shape[0]:,} patches × {X.shape[1]} dims)")

        # ── train BSF ─────────────────────────────────────────────────────────────
        log.info(f"  Training GrassmannianBSF  n_groups={args.n_groups} epochs={args.epochs}...")
        model = bsf.GrassmannianBSF(
            d=runner.spec.dim, n_groups=args.n_groups, group_size=3, l0=16)

        t0 = time.perf_counter()
        model, history = bsf.train(
            model, X,
            epochs=args.epochs,
            batch_size=args.batch_size,
            device=device,
            log_every=max(1, args.epochs // 10),
        )
        train_time = time.perf_counter() - t0
        log.info(f"  Training complete in {train_time:.1f}s")

        # final stats
        model.eval()
        X_t = torch.as_tensor(X, dtype=torch.float32).to(device)
        final_r2   = recon_r2(model, X_t)
        final_l0, final_dead = l0_dead(model, X_t)
        log.info(f"  Final  R2={final_r2:.4f}  L0={final_l0:.1f}  dead={final_dead}")

        # ── save weights ──────────────────────────────────────────────────────────
        torch.save(model.state_dict(), weight_path)
        log.info(f"  Saved weights → {weight_path}")

        # ── log history ───────────────────────────────────────────────────────────
        with log_path.open("w") as fh:
            for i, ep in enumerate(history["epoch"]):
                fh.write(json.dumps({
                    "epoch": ep, "loss": history["loss"][i],
                    "r2": history["r2"][i], "l0": history["l0"][i],
                    "dead": history["dead"][i],
                }) + "\n")
        log.info(f"  Saved history  → {log_path}")

        # ── training curves ───────────────────────────────────────────────────────
        plot_training_curves(history, object_class, BSF_PLOTS)

    # ── evaluate on test split ────────────────────────────────────────────────
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

        flat = centre_and_scale(feats.tokens, pos_mean)      # (n_tiles*n_pat, d)
        err  = compute_recon_error(model, flat, device)       # (n_tiles*n_pat,)

        # image-level score: mean recon error over all patches
        img_score = float(err.mean())
        img_labels.append(int(sample.is_anomalous))
        img_scores.append(img_score)

        # pixel-level: upsample patch map to full image resolution
        grid = feats.grid
        # err is (n_tiles * grid^2,); reshape to list of 2D per-tile maps for stitch()
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

        # save a few heatmaps
        if len(all_gt_masks) <= 10 and sample.is_anomalous:
            save_heatmap(
                score_map, sample.image,
                object_class, sample.image.stem,
                BSF_HEATMAPS,
            )

    img_labels = np.array(img_labels)
    img_scores = np.array(img_scores)

    metrics: dict = {
        "class":          object_class,
        "train_normals":  len(train_normals),
        "test_samples":   len(test_samples),
        "train_time_s":   round(train_time, 1),
        "final_r2":       round(final_r2,   4),
        "final_l0":       round(final_l0,   2),
        "dead_blocks":    final_dead,
        "img_auroc":      None,
        "img_ap":         None,
        "img_f1_best":    None,
        "pix_auroc":      None,
        "pix_ap":         None,
        "pix_f1_best":    None,
        "aupro":          None,
    }

    # image-level metrics
    if len(set(img_labels)) == 2:
        metrics["img_auroc"]   = round(float(roc_auc_score(img_labels, img_scores)), 4)
        metrics["img_ap"]      = round(float(average_precision_score(img_labels, img_scores)), 4)
        metrics["img_f1_best"] = round(best_f1(img_labels, img_scores), 4)

        normal_sc  = img_scores[img_labels == 0]
        anomaly_sc = img_scores[img_labels == 1]
        plot_score_distributions(normal_sc, anomaly_sc, object_class, BSF_PLOTS)

    # pixel-level metrics
    if all_pix_labels:
        pix_labels = np.concatenate(all_pix_labels)
        pix_scores = np.concatenate(all_pix_scores)

        if len(set(pix_labels)) == 2:
            metrics["pix_auroc"]   = round(float(roc_auc_score(pix_labels, pix_scores)), 4)
            metrics["pix_ap"]      = round(float(average_precision_score(pix_labels, pix_scores)), 4)
            metrics["pix_f1_best"] = round(best_f1(pix_labels, pix_scores), 4)

        # AUPRO (evaluates on list of 2D masks/maps with zero huge array allocations)
        metrics["aupro"] = round(aupro(all_gt_masks, all_pred_maps), 4)

        if len(pix_labels) > 1_000_000:
            idx = np.random.choice(len(pix_labels), 1_000_000, replace=False)
            plot_pix_labels = pix_labels[idx]
            plot_pix_scores = pix_scores[idx]
        else:
            plot_pix_labels = pix_labels
            plot_pix_scores = pix_scores

        plot_roc_pr(
            img_labels, img_scores,
            plot_pix_labels, plot_pix_scores,
            object_class, BSF_PLOTS,
        )

    log.info(f"  img AUROC={metrics['img_auroc']}  img AP={metrics['img_ap']}")
    log.info(f"  pix AUROC={metrics['pix_auroc']}  AUPRO={metrics['aupro']}")

    # save metrics JSON
    metric_path = BSF_LOGS / f"{object_class}_metrics.json"
    with metric_path.open("w") as fh:
        json.dump(metrics, fh, indent=2)
    log.info(f"  Saved metrics  → {metric_path}")

    return metrics


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and evaluate BSF on VisA rigid classes.")
    parser.add_argument("--train",      type=int, default=80,
                        help="max normal frames per class for training (default: 80)")
    parser.add_argument("--epochs",     type=int, default=40,
                        help="training epochs (default: 40)")
    parser.add_argument("--batch-size", type=int, default=2048,
                        help="BSF mini-batch size (default: 2048)")
    parser.add_argument("--n-groups",   type=int, default=512,
                        help="number of concept groups (default: 512)")
    parser.add_argument("--classes",    nargs="+", default=list(RIGID_CLASSES),
                        help="subset of classes to run (default: all rigid)")
    parser.add_argument("--resume",     action="store_true",
                        help="skip training if weights already exist on disk and proceed to evaluation")
    args = parser.parse_args()

    # create output dirs
    for d in (BSF_WEIGHTS, BSF_LOGS, BSF_PLOTS, BSF_HEATMAPS):
        d.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(BSF_LOGS / "full_training.log", mode="a", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-7s  %(message)s", datefmt="%H:%M:%S"))
    logging.getLogger().addHandler(file_handler)

    log.info("=" * 60)
    log.info("Starting BSF run (classes: %s, resume=%s)", args.classes, args.resume)
    log.info("=" * 60)

    log.info("Loading VisA index...")
    all_samples = ingest(rigid_only=True)
    log.info(f"  {len(all_samples)} samples indexed")

    log.info("Loading backbone runner...")
    runner = BackboneRunner()
    log.info(f"  {runner.describe()}")

    log.info(f"Loading pos_mean from {POS_MEAN_PATH}...")
    pos_mean = np.load(POS_MEAN_PATH).astype(np.float32)

    # ── per-class training + evaluation ───────────────────────────────────────
    all_metrics = []
    for object_class in args.classes:
        m = run_class(object_class, all_samples, runner, pos_mean, args)
        all_metrics.append(m)

    # ── summary table ─────────────────────────────────────────────────────────
    existing_metrics = {}
    for mf in BSF_LOGS.glob("*_metrics.json"):
        try:
            with mf.open("r") as f:
                c_data = json.load(f)
                if isinstance(c_data, dict) and "class" in c_data:
                    existing_metrics[c_data["class"]] = c_data
        except Exception:
            pass

    for m in all_metrics:
        existing_metrics[m["class"]] = m

    merged_metrics = [existing_metrics[c] for c in RIGID_CLASSES if c in existing_metrics]
    for c, data in existing_metrics.items():
        if c not in RIGID_CLASSES:
            merged_metrics.append(data)

    summary_path = BSF_LOGS / "summary.json"
    with summary_path.open("w") as fh:
        json.dump(merged_metrics, fh, indent=2)

    print()
    print("=" * 90)
    print(f"{'Class':<14}  {'imgAUROC':>9}  {'imgAP':>7}  {'imgF1':>7}  "
          f"{'pixAUROC':>9}  {'pixAP':>7}  {'AUPRO':>7}  {'R2':>6}  {'L0':>5}")
    print("-" * 90)
    for m in merged_metrics:
        print(
            f"{m['class']:<14}  "
            f"{str(m['img_auroc']):>9}  "
            f"{str(m['img_ap']):>7}  "
            f"{str(m['img_f1_best']):>7}  "
            f"{str(m['pix_auroc']):>9}  "
            f"{str(m['pix_ap']):>7}  "
            f"{str(m['aupro']):>7}  "
            f"{m['final_r2']:>6.4f}  "
            f"{m['final_l0']:>5.1f}"
        )
    print("=" * 90)
    print(f"\nAll outputs saved to:\n  weights -> {BSF_WEIGHTS}\n  logs    -> {BSF_LOGS}")


if __name__ == "__main__":
    main()
