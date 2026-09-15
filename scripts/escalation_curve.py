"""Escalation trade-off curve — headline result.

Reads scores from the SQLite trace DB produced by run_agent.py.
Sweeps the OOD escalation threshold across percentiles of the stored score
distribution and records:
  - autonomy_rate  = fraction of frames auto-accepted (not escalated)
  - accuracy       = precision within the accepted set (no missed defects)
  - sensitivity    = recall of anomalies via escalation
  - false_esc_rate = fraction of normals wrongly escalated

Outputs
-------
  logs/agent/escalation_curve.json   machine-readable sweep table
  logs/agent/escalation_curve.md     human-readable table for the report
  logs/agent/plots/escalation_trade_off.png

Usage::

    uv run python scripts/escalation_curve.py
    uv run python scripts/escalation_curve.py --db logs/agent/run_20260911_230345.db
    uv run python scripts/escalation_curve.py --n-points 100
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
AGENT_LOGS   = PROJECT_ROOT / "logs" / "agent"
AGENT_PLOTS  = AGENT_LOGS / "plots"


def latest_db(log_dir: Path) -> Path:
    dbs = sorted(log_dir.glob("run_*.db"))
    if not dbs:
        raise FileNotFoundError(f"No run_*.db found in {log_dir}. Run run_agent.py first.")
    return dbs[-1]


def load_rows(db_path: Path) -> list[dict]:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM decisions ORDER BY timestamp_utc").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def sweep(
    rows: list[dict],
    n_points: int = 200,
) -> list[dict]:
    """Sweep OOD escalate threshold; record trade-off metrics at each point."""
    ood       = np.array([r["ood_score"]           for r in rows])
    is_defect = np.array([r["vision_is_defective"]  for r in rows], dtype=bool)

    # The VisA is_anomalous ground truth is in the frame_id path
    # (".../Data/Images/Anomaly/..." vs ".../Data/Images/Normal/...").
    is_anomalous = np.array(
        ["anomaly" in (r.get("frame_id") or "").lower() for r in rows], dtype=bool
    )
    # Fall back to vision label if no path info (e.g. synthetic frames without path)
    has_path_info = any(
        "anomaly" in (r.get("frame_id") or "").lower() or "normal" in (r.get("frame_id") or "").lower()
        for r in rows
    )
    if not has_path_info:
        is_anomalous = is_defect

    thresholds = np.percentile(ood, np.linspace(0, 100, n_points + 1))
    thresholds = np.unique(thresholds)

    results = []
    n_total   = len(rows)
    n_normal  = int((~is_anomalous).sum())
    n_anomaly = int(is_anomalous.sum())

    for t in thresholds:
        escalated = ood >= t
        accepted  = ~escalated

        tp = int((escalated & is_anomalous).sum())
        fn = int((accepted  & is_anomalous).sum())
        fp = int((escalated & ~is_anomalous).sum())
        tn = int((accepted  & ~is_anomalous).sum())

        autonomy_normals   = tn / max(n_normal, 1)     # fraction of normals auto-handled
        autonomy_overall   = (tn + fn) / max(n_total, 1) # fraction of all frames accepted
        sensitivity        = tp / max(tp + fn, 1)     # anomaly recall via escalation
        false_esc          = fp / max(n_normal, 1)     # fraction of normals wrongly escalated
        precision_accepted = tn / max(tn + fn, 1) if (tn + fn) > 0 else 1.0  # clean rate in accepted set

        results.append({
            "threshold":          float(round(t, 2)),
            "autonomy_rate":      round(autonomy_normals,     4),
            "overall_autonomy":   round(autonomy_overall,     4),
            "sensitivity":        round(sensitivity,          4),
            "false_esc_rate":     round(false_esc,            4),
            "precision_accepted": round(precision_accepted,   4),
            "accept": tn + fn, "escalate": tp + fp, "TP": tp, "FP": fp, "TN": tn, "FN": fn,
            "n_total": n_total, "n_normal": n_normal, "n_anomaly": n_anomaly,
        })

    return results


def find_operating_points(results: list[dict], targets: tuple[float, ...] = (0.80, 0.90, 0.95, 0.99)) -> list[dict]:
    """Find operating points closest to target sensitivities."""
    sens = np.array([r["sensitivity"] for r in results])
    points = []
    for target in targets:
        idx = int(np.argmin(np.abs(sens - target)))
        r = results[idx].copy()
        r["target_sensitivity"] = target
        points.append(r)
    return points


def write_md(
    results: list[dict],
    out_dir: Path,
    db_path: Path,
    per_class_results: dict[str, list[dict]] | None = None,
    current_operating_stats: dict | None = None,
) -> None:
    n_total   = results[0]["n_total"] if results else 0
    n_normal  = results[0]["n_normal"] if results else 0
    n_anomaly = results[0]["n_anomaly"] if results else 0

    lines = [
        "# Escalation Trade-off Curve — Evaluation Report",
        "",
        f"*Source DB: `{db_path.name}`*",
        f"*Total Decisions Evaluated: {n_total} (Normal: {n_normal}, Anomaly: {n_anomaly})*",
        "",
        "The **autonomy rate** is the fraction of normal frames the system handles",
        "autonomously (no human review needed). **Sensitivity** is the fraction of anomaly",
        "frames that reach a human. The trade-off is swept by varying the OOD escalation threshold.",
        "",
        "## Key Operating Points (Aggregate)",
        "",
        "| Target Sensitivity | Threshold | Autonomy (Normals) | False Escalation | Accepted Accuracy | Auto-Accepted (All) |",
        "|---|---|---|---|---|---|",
    ]

    op_points = find_operating_points(results, targets=(0.80, 0.90, 0.95, 0.99))
    for op in op_points:
        lines.append(
            f"| {op['target_sensitivity']:>17.0%} | {op['threshold']:>9.1f} | "
            f"{op['autonomy_rate']:>18.1%} | {op['false_esc_rate']:>16.1%} | "
            f"{op['precision_accepted']:>17.1%} | {op['overall_autonomy']:>19.1%} |"
        )

    if current_operating_stats:
        lines.extend([
            "",
            "## Current Run Operating Point (as configured)",
            "",
            f"Configured thresholds produced the following terminal decisions:",
            f"- **ACCEPT**: {current_operating_stats.get('accept', 0)}",
            f"- **RELOOK**: {current_operating_stats.get('relook', 0)}",
            f"- **ESCALATE**: {current_operating_stats.get('escalate', 0)}",
            f"- **Sensitivity (defect catch rate)**: {current_operating_stats.get('sensitivity', 0):.1%}",
            f"- **Specificity (normal pass rate)**: {current_operating_stats.get('specificity', 0):.1%}",
            f"- **False Escalation Rate**: {current_operating_stats.get('false_esc_rate', 0):.1%}",
        ])

    if per_class_results:
        lines.extend([
            "",
            "## Per-Class Operating Points (@ 95% Sensitivity)",
            "",
            "| Class | Decisions | Normals | Anomalies | Threshold @ 95% Sens | Autonomy (Normals) | False Escalation |",
            "|---|---|---|---|---|---|---|",
        ])
        for cls_name, cls_res in per_class_results.items():
            if not cls_res:
                continue
            cls_ops = find_operating_points(cls_res, targets=(0.95,))
            if cls_ops:
                cop = cls_ops[0]
                lines.append(
                    f"| {cls_name:<10} | {cop['n_total']:>9} | {cop['n_normal']:>7} | {cop['n_anomaly']:>9} | "
                    f"{cop['threshold']:>20.1f} | {cop['autonomy_rate']:>18.1%} | {cop['false_esc_rate']:>16.1%} |"
                )

    lines.extend([
        "",
        "## Full Sweep Table (Sampled)",
        "",
        "| Threshold | Autonomy (Normals) | Sensitivity | FalseEscRate | PrecisionAccepted | Auto-Accepted (All) |",
        "|---|---|---|---|---|---|",
    ])
    # Show every Nth row to keep table readable
    stride = max(1, len(results) // 30)
    for r in results[::stride]:
        lines.append(
            f"| {r['threshold']:>9.1f} | {r['autonomy_rate']:>18.3f} | "
            f"{r['sensitivity']:>11.3f} | {r['false_esc_rate']:>12.3f} | "
            f"{r['precision_accepted']:>17.3f} | {r['overall_autonomy']:>19.3f} |"
        )

    out_path = out_dir / "escalation_curve.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")


def plot(results: list[dict], out_dir: Path, db_path: Path, prefix: str = "") -> Path:
    autonomy = [r["autonomy_rate"]  for r in results]
    sens     = [r["sensitivity"]    for r in results]
    fer      = [r["false_esc_rate"] for r in results]

    title_suffix = f" — {prefix}" if prefix else ""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"Parallax — Escalation Trade-off Curve{title_suffix}", fontsize=14, fontweight="bold")

    # Left: Sensitivity vs False-Escalation (ROC-style)
    ax = axes[0]
    ax.plot(fer, sens, "b-o", ms=3, lw=1.5, label="OOD threshold sweep")
    ax.set_xlabel("False Escalation Rate (normals wrongly routed to human)")
    ax.set_ylabel("Sensitivity (anomaly frames caught)")
    ax.set_title("Sensitivity vs False Escalation Rate")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.legend()

    # Right: Autonomy vs Sensitivity
    ax = axes[1]
    ax.plot(autonomy, sens, "g-o", ms=3, lw=1.5)
    ax.set_xlabel("Autonomy Rate (fraction of normals auto-accepted)")
    ax.set_ylabel("Sensitivity")
    ax.set_title("Autonomy vs Sensitivity  ←  headline trade-off")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)

    # annotate key operating points
    for target_sens in (0.80, 0.90, 0.95, 0.99):
        idx = int(np.argmin(np.abs(np.array(sens) - target_sens)))
        ax.annotate(
            f"Sens={sens[idx]:.0%}\nAuto={autonomy[idx]:.0%}",
            (autonomy[idx], sens[idx]),
            textcoords="offset points", xytext=(8, -14),
            fontsize=8, arrowprops=dict(arrowstyle="->", lw=0.8, color="black"),
            bbox=dict(boxstyle="round,pad=0.2", facecolor="yellow", alpha=0.3),
        )

    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{prefix}_escalation_trade_off.png" if prefix else "escalation_trade_off.png"
    p = out_dir / filename
    plt.savefig(p, dpi=150)
    plt.close(fig)
    return p


def compute_current_stats(rows: list[dict]) -> dict:
    """Compute performance stats for the decisions as actually logged."""
    is_anomalous = np.array(["anomaly" in (r.get("frame_id") or "").lower() for r in rows], dtype=bool)
    escalated = np.array([r["decision"] == "ESCALATE" for r in rows], dtype=bool)
    accepted  = np.array([r["decision"] == "ACCEPT" for r in rows], dtype=bool)
    relook    = np.array([r["decision"] == "RELOOK" for r in rows], dtype=bool)

    tp = int((escalated & is_anomalous).sum())
    fn = int((accepted & is_anomalous).sum())
    fp = int((escalated & ~is_anomalous).sum())
    tn = int((accepted & ~is_anomalous).sum())

    n_normal = max(int((~is_anomalous).sum()), 1)
    n_anomaly = max(int(is_anomalous.sum()), 1)

    return {
        "accept": int(accepted.sum()),
        "relook": int(relook.sum()),
        "escalate": int(escalated.sum()),
        "TP": tp, "FP": fp, "TN": tn, "FN": fn,
        "sensitivity": tp / n_anomaly,
        "specificity": tn / n_normal,
        "false_esc_rate": fp / n_normal,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the escalation trade-off curve.")
    parser.add_argument("--db",       type=Path, default=None,
                        help="Path to SQLite trace DB (default: latest run_*.db).")
    parser.add_argument("--classes",  nargs="+", default=None,
                        help="Filter to specific classes (e.g. --classes pcb1 pcb2).")
    parser.add_argument("--n-points", type=int, default=200,
                        help="Number of threshold sweep points (default: 200).")
    args = parser.parse_args()

    db_path = args.db or latest_db(AGENT_LOGS)
    print(f"Reading trace DB: {db_path}")

    rows = load_rows(db_path)
    print(f"  {len(rows)} decisions loaded")

    if args.classes:
        rows = [r for r in rows if r["object_class"] in args.classes]
        print(f"  {len(rows)} decisions after filtering for {args.classes}")

    if not rows:
        print("No decisions to evaluate.")
        return

    # Aggregate sweep
    results = sweep(rows, n_points=args.n_points)

    # Per-class sweeps
    classes_in_db = sorted(list(set(r["object_class"] for r in rows)))
    per_class_results = {}
    for cls in classes_in_db:
        cls_rows = [r for r in rows if r["object_class"] == cls]
        cls_has_norm = any("normal" in (r.get("frame_id") or "").lower() for r in cls_rows)
        cls_has_anml = any("anomaly" in (r.get("frame_id") or "").lower() for r in cls_rows)
        if len(cls_rows) > 5 and cls_has_norm and cls_has_anml:
            cls_sweep = sweep(cls_rows, n_points=min(args.n_points, len(cls_rows)))
            per_class_results[cls] = cls_sweep
            p = plot(cls_sweep, AGENT_PLOTS, db_path, prefix=cls)
            print(f"Saved plot -> {p}")
        elif len(cls_rows) > 0:
            print(f"  Skipping trade-off plot for {cls}: incomplete test split (normals={cls_has_norm}, anomalies={cls_has_anml}, n={len(cls_rows)})")

    current_stats = compute_current_stats(rows)

    # Save outputs
    AGENT_PLOTS.mkdir(parents=True, exist_ok=True)
    json_path = AGENT_LOGS / "escalation_curve.json"
    with json_path.open("w") as f:
        json.dump({
            "aggregate": results,
            "operating_points": find_operating_points(results),
            "current_run_stats": current_stats,
            "per_class": per_class_results,
        }, f, indent=2)
    print(f"Saved JSON  -> {json_path}")

    write_md(results, AGENT_LOGS, db_path, per_class_results=per_class_results, current_operating_stats=current_stats)
    print(f"Saved MD    -> {AGENT_LOGS / 'escalation_curve.md'}")

    p_agg = plot(results, AGENT_PLOTS, db_path)
    print(f"Saved plot -> {p_agg}")
    print("Done.")


if __name__ == "__main__":
    main()
