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

    # We treat vision_is_defective as ground-truth anomaly proxy.
    # The VisA is_anomalous ground truth is in the frame_id path
    # (".../test/anomaly/..." vs ".../test/normal/...").
    is_anomalous = np.array(
        ["anomaly" in (r.get("frame_id") or "") for r in rows], dtype=bool
    )
    # Fall back to vision label if no path info (e.g. synthetic frames)
    if is_anomalous.sum() == 0:
        is_anomalous = is_defect

    thresholds = np.percentile(ood, np.linspace(0, 100, n_points + 1))
    thresholds = np.unique(thresholds)

    results = []
    for t in thresholds:
        escalated = ood >= t
        accepted  = ~escalated

        tp = int((escalated & is_anomalous).sum())
        fn = int((accepted  & is_anomalous).sum())
        fp = int((escalated & ~is_anomalous).sum())
        tn = int((accepted  & ~is_anomalous).sum())

        n_total     = len(rows)
        autonomy    = tn / max(n_total, 1)      # fraction of normals auto-handled
        sensitivity = tp / max(tp + fn, 1)      # anomaly recall via escalation
        false_esc   = fp / max(fp + tn, 1)      # false alarm rate on normals
        precision_accepted = tn / max(tn + fn, 1)  # how clean the accepted set is

        results.append({
            "threshold":          float(round(t, 2)),
            "autonomy_rate":      round(autonomy,             4),
            "sensitivity":        round(sensitivity,          4),
            "false_esc_rate":     round(false_esc,            4),
            "precision_accepted": round(precision_accepted,   4),
            "accept": tn, "escalate": tp + fp, "TP": tp, "FP": fp, "TN": tn, "FN": fn,
        })

    return results


def write_md(results: list[dict], out_dir: Path, db_path: Path) -> None:
    lines = [
        "# Escalation Trade-off Curve",
        "",
        f"*Source DB: `{db_path.name}`*",
        "",
        "The **autonomy rate** is the fraction of normal frames the system handles",
        "autonomously (no human review needed). **Sensitivity** is the fraction of anomaly",
        "frames that reach a human. The trade-off is swept by varying the OOD escalation threshold.",
        "",
        "| Threshold | Autonomy | Sensitivity | FalseEscRate | PrecisionAccepted |",
        "|---|---|---|---|---|",
    ]
    # Show every 10th row to keep the table readable
    stride = max(1, len(results) // 30)
    for r in results[::stride]:
        lines.append(
            f"| {r['threshold']:>10.1f} | {r['autonomy_rate']:>8.3f} | "
            f"{r['sensitivity']:>11.3f} | {r['false_esc_rate']:>12.3f} | "
            f"{r['precision_accepted']:>17.3f} |"
        )
    (out_dir / "escalation_curve.md").write_text("\n".join(lines), encoding="utf-8")


def plot(results: list[dict], out_dir: Path, db_path: Path) -> None:
    autonomy = [r["autonomy_rate"]  for r in results]
    sens     = [r["sensitivity"]    for r in results]
    fer      = [r["false_esc_rate"] for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Parallax — Escalation Trade-off Curve", fontsize=14, fontweight="bold")

    # Left: Sensitivity vs False-Escalation (ROC-style)
    ax = axes[0]
    ax.plot(fer, sens, "b-o", ms=3, lw=1.5, label="OOD threshold sweep")
    ax.set_xlabel("False Escalation Rate (normals wrongly routed to human)")
    ax.set_ylabel("Sensitivity (anomaly frames caught)")
    ax.set_title("Sensitivity vs False Escalation Rate")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Right: Autonomy vs Sensitivity
    ax = axes[1]
    ax.plot(autonomy, sens, "g-o", ms=3, lw=1.5)
    ax.set_xlabel("Autonomy Rate (fraction of normals auto-accepted)")
    ax.set_ylabel("Sensitivity")
    ax.set_title("Autonomy vs Sensitivity  ←  headline trade-off")
    ax.grid(True, alpha=0.3)
    # annotate a few operating points
    for target_sens in (0.80, 0.90, 0.95, 0.99):
        idx = np.argmin(np.abs(np.array(sens) - target_sens))
        ax.annotate(
            f"Sens={target_sens:.0%}\nAuto={autonomy[idx]:.0%}",
            (autonomy[idx], sens[idx]),
            textcoords="offset points", xytext=(8, -14),
            fontsize=7.5, arrowprops=dict(arrowstyle="-", lw=0.5),
        )

    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "escalation_trade_off.png"
    plt.savefig(p, dpi=150)
    plt.close(fig)
    print(f"Saved plot -> {p}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the escalation trade-off curve.")
    parser.add_argument("--db",       type=Path, default=None,
                        help="Path to SQLite trace DB (default: latest run_*.db).")
    parser.add_argument("--n-points", type=int, default=200,
                        help="Number of threshold sweep points (default: 200).")
    args = parser.parse_args()

    db_path = args.db or latest_db(AGENT_LOGS)
    print(f"Reading trace DB: {db_path}")

    rows = load_rows(db_path)
    print(f"  {len(rows)} decisions loaded")

    results = sweep(rows, n_points=args.n_points)

    # JSON
    AGENT_PLOTS.mkdir(parents=True, exist_ok=True)
    json_path = AGENT_LOGS / "escalation_curve.json"
    with json_path.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved JSON  -> {json_path}")

    write_md(results, AGENT_LOGS, db_path)
    print(f"Saved MD    -> {AGENT_LOGS / 'escalation_curve.md'}")

    plot(results, AGENT_PLOTS, db_path)
    print("Done.")


if __name__ == "__main__":
    main()
