"""Calibrate OOD and BSF thresholds from measured score distributions.

Reads the per-class *_metrics.json files written by run_agent.py (which always
contain ood_normal_mean, ood_anomaly_mean, bsf_normal_mean, bsf_anomaly_mean
regardless of the run's decision counts), then derives σ-based thresholds and
re-launches run_agent.py with the calibrated values.

Usage::

    # Dry-run — print calibrated config only
    uv run python scripts/calibrate_thresholds.py

    # Actually re-run the agent with calibrated thresholds
    uv run python scripts/calibrate_thresholds.py --run

    # Tune aggressiveness (σ multiplier for relook / escalate)
    uv run python scripts/calibrate_thresholds.py --run --relook-sigma 1.0 --escalate-sigma 2.0
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
AGENT_LOGS   = PROJECT_ROOT / "logs" / "agent"


def load_class_stats() -> list[dict]:
    """Load all per-class metrics JSONs."""
    stats = []
    for path in sorted(AGENT_LOGS.glob("*_metrics.json")):
        if path.stem in ("summary",):
            continue
        with path.open() as f:
            stats.append(json.load(f))
    return stats


def calibrate(
    stats: list[dict],
    relook_sigma: float = 1.5,
    escalate_sigma: float = 2.5,
) -> tuple[float, float, float, float]:
    """Return (ood_relook, ood_escalate, bsf_relook, bsf_escalate).

    Strategy:
      - OOD relook  threshold = mean_normal_ood  + relook_sigma  * std_spread
      - OOD escalate threshold = mean_normal_ood  + escalate_sigma * std_spread
      where std_spread ≈ (mean_anomaly - mean_normal) / 2 (half-separation as
      a robust σ proxy when we don't store per-class std).
      - BSF follows the same pattern on bsf_normal_mean / bsf_anomaly_mean.
    """
    ood_normals, ood_anomalies   = [], []
    bsf_normals, bsf_anomalies   = [], []

    for s in stats:
        if s.get("ood_normal_mean", 0) > 0:
            ood_normals.append(s["ood_normal_mean"])
            ood_anomalies.append(s["ood_anomaly_mean"])
        if s.get("bsf_normal_mean", 0) > 0:
            bsf_normals.append(s["bsf_normal_mean"])
            bsf_anomalies.append(s["bsf_anomaly_mean"])

    def _thresh(normals: list[float], anomalies: list[float],
                relook_s: float, escalate_s: float,
                digits: int = 2, min_spread: float = 1e-4) -> tuple[float, float]:
        mean_n  = sum(normals)  / len(normals)
        mean_a  = sum(anomalies) / len(anomalies)
        spread  = max((mean_a - mean_n) / 2.0, min_spread)
        relook  = round(mean_n + relook_s   * spread, digits)
        escalate = round(mean_n + escalate_s * spread, digits)
        # Safety: guarantee relook < escalate
        if escalate <= relook:
            escalate = round(relook + (10 ** -digits * 5), digits)
        return relook, escalate

    ood_relook, ood_escalate = _thresh(ood_normals, ood_anomalies,
                                        relook_sigma, escalate_sigma, digits=2, min_spread=10.0)
    bsf_relook, bsf_escalate = _thresh(bsf_normals, bsf_anomalies,
                                        relook_sigma, escalate_sigma, digits=4, min_spread=0.02)

    return ood_relook, ood_escalate, bsf_relook, bsf_escalate


def print_table(stats: list[dict], ood_r: float, ood_e: float,
                bsf_r: float, bsf_e: float) -> None:
    header = f"{'Class':<12}  {'OOD normal':>12}  {'OOD anomaly':>12}  {'BSF normal':>10}  {'BSF anomaly':>10}"
    print(header)
    print("-" * len(header))
    for s in stats:
        print(f"{s['class']:<12}  {s['ood_normal_mean']:>12.1f}  "
              f"{s['ood_anomaly_mean']:>12.1f}  "
              f"{s['bsf_normal_mean']:>10.4f}  "
              f"{s['bsf_anomaly_mean']:>10.4f}")
    print()
    print("Calibrated thresholds:")
    print(f"  ood_relook_threshold   = {ood_r}")
    print(f"  ood_escalate_threshold = {ood_e}")
    print(f"  bsf_relook_threshold   = {bsf_r}")
    print(f"  bsf_escalate_threshold = {bsf_e}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate agent thresholds.")
    parser.add_argument("--run", action="store_true",
                        help="Re-launch run_agent.py with calibrated thresholds.")
    parser.add_argument("--relook-sigma",   type=float, default=1.5,
                        help="σ multiplier for relook threshold (default: 1.5).")
    parser.add_argument("--escalate-sigma", type=float, default=2.5,
                        help="σ multiplier for escalate threshold (default: 2.5).")
    parser.add_argument("--classes", nargs="+", default=None,
                        help="Subset of classes to pass to run_agent.py.")
    parser.add_argument("--max-test", type=int, default=200)
    args = parser.parse_args()

    stats = load_class_stats()
    if not stats:
        print(f"ERROR: no *_metrics.json files found in {AGENT_LOGS}", file=sys.stderr)
        print("Run scripts/run_agent.py first to generate score distributions.", file=sys.stderr)
        sys.exit(1)

    ood_r, ood_e, bsf_r, bsf_e = calibrate(
        stats,
        relook_sigma=args.relook_sigma,
        escalate_sigma=args.escalate_sigma,
    )

    print_table(stats, ood_r, ood_e, bsf_r, bsf_e)

    if not args.run:
        print("Pass --run to re-launch run_agent.py with these thresholds.")
        return

    run_agent = PROJECT_ROOT / "scripts" / "run_agent.py"
    cmd = [
        sys.executable, str(run_agent),
        "--ood-relook",   str(ood_r),
        "--ood-escalate", str(ood_e),
        "--bsf-relook",   str(bsf_r),
        "--bsf-escalate", str(bsf_e),
        "--max-test",     str(args.max_test),
    ]
    if args.classes:
        cmd += ["--classes"] + args.classes

    print("Launching:", " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
