# Escalation Trade-off Curve — Evaluation Report

*Source DB: `run_20260911_230345.db`*
*Total Decisions Evaluated: 623 (Normal: 324, Anomaly: 299)*

The **autonomy rate** is the fraction of normal frames the system handles
autonomously (no human review needed). **Sensitivity** is the fraction of anomaly
frames that reach a human. The trade-off is swept by varying the OOD escalation threshold.

## Key Operating Points (Aggregate)

| Target Sensitivity | Threshold | Autonomy (Normals) | False Escalation | Accepted Accuracy | Auto-Accepted (All) |
|---|---|---|---|---|---|
|               80% |    2114.9 |              39.2% |            60.8% |             67.9% |               30.0% |
|               90% |    2004.8 |              25.3% |            74.7% |             73.2% |               18.0% |
|               95% |    1935.1 |              19.4% |            80.6% |             80.8% |               12.5% |
|               99% |    1817.2 |               8.9% |            91.0% |             90.6% |                5.1% |

## Current Run Operating Point (as configured)

Configured thresholds produced the following terminal decisions:
- **ACCEPT**: 0
- **RELOOK**: 0
- **ESCALATE**: 623
- **Sensitivity (defect catch rate)**: 100.0%
- **Specificity (normal pass rate)**: 0.0%
- **False Escalation Rate**: 100.0%

## Per-Class Operating Points (@ 95% Sensitivity)

| Class | Decisions | Normals | Anomalies | Threshold @ 95% Sens | Autonomy (Normals) | False Escalation |
|---|---|---|---|---|---|---|
| pcb1       |       200 |     100 |       100 |               1985.7 |              21.0% |            79.0% |
| pcb2       |       200 |     100 |       100 |               1842.7 |              18.0% |            82.0% |
| pcb3       |       200 |     101 |        99 |               2036.0 |              18.8% |            81.2% |

## Full Sweep Table (Sampled)

| Threshold | Autonomy (Normals) | Sensitivity | FalseEscRate | PrecisionAccepted | Auto-Accepted (All) |
|---|---|---|---|---|---|
|    1553.7 |              0.000 |       1.000 |        1.000 |             1.000 |               0.000 |
|    1756.6 |              0.056 |       0.997 |        0.944 |             0.947 |               0.030 |
|    1825.0 |              0.102 |       0.983 |        0.898 |             0.868 |               0.061 |
|    1873.4 |              0.148 |       0.973 |        0.852 |             0.857 |               0.090 |
|    1918.5 |              0.188 |       0.953 |        0.812 |             0.813 |               0.120 |
|    1970.7 |              0.219 |       0.923 |        0.781 |             0.755 |               0.151 |
|    2004.8 |              0.253 |       0.900 |        0.747 |             0.732 |               0.180 |
|    2031.3 |              0.293 |       0.880 |        0.707 |             0.725 |               0.210 |
|    2062.9 |              0.324 |       0.850 |        0.676 |             0.700 |               0.241 |
|    2084.2 |              0.355 |       0.823 |        0.645 |             0.684 |               0.270 |
|    2114.9 |              0.392 |       0.799 |        0.608 |             0.679 |               0.300 |
|    2138.6 |              0.423 |       0.769 |        0.577 |             0.665 |               0.331 |
|    2170.3 |              0.451 |       0.739 |        0.549 |             0.652 |               0.360 |
|    2200.3 |              0.494 |       0.722 |        0.506 |             0.658 |               0.390 |
|    2226.6 |              0.537 |       0.706 |        0.463 |             0.664 |               0.420 |
|    2268.0 |              0.565 |       0.676 |        0.435 |             0.654 |               0.449 |
|    2282.7 |              0.599 |       0.649 |        0.401 |             0.649 |               0.480 |
|    2309.2 |              0.626 |       0.615 |        0.373 |             0.638 |               0.510 |
|    2335.5 |              0.660 |       0.592 |        0.340 |             0.637 |               0.539 |
|    2385.4 |              0.685 |       0.555 |        0.315 |             0.625 |               0.570 |
|    2406.3 |              0.719 |       0.528 |        0.281 |             0.623 |               0.600 |
|    2456.6 |              0.753 |       0.505 |        0.247 |             0.622 |               0.629 |
|    2502.9 |              0.778 |       0.468 |        0.222 |             0.613 |               0.660 |
|    2571.4 |              0.806 |       0.435 |        0.194 |             0.607 |               0.690 |
|    2620.0 |              0.821 |       0.391 |        0.179 |             0.594 |               0.719 |
|    2676.5 |              0.849 |       0.358 |        0.151 |             0.589 |               0.750 |
|    2737.3 |              0.874 |       0.321 |        0.127 |             0.582 |               0.780 |
|    2805.7 |              0.901 |       0.291 |        0.099 |             0.579 |               0.809 |
|    2877.4 |              0.929 |       0.258 |        0.071 |             0.576 |               0.840 |
|    2976.9 |              0.948 |       0.214 |        0.052 |             0.566 |               0.870 |
|    3143.2 |              0.963 |       0.171 |        0.037 |             0.557 |               0.899 |
|    3317.9 |              0.969 |       0.114 |        0.031 |             0.542 |               0.929 |
|    3639.7 |              0.982 |       0.064 |        0.018 |             0.532 |               0.960 |
|    4246.7 |              0.994 |       0.017 |        0.006 |             0.523 |               0.989 |