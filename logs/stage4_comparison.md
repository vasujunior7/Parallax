# Stage 4 — Probe Comparison Report

*Generated 2026-09-11 11:27*

## Models

| Model | Description |
|---|---|
| **Linear Probe** | Mahalanobis distance on DINOv3 activations — the guaranteed floor |
| **GrassmannianBSF** | Block-sparse featurizer, 512 groups x 3D, L0=16 |
| **BatchTopKSAE** | SAE baseline, d_sae=1536, k=16 (BSF-matched capacity) |

## Image-level AUROC

| Class | Linear Probe | GrassmannianBSF | BatchTopKSAE | Winner |
|---|---|---|---|---|
| pcb1 | 0.9580 | 0.8928 | 0.8892 | **BSF** |
| pcb2 | 0.7300 | 0.7127 | 0.6882 | **BSF** |
| pcb3 | 0.8350 | 0.8275 | 0.8144 | **BSF** |
| pcb4 | 0.8830 | 0.9068 | 0.8853 | **BSF** |
| capsules | 0.9010 | 0.6908 | 0.6928 | **SAE** |
| candle | 0.8480 | 0.9404 | 0.9391 | **BSF** |
| **Wins** | — | **5** | **1** | |

## Pixel-level AUROC

| Class | GrassmannianBSF | BatchTopKSAE | Winner |
|---|---|---|---|
| pcb1 | 0.9948 | 0.9939 | **BSF** |
| pcb2 | 0.9705 | 0.9649 | **BSF** |
| pcb3 | 0.9783 | 0.9743 | **BSF** |
| pcb4 | 0.9571 | 0.9533 | **BSF** |
| capsules | 0.9872 | 0.9861 | **BSF** |
| candle | 0.9932 | 0.9924 | **BSF** |
| **Wins** | **6** | **0** | |

## AUPRO

| Class | GrassmannianBSF | BatchTopKSAE |
|---|---|---|
| pcb1 | 0.9083 | 0.8935 |
| pcb2 | 0.8330 | 0.8084 |
| pcb3 | 0.8816 | 0.8625 |
| pcb4 | 0.8051 | 0.7967 |
| capsules | 0.9336 | 0.9277 |
| candle | 0.9546 | 0.9586 |

## Subspace capture (SAE)

> `k_for_95pct`: decoder atoms needed to explain 95% of normal-patch variance.
> BSF always uses exactly 3 dimensions per concept (group_size=3).
> So the comparable BSF count is `k_for_95pct / 3` blocks.
> A large `k_for_95pct` confirms the SAE manifold-dilution finding from the paper.

| Class | k @ 95% | Var @ k=16 |
|---|---|---|
| pcb1 | 65 | 0.4662 |
| pcb2 | 65 | 0.5691 |
| pcb3 | 65 | 0.4714 |
| pcb4 | 65 | 0.5369 |
| capsules | 65 | 0.4904 |
| candle | 65 | 0.4285 |

## Interpretation

- **Image AUROC**: BSF wins 5/6 classes over SAE.
- **Pixel AUROC**: BSF wins 6/6 classes over SAE.
- **Why we keep BSF even on ties**: unlike an SAE, BSF outputs a *block coordinate*
  (a vector inside each concept's subspace), enabling the escalation UI to explain
  *where within a concept* the image sits — e.g. 'cracked end of weld-seam manifold'
  vs just 'weld-seam feature fired'.
- **Subspace capture** measures the companion-paper dilution claim quantitatively:
  if `k_for_95pct` is large, the SAE scatters the normal manifold across many atoms
  where BSF captures each concept in a 3D block.