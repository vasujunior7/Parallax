# Reference Papers

Local copies of the two papers the confidence sidecar is built on. Both are redistributed here under
**CC BY 4.0**, the licence their authors selected on arXiv. Attribution is given below as that licence requires.
Neither file has been modified — each is byte-identical to the copy served by arXiv.

| File | Paper |
|---|---|
| `2606.25234-block-sparse-featurizers.pdf` | Fel, T., Kowal, M., Jacobs, M., Hazra, D., Bhalla, U., Sharkey, L., Bushnaq, L., Grant, S., Haklay, T., Icard, T., Rager, C., Pearce, M., Wurgaft, D., Swann, A., Doshi, F., Boppana, S., Tigges, C., Cammarata, N., Serre, T., Shyam, V., Lewis, O., McGrath, T., Merullo, J., Lubana, E. S., Geiger, A. **Structuring Sparsity: Block-Sparse Featurizers Capture Visual Concept Manifolds.** arXiv:2606.25234, 23 June 2026. <https://arxiv.org/abs/2606.25234> |
| `2604.28119-do-saes-capture-concept-manifolds.pdf` | Bhalla, U., Fel, T., Rager, C., Feucht, S., Haklay, T., Wurgaft, D., Boppana, S., Kowal, M., Shyam, V., Lewis, O., McGrath, T., Merullo, J., Geiger, A., Lubana, E. S. **Do Sparse Autoencoders Capture Concept Manifolds?** arXiv:2604.28119, 30 April 2026. <https://arxiv.org/abs/2604.28119> |

Licence for both: Creative Commons Attribution 4.0 International —
<https://creativecommons.org/licenses/by/4.0/>

## Why these two are here

- **2606.25234** is the method we use. Its reference implementation is vendored at
  `vendor/block-sparse-featurizer`. Two of its results drive our design decisions: recovered concepts are
  typically two- to four-dimensional (which sets `group_size`), and it recovers **shadow and lighting manifolds**
  from DINO features — the exact nuisance variable that makes golden-reference differencing report a false
  defect.
- **2604.28119** is why the SAE alternative is a limitation rather than a preference: SAEs fragment a manifold
  across atoms (*dilution*), so no single SAE feature answers "is this lighting familiar?". Its evaluation code
  is vendored at `vendor/sae-manifold` and supplies our SAE baseline and the subspace-capture metric.

## Re-fetching

```bash
curl -L https://arxiv.org/pdf/2606.25234 -o papers/2606.25234-block-sparse-featurizers.pdf
curl -L https://arxiv.org/pdf/2604.28119 -o papers/2604.28119-do-saes-capture-concept-manifolds.pdf
```

Sizes: 47.8 MiB and 17.0 MiB respectively. See the repository README for how these are tracked.
