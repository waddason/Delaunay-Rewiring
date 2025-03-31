# Graph Rewiring Methods Comparison

## Experiment Setup
- Dataset: MUTAG
- Evaluation: 5 different splits (seeds: 42, 123, 456, 789, 101112)
- Methods compared: Original graph, EGP, CGP, Delaunay rewiring
- UMAP Settings for Delaunay: 
  - n_neighbors: 5
  - min_dist: 0.1
  - metric: Euclidean
  - (Parameters selected through hyperparameter grid search)

## Results

### Accuracy
1. **Delaunay**: 88.00% ± 2.45%
1. **EGP**: 88.00% ± 4.00%
2. **Original**: 84.00% ± 5.83%
3. **CGP**: 83.00% ± 5.10%

### Training Speed
| Method   | Convergence | Best Performance |
|----------|-------------|------------------|
| CGP      | 54.6 epochs | 9.6 epochs      |
| Delaunay | 68.4 epochs | 18.8 epochs     |
| Original | 78.4 epochs | 29.0 epochs     |
| EGP      | 84.2 epochs | 37.2 epochs     |

## Key Findings
- Delaunay with optimized UMAP parameters ties EGP for best accuracy (88.00%)
- Delaunay shows most stable performance (lowest std: ±2.45%)
- CGP achieves fastest convergence but lower accuracy
- EGP matches Delaunay's accuracy but requires longer training
- All rewiring methods improve stability over baseline
