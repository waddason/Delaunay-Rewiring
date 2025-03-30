# Graph Rewiring Methods Comparison

## Experiment Setup
- Dataset: MUTAG
- Evaluation: 5 different splits (seeds: 42, 123, 456, 789, 101112)
- Methods compared: Original graph, EGP, CGP, Delaunay rewiring

## Results

### Accuracy
1. **Delaunay**: 89.00% ± 5.83%
2. **EGP**: 87.00% ± 4.00%
3. **CGP**: 84.00% ± 5.83%
4. **Original**: 82.00% ± 10.30%

### Training Speed
| Method   | Convergence | Best Performance |
|----------|-------------|------------------|
| Delaunay | 58.2 epochs | 8.2 epochs      |
| CGP      | 62.2 epochs | 12.2 epochs     |
| EGP      | 75.4 epochs | 28.2 epochs     |
| Original | 76.6 epochs | 27.0 epochs     |

## Key Findings
- Delaunay rewiring achieves best performance and fastest convergence
- EGP shows most stable performance (lowest std: ±4.00%)
- Original method has highest variance (±10.30%)
- All rewiring methods improve over baseline
