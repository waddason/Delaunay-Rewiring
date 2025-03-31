# Graph Rewiring Methods Comparison

This directory contains a comparison of different graph transformation methods evaluated using stratified 5-fold cross-validation on the MUTAG dataset.

## Implementation Details

### Model Architecture
- 4-layer Graph Isomorphism Network (GIN)
- Per layer: Linear → BatchNorm → ReLU → Linear
- Hidden dimension: 64
- Dropout: 0.5
- Global mean pooling
- Adam optimizer (lr=0.001)
- ReduceLROnPlateau scheduler (patience=20, factor=0.5)

### Training & Evaluation
- Dataset: MUTAG (188 graphs)
- Stratified 5-fold cross-validation
- Batch size: 32
- Maximum epochs: 200
- Early stopping patience: 50
- Random seed: 42

### Graph Transforms
1. Delaunay Rewiring
   - UMAP dimensionality reduction to 2D
   - Parameters: n_neighbors=5, min_dist=0.1, metric=euclidean
   - QJ option for robust triangulation
   - Small random noise (1e-6) to prevent coplanar points
   - Result graphs are undirected with self-loops

2. CGP (Cayley Graph Propagation)
   - Uses complete Cayley graph structure
   - Alternates between original and Cayley edges
   - Includes virtual nodes with zero features

3. EGP (Expander Graph Propagation)
   - Truncated version of Cayley graph
   - Alternates between original and expander edges
   - No virtual nodes

## Latest Results
- CGP: 92.01% ± 3.80%
- EGP: 89.90% ± 1.94%
- Original: 89.84% ± 4.05%
- Delaunay: 88.31% ± 2.64%

See `experiment_summary.md` for detailed analysis.
