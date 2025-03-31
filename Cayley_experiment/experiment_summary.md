# Graph Rewiring Methods Comparison

## Experiment Setup
- Dataset: MUTAG
- Model: 4-layer GIN (hidden_dim=64)
- Evaluation: Stratified 5-fold cross-validation
- Methods compared: Original graph, EGP, CGP, Delaunay rewiring

### Implementation Details
1. Model Architecture:
   - Graph Isomorphism Network (GIN)
   - 4 layers with alternating edge types
   - Hidden dimension: 64
   - Dropout: 0.5
   - Global mean pooling
   - BatchNorm after each convolution

2. Training:
   - Adam optimizer (lr=0.001)
   - ReduceLROnPlateau scheduler
   - Batch size: 32
   - Early stopping patience: 50
   - Max epochs: 200

3. Delaunay Transform:
   - UMAP parameters:
     - n_neighbors: 5
     - min_dist: 0.1
     - metric: euclidean
     - n_components: 2
   - QJ option for robust triangulation
   - Small random noise for stability
   - Undirected with self-loops

## Results

### Cross-validation Accuracy (5-fold)
1. **CGP**: 92.01% ± 3.80%
2. **EGP**: 89.90% ± 1.94%
3. **Original**: 89.84% ± 4.05%
4. **Delaunay**: 88.31% ± 2.64%

### Per-Method Analysis

#### CGP (92.01% ± 3.80%)
- Highest average accuracy
- Individual fold accuracies: 92.11%, 89.47%, 97.37%, 86.49%, 94.59%
- Best performance in fold 3 (97.37%)
- Moderate variance across folds

#### EGP (89.90% ± 1.94%)
- Most stable performance (lowest std)
- Individual fold accuracies: 89.47%, 86.84%, 92.11%, 89.19%, 91.89%
- Consistent performance across folds
- Low variance (±1.94%)

#### Original (89.84% ± 4.05%)
- Strong baseline performance
- Individual fold accuracies: 92.11%, 92.11%, 94.74%, 86.49%, 83.78%
- Highest variance among methods (±4.05%)
- Performance drops in later folds

#### Delaunay (88.31% ± 2.64%)
- Individual fold accuracies: 86.84%, 92.11%, 84.21%, 89.19%, 89.19%
- Moderate variance (±2.64%)
- Most consistent in folds 4 and 5 (89.19%)
- Lower but stable performance

## Key Findings
1. Method Comparison:
   - CGP achieves highest accuracy but moderate variance
   - EGP provides best stability with good accuracy
   - Original method surprisingly competitive but unstable
   - Delaunay offers consistent but slightly lower performance

2. Statistical Significance:
   - All methods perform within ±4.05% standard deviation
   - EGP shows remarkably low variance (±1.94%)
   - Stratified k-fold helps reduce evaluation bias

3. Training Dynamics:
   - Early stopping typically triggers between 40-120 epochs
   - Learning rate reduction helps stabilize training
   - All methods benefit from patience in validation
