# Cayley Graph Experiment

This directory contains the comparison between Delaunay rewiring (using UMAP with Euclidean metric) and existing graph rewiring approaches (CGP, EGP).

## Files
- `experiment_summary.md`: Main results and findings
- `final_results_euclidean.txt`: Detailed experiment logs
- Implementation files:
  - `graph_classification.py`
  - `cayley_transform.py`
  - `delaunay_transform.py`

## Results
- Delaunay rewiring and EGP both achieve 88.00% accuracy
- Delaunay shows lowest variance (±2.45%)
- More details in `experiment_summary.md`
