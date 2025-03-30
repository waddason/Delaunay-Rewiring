# Cayley Graph Experiment

This directory contains the comparison between our Delaunay rewiring method and existing graph rewiring approaches (CGP, EGP).

## Files
- `experiment_summary.md`: Main results and findings
- `final_results.txt`: Detailed experiment logs
- Original implementation source files:
  - `graph_classification.py`
  - `cayley_transform.py`

## Results
Delaunay rewiring consistently outperformed other methods, showing:
- Higher accuracy: 89.00% ± 5.83%
- Faster convergence: 58.2 epochs average
- Earlier peak performance: 8.2 epochs average

For full analysis see `experiment_summary.md`.
