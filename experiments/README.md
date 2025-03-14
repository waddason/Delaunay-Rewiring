# Delaunay Rewiring Experiments

This directory contains experiments evaluating the Delaunay rewiring approach on various datasets. Currently implemented:

## Wisconsin Dataset

### Quick Start
```bash
# Run baseline experiment
python wisconsin_experiment.py --mode baseline --num_runs 10

# Run Delaunay rewiring experiment
python wisconsin_experiment.py --mode delaunay --num_runs 10

# Compare results (latest runs)
python wisconsin/compare_results.py

# Compare results with aggregation
python wisconsin/compare_results.py --aggregate
```

### Directory Structure
```
wisconsin/
├── plots/
│   ├── degree_distributions/     # Degree distribution plots
│   ├── curvature_distributions/ # Curvature distribution plots
│   └── performance_comparison_*.png
├── results/
│   ├── baseline_results_*.json   # Baseline experiment results
│   └── delaunay_results_*.json  # Delaunay experiment results
├── logs/
│   ├── baseline/                # Training logs for baseline
│   └── delaunay/               # Training logs for Delaunay
├── compare_results.py           # Results analysis script
└── experiment_report.md         # Comprehensive experiment report
```

### Requirements
- PyTorch Geometric
- UMAP-learn
- NetworkX
- GraphRicciCurvature
- matplotlib
- scipy

### Reproducing Results
1. Ensure all required packages are installed
2. Run both baseline and Delaunay experiments
3. Use compare_results.py to analyze results
4. Check experiment_report.md for detailed findings

### Key Results
- GCN: 54.90% → 67.55% (+12.6%)
- GAT: 55.88% → 69.12% (+13.2%)
- Homophily: 0.366 → 0.718 (+96%)

### Notes
- All experiments use fixed random seeds for reproducibility
- Results are automatically saved with timestamps
- Visualization plots are generated automatically
- See experiment_report.md for full details
