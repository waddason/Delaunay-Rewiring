# Delaunay Rewiring for Graph Neural Networks

This repository contains the implementation and experiments for the Delaunay rewiring approach to improve Graph Neural Network performance by addressing over-squashing through geometric graph reconstruction.

## Overview

The Delaunay rewiring method reconstructs graph connectivity using Delaunay triangulation in the learned feature space, leading to:
- Improved homophily
- Reduced over-squashing
- Better message passing
- Significant performance gains

## Key Results on Wisconsin Dataset

| Model | Baseline | Delaunay | Improvement |
|-------|----------|----------|-------------|
| GCN   | 54.90%   | 67.55%   | +12.6%      |
| GAT   | 55.88%   | 69.12%   | +13.2%      |

Graph Properties:
- Homophily: 0.366 → 0.718 (+96%)
- Curvature Range: [-0.475, 0.250] → [-0.214, 0.200]
- All improvements statistically significant (p < 0.0001)

## Repository Structure

```
Delaunay-Rewiring/
├── experiments/           # Experimental code and results
│   ├── wisconsin/        # Wisconsin dataset experiments
│   ├── README.md         # Experiment documentation
│   ├── requirements.txt  # Package dependencies
│   └── SETUP.md         # Environment setup guide
├── defense/              # Presentation materials
└── report/              # Paper drafts and materials
```

## Getting Started

1. Set up the environment:
```bash
# Install dependencies (see experiments/SETUP.md for detailed instructions)
cd experiments
pip install -r requirements.txt
```

2. Run experiments:
```bash
# Run baseline and Delaunay experiments
python wisconsin_experiment.py --mode baseline --num_runs 10
python wisconsin_experiment.py --mode delaunay --num_runs 10

# Analyze results
python experiments/wisconsin/compare_results.py --aggregate
```

3. View results:
- Check `experiments/wisconsin/experiment_report.md` for detailed analysis
- Plots are saved in `experiments/wisconsin/plots/`
- Raw results in `experiments/wisconsin/results/`

## Documentation

- [Experiment Documentation](experiments/README.md)
- [Setup Guide](experiments/SETUP.md)
- [Full Experiment Report](experiments/wisconsin/experiment_report.md)

## Requirements

- Python 3.10+
- PyTorch with CUDA support
- PyTorch Geometric
- UMAP-learn
- NetworkX (<3.0)
- GraphRicciCurvature

See [requirements.txt](experiments/requirements.txt) for full list.