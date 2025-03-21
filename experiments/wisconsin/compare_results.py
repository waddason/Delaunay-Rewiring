import json
import os
import glob
import argparse
import sys
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import RESULTS_DIR, PLOTS_DIR

def load_results(mode, specific_file=None):
    """Load results files for given mode."""
    pattern = os.path.join(RESULTS_DIR, f"{mode}_results_*.json")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No results found for {mode}")
    
    if specific_file:
        if specific_file not in files:
            raise FileNotFoundError(f"Specified file not found: {specific_file}")
        files = [specific_file]
    
    results = []
    for file in files:
        with open(file, 'r') as f:
            timestamp = '_'.join(os.path.basename(file).split('_')[-2:]).replace('.json', '')
            results.append({
                'timestamp': timestamp,
                'data': json.load(f)
            })
    
    return results

def aggregate_results(results):
    """Aggregate multiple result files."""
    all_gcn_acc = []
    all_gat_acc = []
    graph_stats = []
    
    for result in results:
        data = result['data']
        all_gcn_acc.extend(data['gcn_results']['accuracies'])
        all_gat_acc.extend(data['gat_results']['accuracies'])
        graph_stats.append({
            'timestamp': result['timestamp'],
            'homophily': data['graph_stats']['homophily'],
            'curvature_range': [
                data['graph_stats']['curvature_stats']['first_decile'],
                data['graph_stats']['curvature_stats']['ninth_decile']
            ],
            'degree_stats': data['graph_stats']['degree_stats']
        })
    
    return {
        'gcn_accuracies': all_gcn_acc,
        'gat_accuracies': all_gat_acc,
        'graph_stats': graph_stats,
        'dataset_stats': results[0]['data']['dataset_stats']  # Use first file for dataset stats
    }

def print_dataset_info(data):
    """Print dataset statistics."""
    print("Dataset Statistics:")
    print("-" * 50)
    print(f"Nodes: {data['num_nodes']}")
    print(f"Features: {data['num_features']}")
    print(f"Classes: {data['num_classes']}")
    print()

def print_graph_stats(stats, mode):
    """Print graph property statistics."""
    print(f"{mode.title()} Graph Properties:")
    print("-" * 50)
    for i, stat in enumerate(stats):
        timestamp = datetime.strptime(stat['timestamp'], '%Y%m%d_%H%M%S')
        print(f"Experiment {i+1} ({timestamp.strftime('%Y-%m-%d %H:%M:%S')}):")
        print(f"- Mean Degree: {stat['degree_stats']['mean_degree']:.2f}")
        print(f"- Homophily: {stat['homophily']:.3f}")
        print(f"- Curvature Range: [{stat['curvature_range'][0]:.3f}, {stat['curvature_range'][1]:.3f}]")
    print()

def print_model_results(baseline_accs, delaunay_accs, model_name):
    """Print detailed model performance comparison."""
    print(f"{model_name} Results:")
    print("-" * 50)
    
    # Basic statistics
    b_mean, b_std = np.mean(baseline_accs), np.std(baseline_accs)
    d_mean, d_std = np.mean(delaunay_accs), np.std(delaunay_accs)
    improvement = (d_mean - b_mean) * 100
    
    print(f"Baseline: {b_mean:.4f} ± {b_std:.4f} (min: {min(baseline_accs):.4f}, max: {max(baseline_accs):.4f})")
    print(f"Delaunay: {d_mean:.4f} ± {d_std:.4f} (min: {min(delaunay_accs):.4f}, max: {max(delaunay_accs):.4f})")
    print(f"Improvement: {improvement:.1f}%")
    
    # Statistical significance
    t_stat, p_value = stats.ttest_ind(baseline_accs, delaunay_accs)
    print(f"Statistical Significance:")
    print(f"- t-statistic: {t_stat:.4f}")
    print(f"- p-value: {p_value:.4f}")
    print(f"- Significant: {'Yes' if p_value < 0.05 else 'No'} (α=0.05)")
    print()

def plot_results(baseline_results, delaunay_results, save_dir):
    """Create and save comparison plots."""
    # Performance boxplot
    plt.figure(figsize=(15, 7))
    
    # GCN comparison
    plt.subplot(1, 2, 1)
    box_data = [baseline_results['gcn_accuracies'], delaunay_results['gcn_accuracies']]
    bp = plt.boxplot(box_data, tick_labels=['Baseline', 'Delaunay'])
    plt.title('GCN Performance Comparison', fontsize=12, pad=10)
    plt.ylabel('Accuracy', fontsize=10)
    
    # Add statistical annotations
    b_mean = np.mean(baseline_results['gcn_accuracies'])
    d_mean = np.mean(delaunay_results['gcn_accuracies'])
    plt.plot([1, 1], [b_mean, b_mean], 'k*', markersize=10, label=f'Mean: {b_mean:.3f}')
    plt.plot([2, 2], [d_mean, d_mean], 'k*', markersize=10, label=f'Mean: {d_mean:.3f}')
    
    # GAT comparison
    plt.subplot(1, 2, 2)
    box_data = [baseline_results['gat_accuracies'], delaunay_results['gat_accuracies']]
    bp = plt.boxplot(box_data, tick_labels=['Baseline', 'Delaunay'])
    plt.title('GAT Performance Comparison', fontsize=12, pad=10)
    plt.ylabel('Accuracy', fontsize=10)
    
    # Add statistical annotations
    b_mean = np.mean(baseline_results['gat_accuracies'])
    d_mean = np.mean(delaunay_results['gat_accuracies'])
    plt.plot([1, 1], [b_mean, b_mean], 'k*', markersize=10, label=f'Mean: {b_mean:.3f}')
    plt.plot([2, 2], [d_mean, d_mean], 'k*', markersize=10, label=f'Mean: {d_mean:.3f}')
    
    # Add legend
    plt.legend(fontsize=8)
    
    # Adjust layout and save
    plt.suptitle('Performance Comparison: Baseline vs Delaunay Rewiring', 
                fontsize=14, y=1.02)
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(os.path.join(save_dir, f'performance_comparison_{timestamp}.png'), 
                bbox_inches='tight', dpi=300)
    plt.close()

def main(args):
    # Load results
    baseline_results = load_results("baseline", args.baseline)
    delaunay_results = load_results("delaunay", args.delaunay)
    
    if args.aggregate:
        print(f"Aggregating results from {len(baseline_results)} baseline and {len(delaunay_results)} delaunay experiments\n")
        baseline_agg = aggregate_results(baseline_results)
        delaunay_agg = aggregate_results(delaunay_results)
        
        # Print comprehensive analysis
        print_dataset_info(baseline_agg['dataset_stats'])
        print_graph_stats(baseline_agg['graph_stats'], "baseline")
        print_graph_stats(delaunay_agg['graph_stats'], "delaunay")
        
        print("Model Performance Comparison:")
        print("=" * 50)
        print_model_results(baseline_agg['gcn_accuracies'], 
                          delaunay_agg['gcn_accuracies'], "GCN")
        print_model_results(baseline_agg['gat_accuracies'], 
                          delaunay_agg['gat_accuracies'], "GAT")
        
        # Create plots
        plot_results(baseline_agg, delaunay_agg, PLOTS_DIR)
        
    else:
        # Use only the most recent results (or specified files)
        baseline = baseline_results[-1]['data']
        delaunay = delaunay_results[-1]['data']
        
        print(f"Comparing results from:")
        print(f"Baseline: {baseline_results[-1]['timestamp']}")
        print(f"Delaunay: {delaunay_results[-1]['timestamp']}\n")
        
        # Print dataset info
        print_dataset_info(baseline['dataset_stats'])
        
        # Print single experiment results
        print("Graph Properties Comparison:")
        print("-" * 50)
        print("Baseline Graph:")
        print(f"- Mean Degree: {baseline['graph_stats']['degree_stats']['mean_degree']:.2f}")
        print(f"- Homophily: {baseline['graph_stats']['homophily']:.3f}")
        print(f"- Curvature Range: [{baseline['graph_stats']['curvature_stats']['first_decile']:.3f}, "
              f"{baseline['graph_stats']['curvature_stats']['ninth_decile']:.3f}]")
        print()
        
        print("Delaunay Graph:")
        print(f"- Mean Degree: {delaunay['graph_stats']['degree_stats']['mean_degree']:.2f}")
        print(f"- Homophily: {delaunay['graph_stats']['homophily']:.3f}")
        print(f"- Curvature Range: [{delaunay['graph_stats']['curvature_stats']['first_decile']:.3f}, "
              f"{delaunay['graph_stats']['curvature_stats']['ninth_decile']:.3f}]")
        print()
        
        print("Model Performance Comparison:")
        print("-" * 50)
        print_model_results(baseline['gcn_results']['accuracies'],
                          delaunay['gcn_results']['accuracies'], "GCN")
        print_model_results(baseline['gat_results']['accuracies'],
                          delaunay['gat_results']['accuracies'], "GAT")
        
        # Create plots for single experiment comparison
        plot_results({
            'gcn_accuracies': baseline['gcn_results']['accuracies'],
            'gat_accuracies': baseline['gat_results']['accuracies']
        }, {
            'gcn_accuracies': delaunay['gcn_results']['accuracies'],
            'gat_accuracies': delaunay['gat_results']['accuracies']
        }, PLOTS_DIR)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compare experiment results')
    parser.add_argument('--baseline', type=str, help='Specific baseline results file to use')
    parser.add_argument('--delaunay', type=str, help='Specific delaunay results file to use')
    parser.add_argument('--aggregate', action='store_true', help='Aggregate results from all available files')
    
    args = parser.parse_args()
    main(args)
