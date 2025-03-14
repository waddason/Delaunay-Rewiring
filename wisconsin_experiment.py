import os
import time
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.datasets import WebKB
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import (
    homophily,
    add_self_loops,
    to_undirected,
)
from torch_geometric.nn import GCNConv, GATConv
import umap
import networkx as nx
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
from GraphRicciCurvature.OllivierRicci import OllivierRicci
from datetime import datetime

class GCN(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes):
        super().__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x

class GAT(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes):
        super().__init__()
        self.conv1 = GATConv(num_features, hidden_channels, heads=8, dropout=0.5)
        self.conv2 = GATConv(hidden_channels * 8, num_classes, heads=1, dropout=0.5)

    def forward(self, x, edge_index):
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x

class Experiment:
    def __init__(self, exp_dir, mode):
        self.exp_dir = exp_dir
        self.mode = mode
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create directories if they don't exist
        self.plots_dir = os.path.join(exp_dir, 'plots')
        self.results_dir = os.path.join(exp_dir, 'results')
        self.logs_dir = os.path.join(exp_dir, 'logs', mode)
        
        os.makedirs(self.plots_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        
        # Set random seeds
        torch.manual_seed(42)
        np.random.seed(42)
        
    def save_plot(self, plt, name):
        """Save plot to appropriate subdirectory."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.mode}_{name}_{timestamp}.png"
        
        # Determine subdirectory based on plot type
        if 'degree' in name:
            subdir = 'degree_distributions'
        elif 'curvature' in name:
            subdir = 'curvature_distributions'
        else:
            subdir = ''
        
        save_dir = os.path.join(self.plots_dir, subdir)
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, filename))
        plt.close()

    def save_results(self, results):
        """Save results to JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.mode}_results_{timestamp}.json"
        with open(os.path.join(self.results_dir, filename), 'w') as f:
            json.dump(results, f, indent=4)

    def create_delaunay_graph(self, positions):
        """Create Delaunay graph from node positions."""
        positions = positions.cpu().numpy()
        delaunay = Delaunay(positions, qhull_options="QJ")
        edges = []
        for simplex in delaunay.simplices:
            for i in range(3):
                for j in range(i + 1, 3):
                    edge = (simplex[i], simplex[j])
                    edges.append(edge)
                    edges.append((simplex[j], simplex[i]))
        
        delaunay_graph = nx.Graph(edges)
        edge_index = torch.tensor(list(delaunay_graph.edges)).t().contiguous()
        return edge_index

    def plot_degree_distribution(self, edge_index, title):
        """Plot and save degree distribution."""
        G = nx.Graph()
        G.add_edges_from(edge_index.t().tolist())
        degrees = [val for (node, val) in G.degree()]
        
        plt.figure()
        plt.hist(degrees, bins=range(min(degrees), max(degrees) + 1), 
                alpha=0.7, color="b", edgecolor="black")
        plt.xlabel("Degree")
        plt.ylabel("Number of nodes")
        plt.title(f"Degree Distribution - {title}")
        plt.grid(True)
        
        self.save_plot(plt, f"degree_dist_{title.lower().replace(' ', '_')}")
        
        stats = {
            "max_degree": max(degrees),
            "min_degree": min(degrees),
            "mean_degree": sum(degrees)/len(degrees)
        }
        return stats

    def plot_curvature_distribution(self, edge_index, title):
        """Plot and save curvature distribution."""
        G = nx.Graph()
        G.add_edges_from(edge_index.t().tolist())
        
        orc = OllivierRicci(G, alpha=0.5, verbose="ERROR")
        orc.compute_ricci_curvature()
        G = orc.G
        
        curvatures = list(nx.get_edge_attributes(G, "ricciCurvature").values())
        curvatures.sort()
        n = len(curvatures)
        
        stats = {
            "first_decile": curvatures[int(n * 0.1)],
            "ninth_decile": curvatures[int(n * 0.9)]
        }
        
        plt.figure()
        plt.hist(curvatures, bins=20)
        plt.xlabel("Ricci curvature")
        plt.ylabel("Number of edges")
        plt.title(f"Curvature Distribution - {title}")
        
        self.save_plot(plt, f"curvature_dist_{title.lower().replace(' ', '_')}")
        return stats

    def train_model(self, model, data, train_mask, val_mask, optimizer, criterion):
        """Train the model for one epoch."""
        model.train()
        optimizer.zero_grad()
        out = model(data.x.to(self.device), data.edge_index.to(self.device))
        loss = criterion(out[train_mask], data.y[train_mask].to(self.device))
        loss.backward()
        optimizer.step()
        
        model.eval()
        with torch.no_grad():
            out = model(data.x.to(self.device), data.edge_index.to(self.device))
            pred = out.argmax(dim=1)
            train_acc = (pred[train_mask] == data.y[train_mask].to(self.device)).float().mean()
            val_acc = (pred[val_mask] == data.y[val_mask].to(self.device)).float().mean()
        
        return loss.item(), train_acc.item(), val_acc.item()

    def test_model(self, model, data, mask):
        """Test the model."""
        model.eval()
        with torch.no_grad():
            out = model(data.x.to(self.device), data.edge_index.to(self.device))
            pred = out.argmax(dim=1)
            acc = (pred[mask] == data.y[mask].to(self.device)).float().mean()
        return acc.item()

    def run_experiment(self, model_class, data, train_mask, val_mask, test_mask, 
                      hidden_channels, lr, weight_decay, max_epochs=2000, patience=100):
        """Run a single experiment."""
        model = model_class(
            num_features=data.num_features,
            hidden_channels=hidden_channels,
            num_classes=data.y.max().item() + 1
        ).to(self.device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = torch.nn.CrossEntropyLoss()
        
        best_val_acc = 0
        best_test_acc = 0
        patience_counter = 0
        
        for epoch in range(1, max_epochs + 1):
            loss, train_acc, val_acc = self.train_model(
                model, data, train_mask, val_mask, optimizer, criterion
            )
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_test_acc = self.test_model(model, data, test_mask)
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                break
        
        return best_test_acc, epoch

def main(args):
    # Initialize experiment
    exp = Experiment(args.exp_dir, args.mode)
    print(f"Running {args.mode} experiment on {exp.device}")
    
    # Load and preprocess data
    dataset = WebKB(root="data/", name="Wisconsin", transform=NormalizeFeatures())
    data = dataset[0]
    
    # Log dataset statistics
    dataset_stats = {
        "num_nodes": data.num_nodes,
        "num_edges": data.num_edges,
        "num_features": data.num_features,
        "num_classes": data.y.max().item() + 1
    }
    
    # Process graph based on mode
    if args.mode == "delaunay":
        # Create Delaunay graph
        reducer = umap.UMAP(n_components=2)
        reduced_data = reducer.fit_transform(data.x)
        edge_index = exp.create_delaunay_graph(torch.tensor(reduced_data))
        edge_index = to_undirected(edge_index)
        edge_index, _ = add_self_loops(edge_index, num_nodes=data.num_nodes)
    else:
        # Use original graph
        edge_index = to_undirected(data.edge_index)
        edge_index, _ = add_self_loops(edge_index, num_nodes=data.num_nodes)
    
    # Analyze and plot graph properties
    degree_stats = exp.plot_degree_distribution(edge_index, f"{args.mode.title()} Graph")
    curvature_stats = exp.plot_curvature_distribution(edge_index, f"{args.mode.title()} Graph")
    graph_homophily = float(homophily(edge_index, data.y))
    
    # Store graph statistics
    graph_stats = {
        "degree_stats": degree_stats,
        "curvature_stats": curvature_stats,
        "homophily": graph_homophily
    }
    
    # Experiment settings
    settings = {
        "num_runs": args.num_runs,
        "hidden_channels": args.hidden_channels,
        "lr": args.lr,
        "weight_decay": args.weight_decay
    }
    
    # Results storage
    gcn_results = []
    gat_results = []
    
    # Run experiments
    for run in range(args.num_runs):
        print(f"\nRun {run + 1}/{args.num_runs}")
        
        # Create random splits
        indices = torch.randperm(data.num_nodes)
        train_size = int(0.6 * data.num_nodes)
        val_size = int(0.2 * data.num_nodes)
        
        train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        
        train_mask[indices[:train_size]] = True
        val_mask[indices[train_size:train_size + val_size]] = True
        test_mask[indices[train_size + val_size:]] = True
        
        # Set graph structure
        data.edge_index = edge_index
        
        # Run with GCN
        print("Training GCN...")
        gcn_acc, gcn_epochs = exp.run_experiment(
            GCN, data, train_mask, val_mask, test_mask,
            args.hidden_channels, args.lr, args.weight_decay
        )
        gcn_results.append(gcn_acc)
        
        # Run with GAT
        print("Training GAT...")
        gat_acc, gat_epochs = exp.run_experiment(
            GAT, data, train_mask, val_mask, test_mask,
            args.hidden_channels, args.lr, args.weight_decay
        )
        gat_results.append(gat_acc)
    
    # Compile results
    results = {
        "dataset_stats": dataset_stats,
        "graph_stats": graph_stats,
        "settings": settings,
        "gcn_results": {
            "accuracies": gcn_results,
            "mean": np.mean(gcn_results),
            "std": np.std(gcn_results)
        },
        "gat_results": {
            "accuracies": gat_results,
            "mean": np.mean(gat_results),
            "std": np.std(gat_results)
        }
    }
    
    # Save results
    exp.save_results(results)
    
    # Print final results
    print("\nFinal Results:")
    print(f"GCN Test Accuracy: {np.mean(gcn_results):.4f} ± {np.std(gcn_results):.4f}")
    print(f"GAT Test Accuracy: {np.mean(gat_results):.4f} ± {np.std(gat_results):.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Wisconsin experiments')
    parser.add_argument('--mode', type=str, choices=['baseline', 'delaunay'], required=True,
                      help='Experiment mode: baseline or delaunay')
    parser.add_argument('--exp_dir', type=str, default='Delaunay-Rewiring/experiments/wisconsin',
                      help='Directory for saving experiment results')
    parser.add_argument('--num_runs', type=int, default=10,
                      help='Number of experimental runs')
    parser.add_argument('--hidden_channels', type=int, default=32,
                      help='Number of hidden channels')
    parser.add_argument('--lr', type=float, default=0.005,
                      help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-6,
                      help='Weight decay')
    
    args = parser.parse_args()
    main(args)
