import torch
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import to_undirected, add_self_loops
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.nn.conv import GINConv
from torch_geometric.loader import DataLoader
import numpy as np
from collections import deque
import warnings
from sklearn.model_selection import StratifiedKFold
warnings.filterwarnings('ignore')

# Import our transforms
from delaunay_transform import DelaunayTransform
from cayley_transform import ExpanderTransform
import sys
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)

class GNN_node(torch.nn.Module):
    def __init__(self, num_features, hidden_dim, transform_name=None, is_cgp=False):
        super().__init__()
        self.transform_name = transform_name
        self.is_cgp = transform_name in ['EGP', 'CGP'] and is_cgp
        self.num_layers = 4
        self.dropout = 0.5
        
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        
        for layer in range(self.num_layers):
            input_dim = num_features if layer == 0 else hidden_dim
            gnn_nn = torch.nn.Sequential(
                torch.nn.Linear(input_dim, hidden_dim),
                torch.nn.BatchNorm1d(hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, hidden_dim)
            )
            self.convs.append(GINConv(gnn_nn))
            self.batch_norms.append(torch.nn.BatchNorm1d(hidden_dim))

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        if self.is_cgp:
            x_embeddings = torch.zeros_like(x)
            x_embeddings[~data.virtual_node_mask] = x[~data.virtual_node_mask]
            h = x_embeddings
        else:
            h = x.float()
        
        for layer in range(self.num_layers):
            if self.transform_name in ['EGP', 'CGP'] and layer % 2 == 1:
                h = self.convs[layer](h, data.expander_edge_index)
            else:
                h = self.convs[layer](h, edge_index)
                
            h = self.batch_norms[layer](h)
            if layer == self.num_layers - 1:
                h = F.dropout(h, self.dropout, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.dropout, training=self.training)
        
        return h

class GNN(torch.nn.Module):
    def __init__(self, num_features, hidden_dim, num_classes, transform_name=None, is_cgp=False):
        super().__init__()
        self.is_cgp = is_cgp
        self.gnn_node = GNN_node(num_features, hidden_dim, transform_name, is_cgp)
        self.pool = global_mean_pool
        self.classifier = torch.nn.Linear(hidden_dim, num_classes)

    def forward(self, data):
        h_node = self.gnn_node(data)
        
        if self.is_cgp:
            real_nodes_mask = ~data.virtual_node_mask
            h_node = h_node[real_nodes_mask]
            batch = data.batch[real_nodes_mask]
        else:
            batch = data.batch
            
        h_graph = self.pool(h_node, batch)
        return self.classifier(h_graph)

def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for data in loader:
        optimizer.zero_grad()
        data = data.to(device)
        out = model(data)
        loss = F.cross_entropy(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
        pred = out.argmax(dim=1)
        correct += (pred == data.y).sum().item()
        total += len(data.y)
    
    return total_loss / len(loader.dataset), correct / total

def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            pred = model(data).argmax(dim=1)
            correct += (pred == data.y).sum().item()
            total += len(data.y)
    
    return correct / total

def train_single_fold(model, train_loader, val_loader, params, device, fold):
    optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=20, min_lr=1e-5
    )
    
    best_val_acc = 0
    patience_counter = 0
    
    metrics = {
        'train_acc': [],
        'val_acc': [],
        'train_loss': [],
        'best_epoch': 0,
        'convergence_epoch': 0
    }
    
    for epoch in range(params['epochs']):
        # Training
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
        val_acc = evaluate(model, val_loader, device)
        
        metrics['train_acc'].append(train_acc)
        metrics['val_acc'].append(val_acc)
        metrics['train_loss'].append(train_loss)
        
        scheduler.step(val_acc)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            metrics['best_epoch'] = epoch
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= 50:
            metrics['convergence_epoch'] = epoch
            break
            
        if (epoch + 1) % 20 == 0:
            logging.info(f"Fold {fold} - Epoch {epoch+1}: Train Acc = {train_acc:.4f}, Val Acc = {val_acc:.4f}, Loss = {train_loss:.4f}")
    
    return metrics, best_val_acc

def k_fold_cross_validation(dataset, params, device, method_name, is_cgp, n_folds=5):
    # Get labels for stratification
    labels = torch.tensor([data.y.item() for data in dataset])
    
    # Initialize stratified k-fold
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(torch.zeros(len(dataset)), labels)):
        logging.info(f"\nFold {fold+1}/{n_folds}")
        
        # Create data loaders for this fold
        train_dataset = dataset[train_idx.tolist()]
        val_dataset = dataset[val_idx.tolist()]
        
        train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=params['batch_size'])
        
        # Initialize model
        model = GNN(
            num_features=dataset.num_features,
            hidden_dim=params['hidden_dim'],
            num_classes=dataset.num_classes,
            transform_name=method_name,
            is_cgp=is_cgp
        ).to(device)
        
        # Train model
        metrics, val_acc = train_single_fold(model, train_loader, val_loader, params, device, fold+1)
        fold_results.append(val_acc)
        
        logging.info(f"Fold {fold+1} validation accuracy: {val_acc:.4f}")
    
    mean_acc = np.mean(fold_results)
    std_acc = np.std(fold_results)
    
    logging.info(f"\nCross-validation results:")
    logging.info(f"Mean accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
    
    return mean_acc, std_acc

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    
    # Parameters
    params = {
        'dataset_name': 'MUTAG',
        'hidden_dim': 64,
        'batch_size': 32,
        'learning_rate': 0.001,
        'epochs': 200
    }
    
    # Load datasets with different transforms
    transforms = {
        'original': (NormalizeFeatures(), False),
        'EGP': (ExpanderTransform(params['dataset_name'], 'EGP'), False),
        'CGP': (ExpanderTransform(params['dataset_name'], 'CGP'), True),
        'delaunay': (DelaunayTransform(params['dataset_name']), False)
    }
    
    results = {}
    
    for method_name, (transform, is_cgp) in transforms.items():
        logging.info(f"\nRunning {method_name} experiment...")
        
        try:
            dataset = TUDataset(
                root=f'data/MUTAG-{method_name.lower()}', 
                name='MUTAG',
                transform=transform,
                pre_transform=NormalizeFeatures()
            )
            
            logging.info(f"Dataset loaded: {len(dataset)} graphs")
            
            mean_acc, std_acc = k_fold_cross_validation(
                dataset, params, device, method_name, is_cgp
            )
            
            results[method_name] = (mean_acc, std_acc)
            
        except Exception as e:
            logging.error(f"Error in {method_name} experiment: {str(e)}")
            continue
    
    logging.info("\nFinal Results:")
    for method_name, (mean, std) in results.items():
        logging.info(f"{method_name}: {mean:.4f} ± {std:.4f}")

if __name__ == "__main__":
    main()
