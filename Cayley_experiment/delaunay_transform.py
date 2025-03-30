import torch
import numpy as np
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import add_self_loops, to_undirected
import umap
from scipy.spatial import Delaunay
import warnings

class DelaunayTransform(BaseTransform):
    """Transform applying Delaunay triangulation on UMAP-reduced node features."""
    def __init__(self, dataset_name, n_neighbors=5, min_dist=0.25, metric='cosine'):
        super().__init__()
        self.dataset = dataset_name
        # Using optimal parameters from hyperparameter study
        self.reducer = umap.UMAP(
            n_components=2,
            n_neighbors=n_neighbors,  # Best: 5
            min_dist=min_dist,       # Best: 0.25 with cosine
            metric=metric,           # Using cosine for better separation
            random_state=42
        )
        self.cached_projections = {}
        self.cached_edges = {}
        
    def __call__(self, data):
        num_nodes = data.num_nodes
        
        # Handle datasets without node features
        if self.dataset in ['COLLAB', 'REDDIT-BINARY', 'IMDB-BINARY'] and data.x is None:
            data.x = torch.ones((num_nodes, 1))
        
        # Use cached results if available
        graph_id = data.get('id', hash(str(data.x)))
        
        try:
            if graph_id in self.cached_edges:
                data.edge_index = self.cached_edges[graph_id]
                return data
            
            # Project node features to 2D using UMAP
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if graph_id not in self.cached_projections:
                    # Normalize features for cosine metric
                    x = data.x.numpy()
                    norms = np.linalg.norm(x, axis=1, keepdims=True)
                    norms[norms == 0] = 1
                    x_normalized = x / norms
                    self.cached_projections[graph_id] = self.reducer.fit_transform(x_normalized)
                reduced_data = self.cached_projections[graph_id]
            
            # Create Delaunay graph
            edge_index = self._create_delaunay_graph(torch.tensor(reduced_data))
            
            # Make undirected and add self-loops
            edge_index = to_undirected(edge_index)
            edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
            
            # Cache the results
            self.cached_edges[graph_id] = edge_index
            
            # Store edges
            data.edge_index = edge_index
            
            return data
            
        except Exception as e:
            print(f"Warning: Delaunay transform failed with error: {str(e)}")
            print("Falling back to original graph structure")
            return data
        
    def _create_delaunay_graph(self, positions):
        """Create graph edges using Delaunay triangulation."""
        positions = positions.cpu().numpy()
        
        # Add small random noise to prevent coplanar points
        positions += np.random.normal(0, 1e-6, positions.shape)
        
        # Compute Delaunay triangulation with QJ option for robustness
        delaunay = Delaunay(positions, qhull_options="QJ")
        edges = []
        
        # Extract edges from triangulation
        for simplex in delaunay.simplices:
            for i in range(3):
                for j in range(i + 1, 3):
                    edges.append((simplex[i], simplex[j]))
                    edges.append((simplex[j], simplex[i]))  # Make undirected
                    
        edge_index = torch.tensor(edges).t().contiguous()
        return edge_index
