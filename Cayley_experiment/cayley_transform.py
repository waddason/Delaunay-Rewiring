import torch
import numpy as np
from collections import deque
from torch_geometric.transforms import BaseTransform
from typing import Dict

def get_cayley_graph(n):
    """Get the edge index of the Cayley graph (Cay(SL(2, Z_n); S_n))."""
    generators = np.array([
        [[1, 1], [0, 1]],
        [[1, n-1], [0, 1]],
        [[1, 0], [1, 1]],
        [[1, 0], [n-1, 1]]
    ])
    
    ind = 1
    queue = deque([np.array([[1, 0], [0, 1]])])
    nodes = {(1, 0, 0, 1): 0}
    senders = []
    receivers = []
    
    while queue:
        x = queue.pop()
        x_flat = (x[0][0], x[0][1], x[1][0], x[1][1])
        
        assert x_flat in nodes
        ind_x = nodes[x_flat]
        
        for i in range(4):
            tx = np.matmul(x, generators[i])
            tx = np.mod(tx, n)
            tx_flat = (tx[0][0], tx[0][1], tx[1][0], tx[1][1])
            
            if tx_flat not in nodes:
                nodes[tx_flat] = ind
                ind += 1
                queue.append(tx)
                
            ind_tx = nodes[tx_flat]
            senders.append(ind_x)
            receivers.append(ind_tx)
    
    return torch.tensor([senders, receivers])

class ExpanderTransform(BaseTransform):
    """Transform applying Cayley Graph Propagation."""
    def __init__(self, dataset, type):
        super().__init__()
        self.dataset = dataset
        self.type = type
        self.cayley_memory: Dict[int, torch.Tensor] = {}
        self.cayley_node_memory: Dict[int, torch.Tensor] = {}

    def __call__(self, data):
        num_nodes = data.num_nodes
        
        # For certain TUDataset(s) the graph structure needs to be augmented
        if self.dataset in ['COLLAB', 'REDDIT-BINARY', 'IMDB-BINARY'] and data.x is None:
            data.x = torch.ones((num_nodes, 1))
            
        cayley_n = self._get_cayley_n(num_nodes)
        
        # EGP Transform
        if self.type == 'EGP':
            data.expander_edge_index = self._get_egp_edge_index(cayley_n, num_nodes)
            return data
            
        # CGP Transform
        data.expander_edge_index, cayley_num_nodes = self._get_cgp_edge_index(cayley_n)
        virtual_num_nodes = cayley_num_nodes - num_nodes
        
        # Create virtual node mask
        data.virtual_node_mask = torch.cat((
            torch.zeros(num_nodes, dtype=torch.bool),
            torch.ones(virtual_num_nodes, dtype=torch.bool)
        ))
        
        # Update input features for virtual nodes
        data.num_nodes = cayley_num_nodes
        data.cayley_num_nodes = cayley_num_nodes
        data.x = torch.cat((
            data.x,
            torch.zeros((virtual_num_nodes, data.x.shape[1]), dtype=data.x.dtype)
        ))
        
        return data

    def _get_cayley_n(self, num_nodes):
        n = 1
        while self._cayley_graph_size(n) < num_nodes:
            n += 1
        return n

    def _cayley_graph_size(self, n):
        n = int(n)
        return n * n * n  # Use n³ for proper sizing

    def _get_egp_edge_index(self, cayley_n, num_nodes):
        if cayley_n not in self.cayley_memory:
            self.cayley_memory[cayley_n] = get_cayley_graph(cayley_n)
        
        cayley_graph_edge_index = self.cayley_memory[cayley_n].clone()
        
        if num_nodes not in self.cayley_node_memory:
            truncated_edge_index = cayley_graph_edge_index[:, 
                torch.logical_and(cayley_graph_edge_index[0] < num_nodes,
                                cayley_graph_edge_index[1] < num_nodes)]
            self.cayley_node_memory[num_nodes] = truncated_edge_index
            
        return self.cayley_node_memory[num_nodes].clone()

    def _get_cgp_edge_index(self, cayley_n):
        cayley_num_nodes = self._cayley_graph_size(cayley_n)
        
        if cayley_n not in self.cayley_memory:
            edge_index = get_cayley_graph(cayley_n)
            self.cayley_memory[cayley_n] = edge_index
            
        return self.cayley_memory[cayley_n].clone(), cayley_num_nodes
