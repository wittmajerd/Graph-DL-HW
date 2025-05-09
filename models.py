import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GCNConv, GATConv, GINConv, NNConv,
    Set2Set, global_mean_pool, global_add_pool
)
from torch_geometric.nn.models import SchNet, DimeNet # Using the models from PyG directly

# Common practice: QM9 node features (data.x) often represent atom types.
# SchNet and DimeNet use data.z (atomic numbers) for their internal embeddings.

class GCNModel(nn.Module):
    def __init__(self, num_node_features: int, hidden_channels: int, num_targets: int,
                 num_layers: int = 3, dropout_rate: float = 0.1):
        super().__init__()
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        if num_layers == 1:
            self.convs.append(GCNConv(num_node_features, hidden_channels)) # Output directly to hidden for linear
        else:
            self.convs.append(GCNConv(num_node_features, hidden_channels))
            self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
            for _ in range(num_layers - 2):
                self.convs.append(GCNConv(hidden_channels, hidden_channels))
                self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
            self.convs.append(GCNConv(hidden_channels, hidden_channels)) # Last conv layer

        self.linear = nn.Linear(hidden_channels, num_targets)

    def forward(self, x, edge_index, batch):
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            if i < self.num_layers -1 : # No batch norm after last conv before pooling
                 if self.batch_norms: # if num_layers > 1
                    x = self.batch_norms[i](x)
            x = F.relu(x)
            if i < self.num_layers - 1: # No dropout after last conv before pooling
                x = F.dropout(x, p=self.dropout_rate, training=self.training)
        
        x = global_mean_pool(x, batch)
        x = self.linear(x)
        return x

class GATModel(nn.Module):
    def __init__(self, num_node_features: int, hidden_channels: int, num_targets: int,
                 num_layers: int = 3, heads: int = 8, dropout_rate: float = 0.1):
        super().__init__()
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        
        self.convs = nn.ModuleList()
        
        # Input layer
        self.convs.append(GATConv(num_node_features, hidden_channels, heads=heads, dropout=dropout_rate))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout_rate))
            
        # Output layer of GAT convolutions (before pooling and linear)
        # The last GATConv layer typically uses heads=1 or concatenates and then reduces dimension.
        # Here, we use heads=1 for the final GAT layer output to match hidden_channels.
        if num_layers > 1:
            self.convs.append(GATConv(hidden_channels * heads, hidden_channels, heads=1, concat=True, dropout=dropout_rate))
        # If num_layers is 1, the first layer's output (hidden_channels * heads) will be used.
        # This needs to be handled by the linear layer input size.
        
        # Adjust linear layer input based on whether the last GAT layer concatenates
        if num_layers == 1:
            linear_in_features = hidden_channels * heads
        else: # num_layers > 1, last GATConv has heads=1 and concat=True, so output is hidden_channels
            linear_in_features = hidden_channels
            
        self.linear = nn.Linear(linear_in_features, num_targets)

    def forward(self, x, edge_index, batch):
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            # ELU activation is common for GAT, and dropout is handled within GATConv
            if i < self.num_layers - 1 : # No activation after the final GAT conv if it's the one before pooling
                 x = F.elu(x)
        
        x = global_mean_pool(x, batch)
        x = self.linear(x)
        return x

class GINModel(nn.Module):
    def __init__(self, num_node_features: int, hidden_channels: int, num_targets: int,
                 num_layers: int = 3, dropout_rate: float = 0.1, train_eps: bool = False):
        super().__init__()
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate

        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        # Input layer MLP
        nn_in = nn.Sequential(
            nn.Linear(num_node_features, hidden_channels),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_channels),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_channels)
        )
        self.convs.append(GINConv(nn_in, eps=0 if not train_eps else torch.rand(1).item(), train_eps=train_eps)) # eps can be learned

        # Hidden layers MLPs
        for _ in range(num_layers - 1):
            mlp = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_channels),
                nn.Linear(hidden_channels, hidden_channels),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_channels)
            )
            self.convs.append(GINConv(mlp, eps=0 if not train_eps else torch.rand(1).item(), train_eps=train_eps))
            self.batch_norms.append(nn.BatchNorm1d(hidden_channels)) # Batch norm after GINConv's MLP application

        self.linear = nn.Linear(hidden_channels, num_targets)

    def forward(self, x, edge_index, batch):
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            if i < self.num_layers - 1: # Apply batch norm for hidden layers
                 x = self.batch_norms[i](x)
            # ReLU is typically part of the MLP in GINConv, but an extra one can be added.
            # For simplicity, relying on MLPs' internal ReLUs.
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
            
        x = global_add_pool(x, batch) # GIN often uses sum pooling
        x = self.linear(x)
        return x

class SchNetModel(nn.Module):
    def __init__(self, num_targets: int, hidden_channels: int = 128, num_filters: int = 128,
                 num_interactions: int = 6, num_gaussians: int = 50, cutoff: float = 10.0,
                 readout_pool: str = 'add'):
        super().__init__()
        # SchNet uses atomic numbers (z) as input, which are then embedded.
        # It does not directly use pre-computed node features (x) in the same way as GCN/GAT.
        self.schnet = SchNet(
            hidden_channels=hidden_channels,
            num_filters=num_filters,
            num_interactions=num_interactions,
            num_gaussians=num_gaussians,
            cutoff=cutoff,
            readout=readout_pool # 'add', 'mean', 'sum'
            # SchNet's output after readout is of size hidden_channels
        )
        self.linear = nn.Linear(hidden_channels, num_targets)

    def forward(self, z, pos, batch):
        # z: atomic numbers, pos: coordinates
        node_representation = self.schnet(z, pos, batch) # SchNet returns pooled representation
        x = self.linear(node_representation)
        return x

class MPNNModel(nn.Module): # Using NNConv
    def __init__(self, num_node_features: int, num_edge_features: int,
                 hidden_channels: int, num_targets: int,
                 num_layers: int = 3, dropout_rate: float = 0.1,
                 processing_steps_set2set: int = 3):
        super().__init__()
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate

        self.convs = nn.ModuleList()
        
        # Edge network for the first layer
        edge_nn1 = nn.Sequential(
            nn.Linear(num_edge_features, hidden_channels), # Process edge features
            nn.ReLU(),
            nn.Linear(hidden_channels, num_node_features * hidden_channels) # Output: in_channels * out_channels
        )
        self.convs.append(NNConv(num_node_features, hidden_channels, edge_nn1, aggr='mean'))

        # Edge networks for subsequent layers
        for _ in range(num_layers - 1):
            edge_nn = nn.Sequential(
                nn.Linear(num_edge_features, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels * hidden_channels) # Output: in_channels * out_channels
            )
            self.convs.append(NNConv(hidden_channels, hidden_channels, edge_nn, aggr='mean'))

        self.set2set = Set2Set(hidden_channels, processing_steps=processing_steps_set2set)
        # Set2Set doubles the output dimension
        self.linear = nn.Linear(2 * hidden_channels, num_targets)

    def forward(self, x, edge_index, edge_attr, batch):
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index, edge_attr)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
        
        x = self.set2set(x, batch)
        x = self.linear(x)
        return x

class DimeNetPlusPlusModel(nn.Module): # Using DimeNet++ from PyG
    def __init__(self, num_targets: int, hidden_channels: int = 128,
                 num_blocks: int = 4, int_emb_size: int = 64, basis_emb_size: int = 8,
                 out_emb_channels: int = 256, num_spherical: int = 7, num_radial: int = 6,
                 cutoff: float = 10.0, envelope_exponent: int = 5):
        super().__init__()
        # DimeNet uses atomic numbers (z) and positions (pos).
        # It has its own embedding for z.
        # DimeNet++ is generally recommended over the original DimeNet.
        # The 'out_channels' in DimeNet++ is the output size *before* a final MLP.
        # We will add our own linear layer for the final target prediction.
        try:
            from torch_geometric.nn.models import DimeNetPlusPlus
            self.dimenet = DimeNetPlusPlus(
                hidden_channels=hidden_channels,
                out_channels=hidden_channels, # Output of DimeNet++ core, then we add a linear layer
                num_blocks=num_blocks,
                int_emb_size=int_emb_size,
                basis_emb_size=basis_emb_size,
                out_emb_channels=out_emb_channels,
                num_spherical=num_spherical,
                num_radial=num_radial,
                cutoff=cutoff,
                envelope_exponent=envelope_exponent
            )
            self.linear = nn.Linear(hidden_channels, num_targets) # Linear layer after DimeNet++
            self.has_dimenetpp = True
        except ImportError:
            print("Warning: DimeNetPlusPlus not found. This model will not work.")
            print("Consider installing the latest PyG version or ensuring all dependencies are met.")
            self.has_dimenetpp = False
            # Fallback or placeholder if DimeNetPlusPlus is not available
            self.dimenet = None
            self.linear = nn.Linear(1, num_targets) # Placeholder


    def forward(self, z, pos, batch):
        if not self.has_dimenetpp or self.dimenet is None:
            # Return zeros or raise error if DimeNetPlusPlus is not available
            return torch.zeros((batch.max().item() + 1, self.linear.out_features), device=z.device)

        node_representation = self.dimenet(z, pos, batch)
        x = self.linear(node_representation)
        return x
