import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GINConv, NNConv, Set2Set, global_mean_pool, global_add_pool
from torch_geometric.nn import SchNet, DimeNet, GCN, GAT, GIN

# 1. GCN - Simple and effective
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.linear = torch.nn.Linear(hidden_channels, out_channels)
    
    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.1, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        
        # Graph-level readout
        x = global_mean_pool(x, batch)
        
        # Prediction layer
        x = self.linear(x)
        return x
    

# 2. GAT - Attention-based approach
class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=8):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads)
        self.conv2 = GATConv(hidden_channels*heads, hidden_channels, heads=1)
        self.linear = torch.nn.Linear(hidden_channels, out_channels)
    
    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        
        # Graph-level readout
        x = global_mean_pool(x, batch)
        
        # Prediction layer
        x = self.linear(x)
        return x
    
class GATModel(torch.nn.Module):
    def __init__(self, node_features, hidden_channels, num_targets):
        super(GATModel, self).__init__()
        self.conv1 = GATConv(node_features, hidden_channels)
        self.conv2 = GATConv(hidden_channels, hidden_channels)
        self.conv3 = GATConv(hidden_channels, hidden_channels)
        self.lin = torch.nn.Linear(hidden_channels, num_targets)
        
    def forward(self, x, edge_index, batch):
        # Graph convolutions
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        
        # Graph-level readout
        x = global_mean_pool(x, batch)
        
        # Prediction
        x = self.lin(x)
        
        return x
    

class GINModel(torch.nn.Module):
    def __init__(self, node_features, hidden_channels, num_targets):
        super(GINModel, self).__init__()
        
        nn1 = torch.nn.Sequential(
            torch.nn.Linear(node_features, hidden_channels),
            torch.nn.BatchNorm1d(hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, hidden_channels)
        )
        self.conv1 = GINConv(nn1)
        
        nn2 = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels),
            torch.nn.BatchNorm1d(hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, hidden_channels)
        )
        self.conv2 = GINConv(nn2)
        
        self.lin = torch.nn.Linear(hidden_channels, num_targets)
        
    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        
        x = global_add_pool(x, batch)
        x = self.lin(x)
        
        return x
    

class SchNetModel(torch.nn.Module):
    def __init__(self, hidden_channels=128, num_targets=14):
        super(SchNetModel, self).__init__()
        
        self.schnet = SchNet(
            hidden_channels=hidden_channels,
            num_filters=128,
            num_interactions=6,
            num_gaussians=50,
            cutoff=10.0
        )
        
        self.lin = torch.nn.Linear(hidden_channels, num_targets)
        
    def forward(self, z, pos, batch):
        x = self.schnet(z, pos, batch)
        x = self.lin(x)
        return x
    
# huber loss
# 3. SchNet - For 3D molecular structures
# This can be used directly
schnet = SchNet(
    hidden_channels=128,
    num_filters=128,
    num_interactions=6,
    num_gaussians=50,
    cutoff=10.0,
    readout='add',
    dipole=False,
    mean=None, std=None,
    atomref=None
)

# Wrapper to adapt SchNet for your task
class SchNetWrapper(torch.nn.Module):
    def __init__(self, num_targets=14):
        super().__init__()
        self.schnet = SchNet(
            hidden_channels=128,
            num_filters=128,
            num_interactions=6,
            num_gaussians=50,
            cutoff=10.0
        )
        self.linear = torch.nn.Linear(128, num_targets)
    
    def forward(self, z, pos, batch):
        x = self.schnet(z, pos, batch)
        x = self.linear(x)
        return x
    

class MPNNModel(torch.nn.Module):
    def __init__(self, node_features, edge_features, hidden_channels, num_targets):
        super(MPNNModel, self).__init__()
        
        # Edge network
        self.edge_network = torch.nn.Sequential(
            torch.nn.Linear(edge_features, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, node_features * hidden_channels)
        )
        
        # Message passing layers
        self.conv = NNConv(node_features, hidden_channels, self.edge_network, aggr='mean')
        
        # Readout
        self.set2set = Set2Set(hidden_channels, processing_steps=3)
        
        # Output layer
        self.lin = torch.nn.Linear(2 * hidden_channels, num_targets)
        
    def forward(self, x, edge_index, edge_attr, batch = 16):
        x = F.relu(self.conv(x, edge_index, edge_attr))
        x = F.relu(self.conv(x, edge_index, edge_attr))
        x = F.relu(self.conv(x, edge_index, edge_attr))
        
        x = self.set2set(x, batch)
        x = self.lin(x)
        
        return x
    

class DimeNetModel(torch.nn.Module):
    def __init__(self, hidden_channels=128, num_targets=14):
        super(DimeNetModel, self).__init__()
        
        self.dimenet = DimeNet(
            hidden_channels=hidden_channels,
            out_channels=num_targets,
            num_blocks=4,
            num_bilinear=8,
            num_spherical=7,
            num_radial=6,
            cutoff=10.0
        )
        
    def forward(self, z, pos, batch):
        return self.dimenet(z, pos, batch)


# 4. DimeNet - Directional message passing for 3D structures
class DimeNetWrapper(torch.nn.Module):
    def __init__(self, num_targets=14):
        super().__init__()
        self.dimenet = DimeNet(
            hidden_channels=128,
            out_channels=num_targets,
            num_blocks=4,
            num_bilinear=8,
            num_spherical=7,
            num_radial=6,
            cutoff=10.0
        )
    
    def forward(self, z, pos, batch):
        return self.dimenet(z, pos, batch)