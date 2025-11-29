import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATConv, GCNConv, GraphConv, GINConv, CuGraphSAGEConv, LayerNorm
from torch.nn import Dropout
from torch.nn import ReLU, Sequential, Linear, LayerNorm



class SAGEEncoder_ov(torch.nn.Module):
    def __init__(self, in_dim, hidden1=117, hidden2=42, out_dim=24, dropout1=0.8, dropout2=0.0):
        super().__init__()

        # First GATConv layer
        self.conv1 = SAGEConv(in_dim, hidden1)
        self.dropout1 = Dropout(dropout1)

        # Second GATConv layer
        self.conv2 = SAGEConv(hidden1, hidden2)
        self.dropout2 = Dropout(dropout2)

        # Separate GATConv layers for mu and logvar
        self.conv_mu = SAGEConv(hidden2, out_dim)
        self.conv_logvar = SAGEConv(hidden2, out_dim)

    def forward(self, x, edge_index):
        # First layer
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout1(x)

        # Second layer
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.dropout2(x)

        # Compute mu and logvar
        mu = self.conv_mu(x, edge_index)
        logvar = self.conv_logvar(x, edge_index)

        return mu, logvar


class SAGEEncoder_breast_new(torch.nn.Module):
    def __init__(self, in_dim, hidden1=86, hidden2=64, out_dim=20, dropout1=0.0, dropout2=0.6):
        super().__init__()

        # First GATConv layer
        self.conv1 = SAGEConv(in_dim, hidden1)
        self.dropout1 = Dropout(dropout1)

        # Second GATConv layer
        self.conv2 = SAGEConv(hidden1, hidden2)
        self.dropout2 = Dropout(dropout2)

        # Separate GATConv layers for mu and logvar
        self.conv_mu = SAGEConv(hidden2, out_dim)
        self.conv_logvar = SAGEConv(hidden2, out_dim)

    def forward(self, x, edge_index):
        # First layer
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout1(x)

        # Second layer
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.dropout2(x)

        # Compute mu and logvar
        mu = self.conv_mu(x, edge_index)
        logvar = self.conv_logvar(x, edge_index)

        return mu, logvar


class Lung_clinical_dfs(torch.nn.Module):
    def __init__(self, in_dim, hidden1=82, hidden2=59, out_dim=20, dropout1=0.0, dropout2=0.6):
        super().__init__()

        # First GATConv layer
        self.conv1 = SAGEConv(in_dim, hidden1)
        self.dropout1 = Dropout(dropout1)

        # Second GATConv layer
        self.conv2 = SAGEConv(hidden1, hidden2)
        self.dropout2 = Dropout(dropout2)

        # Separate GATConv layers for mu and logvar
        self.conv_mu = SAGEConv(hidden2, out_dim)
        self.conv_logvar = SAGEConv(hidden2, out_dim)

    def forward(self, x, edge_index):
        # First layer
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout1(x)

        # Second layer
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.dropout2(x)

        # Compute mu and logvar
        mu = self.conv_mu(x, edge_index)
        logvar = self.conv_logvar(x, edge_index)

        return mu, logvar


class Lung_clinical_os(torch.nn.Module):
    def __init__(self, in_dim, hidden1=123, hidden2=52, out_dim=16, dropout1=0.1, dropout2=0.0):
        super().__init__()

        # First GATConv layer
        self.conv1 = SAGEConv(in_dim, hidden1)
        self.dropout1 = Dropout(dropout1)

        # Second GATConv layer
        self.conv2 = SAGEConv(hidden1, hidden2)
        self.dropout2 = Dropout(dropout2)

        # Separate GATConv layers for mu and logvar
        self.conv_mu = SAGEConv(hidden2, out_dim)
        self.conv_logvar = SAGEConv(hidden2, out_dim)

    def forward(self, x, edge_index):
        # First layer
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout1(x)

        # Second layer
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.dropout2(x)

        # Compute mu and logvar
        mu = self.conv_mu(x, edge_index)
        logvar = self.conv_logvar(x, edge_index)

        return mu, logvar



class Breast_clinical_rfs(torch.nn.Module):
    def __init__(self, in_dim, hidden1=33, hidden2=32, out_dim=18, dropout1=0.0, dropout2=0.8):
        super().__init__()

        # First GATConv layer
        self.conv1 = GCNConv(in_dim, hidden1)
        self.dropout1 = Dropout(dropout1)

        # Second GATConv layer
        self.conv2 = GCNConv(hidden1, hidden2)
        self.dropout2 = Dropout(dropout2)

        # Separate GATConv layers for mu and logvar
        self.conv_mu = GCNConv(hidden2, out_dim)
        self.conv_logvar = GCNConv(hidden2, out_dim)

    def forward(self, x, edge_index):
        # First layer
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout1(x)

        # Second layer
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.dropout2(x)

        # Compute mu and logvar
        mu = self.conv_mu(x, edge_index)
        logvar = self.conv_logvar(x, edge_index)

        return mu, logvar

class Breast_clinical_os(torch.nn.Module):
    def __init__(self, in_dim, hidden1=128, hidden2=64, out_dim=32, dropout1=0.5, dropout2=0.7):
        super().__init__()

        # First GATConv layer
        self.conv1 = GCNConv(in_dim, hidden1)
        self.dropout1 = Dropout(dropout1)

        # Second GATConv layer
        self.conv2 = GCNConv(hidden1, hidden2)
        self.dropout2 = Dropout(dropout2)

        # Separate GATConv layers for mu and logvar
        self.conv_mu = GCNConv(hidden2, out_dim)
        self.conv_logvar = GCNConv(hidden2, out_dim)

    def forward(self, x, edge_index):
        # First layer
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout1(x)

        # Second layer
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.dropout2(x)

        # Compute mu and logvar
        mu = self.conv_mu(x, edge_index)
        logvar = self.conv_logvar(x, edge_index)

        return mu, logvar



# breast: hidden1=89, hidden2=34, out_dim=19, dropout1=0.2, dropout2=0.5
# melanoma: in_dim, hidden1=62, hidden2=39, out_dim=26, dropout1=0.0, dropout2=0.0
class gaeEncoder(torch.nn.Module):
    def __init__(self, in_dim, hidden1=64, hidden2=32, out_dim=20, dropout1=0.5, dropout2=0.5):
        super().__init__()

        self.conv1 = SAGEConv(in_dim, hidden1)
        self.dropout1 = Dropout(dropout1)

        self.conv2 = SAGEConv(hidden1, hidden2)
        self.dropout2 = Dropout(dropout2)

        self.conv3 = SAGEConv(hidden2, out_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout1(x)

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.dropout2(x)

        x = self.conv3(x, edge_index)

        return x

# breast: hidden1=62, hidden2=52, out_dim=25, dropout1=0.9, dropout2=0.8
# melanoma: hidden1=128, hidden2=37, out_dim=22, dropout1=0.6, dropout2=0.7
class gcnEncoder(torch.nn.Module):
    def __init__(self, in_dim, hidden1=64, hidden2=32, out_dim=20, dropout1=0.5, dropout2=0.5):
        super().__init__()

        # First GATConv layer
        self.conv1 = GCNConv(in_dim, hidden1)
        self.dropout1 = Dropout(dropout1)

        # Second GATConv layer
        self.conv2 = GCNConv(hidden1, hidden2)
        self.dropout2 = Dropout(dropout2)

        # Separate GATConv layers for mu and logvar
        self.conv_mu = GCNConv(hidden2, out_dim)
        self.conv_logvar = GCNConv(hidden2, out_dim)

    def forward(self, x, edge_index):
        # First layer
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout1(x)

        # Second layer
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.dropout2(x)

        # Compute mu and logvar
        mu = self.conv_mu(x, edge_index)
        logvar = self.conv_logvar(x, edge_index)

        return mu, logvar



# in_dim, hidden1=53, hidden2=36, out_dim=32, dropout1=0.4158103102205122, dropout2=0.6684187752077051
class SAGEEncoder_HCC(torch.nn.Module):
    def __init__(self, in_dim, hidden1=112, hidden2=20, out_dim=27, dropout1=0.5, dropout2=0.2, heads=1):
        super().__init__()

        # First GATConv layer
        self.conv1 = SAGEConv(in_dim, hidden1)
        self.dropout1 = Dropout(dropout1)

        # Second GATConv layer
        self.conv2 = SAGEConv(hidden1, hidden2)
        self.dropout2 = Dropout(dropout2)

        # Separate GATConv layers for mu and logvar
        self.conv_mu = SAGEConv(hidden2, out_dim)
        self.conv_logvar = SAGEConv(hidden2, out_dim)

    def forward(self, x, edge_index):
        # First layer
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout1(x)

        # Second layer
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.dropout2(x)

        # Compute mu and logvar
        mu = self.conv_mu(x, edge_index)
        logvar = self.conv_logvar(x, edge_index)

        return mu, logvar



# in_dim, hidden1=44, hidden2=42, out_dim=27, dropout1=0.3, dropout2=0.6
class SAGEEncoder_breast(torch.nn.Module):
    def __init__(self, in_dim, hidden1=78, hidden2=41, out_dim=23, dropout1=0.6, dropout2=0.7):
        super().__init__()

        # First GATConv layer
        self.conv1 = SAGEConv(in_dim, hidden1)
        self.dropout1 = Dropout(dropout1)

        # Second GATConv layer
        self.conv2 = SAGEConv(hidden1, hidden2)
        self.dropout2 = Dropout(dropout2)

        # Separate GATConv layers for mu and logvar
        self.conv_mu = SAGEConv(hidden2, out_dim)
        self.conv_logvar = SAGEConv(hidden2, out_dim)

    def forward(self, x, edge_index):
        # First layer
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout1(x)

        # Second layer
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.dropout2(x)

        # Compute mu and logvar
        mu = self.conv_mu(x, edge_index)
        logvar = self.conv_logvar(x, edge_index)

        return mu, logvar


class SAGEEncoder_mela(torch.nn.Module):
    def __init__(self, in_dim, hidden1=71, hidden2=16, out_dim=20, dropout1=0.0, dropout2=0.7, heads=1):
        super().__init__()

        # First GATConv layer
        self.conv1 = SAGEConv(in_dim, hidden1)
        self.dropout1 = Dropout(dropout1)

        # Second GATConv layer
        self.conv2 = SAGEConv(hidden1, hidden2)
        self.dropout2 = Dropout(dropout2)

        # Separate GATConv layers for mu and logvar
        self.conv_mu = SAGEConv(hidden2, out_dim)
        self.conv_logvar = SAGEConv(hidden2, out_dim)

    def forward(self, x, edge_index):
        # First layer
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout1(x)

        # Second layer
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.dropout2(x)

        # Compute mu and logvar
        mu = self.conv_mu(x, edge_index)
        logvar = self.conv_logvar(x, edge_index)

        return mu, logvar
