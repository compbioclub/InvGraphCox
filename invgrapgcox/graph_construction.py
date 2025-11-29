import numpy as np
from sklearn.neighbors import NearestNeighbors
import torch
from torch_geometric.data import Data
from torch_geometric.transforms import GCNNorm
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import random
import gower


def graph_construction(x_train_np, n_neighbors=5):
    # KNN
    neigh = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto').fit(x_train_np)
    distances, indices = neigh.kneighbors(x_train_np)

    # Adjacency Matrix: weights
    sigma = np.median(distances)
    weights = np.exp(- (distances ** 2) / (2 * sigma ** 2))

    N = x_train_np.shape[0]
    adjacency_matrix = np.zeros((N, N), dtype=float)

    for i in range(N):
        for nbr_idx, dist_weight in zip(indices[i], weights[i]):
            adjacency_matrix[i, nbr_idx] = dist_weight
            adjacency_matrix[nbr_idx, i] = dist_weight

    row, col = adjacency_matrix.nonzero()
    edge_index = torch.tensor([row, col], dtype=torch.long)
    edge_weight = torch.tensor(adjacency_matrix[row, col], dtype=torch.float)

    x = torch.tensor(x_train_np, dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_weight)

    norm_transform = GCNNorm(add_self_loops=False)
    data = norm_transform(data)

    return data



def graph_from_numeric_matrix(X, n_neighbors=2, sigma=None, add_self_loops=False):
    # Small example: choose k=2 for tiny train set
    n_neighbors = min(n_neighbors, X.shape[0] - 1)

    nbrs = NearestNeighbors(n_neighbors=n_neighbors+1, metric="euclidean").fit(X)
    distances, indices = nbrs.kneighbors(X)

    # first neighbor is the node itself -> drop self
    distances, indices = distances[:, 1:], indices[:, 1:]

    # pick sigma once from train (median of all non-self distances)
    if sigma is None:
        sigma = np.median(distances) if distances.size > 0 else 1.0

    # RBF edge weights
    weights = np.exp(-(distances**2) / (2 * (sigma**2 + 1e-12)))

    N, k = X.shape[0], indices.shape[1]
    row = np.repeat(np.arange(N), k)   # source nodes
    col = indices.reshape(-1)          # target nodes

    edge_index = torch.tensor(np.vstack([row, col]), dtype=torch.long)
    edge_attr  = torch.tensor(weights.reshape(-1), dtype=torch.float32)
    x          = torch.tensor(X, dtype=torch.float32)

    if add_self_loops:
        self_idx = np.arange(N)
        self_edges = torch.tensor(np.vstack([self_idx, self_idx]), dtype=torch.long)
        edge_index = torch.cat([edge_index, self_edges], dim=1)
        edge_attr  = torch.cat([edge_attr, torch.ones(N, dtype=torch.float32)], dim=0)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr), sigma


def rbf_from_knn(distances, indices, sigma=None):
    # distances: (N,k), indices: (N,k), first neighbor already stripped
    if sigma is None:
        sigma = np.median(distances) if distances.size > 0 else 1.0
    weights = np.exp(-(distances**2) / (2.0 * (sigma**2 + 1e-12)))

    N, k = distances.shape
    row = np.repeat(np.arange(N), k)      # sources i
    col = indices.reshape(-1)             # targets j
    edge_index = torch.tensor(np.vstack([row, col]), dtype=torch.long)
    edge_attr  = torch.tensor(weights.reshape(-1), dtype=torch.float32)
    return edge_index, edge_attr, sigma


def finalize_data(X_full, edge_index, edge_attr, add_self_loops=False, do_norm=True):
    x = torch.tensor(X_full, dtype=torch.float32)
    if add_self_loops:
        self_idx = torch.arange(x.size(0), dtype=torch.long)
        self_edges = torch.stack([self_idx, self_idx], dim=0)
        edge_index = torch.cat([edge_index, self_edges], dim=1)
        edge_attr  = torch.cat([edge_attr, torch.ones(x.size(0), dtype=torch.float32)], dim=0)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    if do_norm:
        data = GCNNorm(add_self_loops=False)(data)
    return data


def graph_from_dataframe_B_gower(df_features, X_node=None, n_neighbors=10, add_self_loops=False, sigma=None):
    N = df_features.shape[0]
    n_neighbors = min(n_neighbors, N - 1)

    # Gower distance in [0,1]
    D = gower.gower_matrix(df_features)  # (N,N)

    # For each i, take k nearest (smallest distances excluding self)
    row_idx, col_idx, w_list = [], [], []
    for i in range(N):
        order = np.argsort(D[i])
        nbrs  = order[1:n_neighbors+1]
        sims  = 1.0 - D[i, nbrs]  # similarity
        row_idx.extend([i]*len(nbrs))
        col_idx.extend(nbrs.tolist())
        w_list.extend(sims.tolist())

    edge_index = torch.tensor([row_idx, col_idx], dtype=torch.long)
    edge_attr  = torch.tensor(w_list, dtype=torch.float32)

    # Node features for the GNN
    if X_node is None:
        X_node = df_features.to_numpy()

    data = finalize_data(X_node, edge_index, edge_attr, add_self_loops=add_self_loops)
    return data
