import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from lifelines import CoxPHFitter
from torch_geometric.nn import VGAE
from utils import set_seed, cindex, centre_to_train_mean

from model.Encoder import SAGEEncoder_HCC
from Preprocessing import graph_construction



duration_col = 'Survival.months'
event_col = 'Survival.status'




def main():
    seed_value = 42
    top_k = 20
    n_neighbors = 10
    set_seed(seed_value)

    device = torch.device("cuda")


    paths = {
        "train": "./dataset/HCC_cancer/train_median.csv",
        "test0": "./dataset/HCC_cancer/test0_median.csv",
        "test1": "./dataset/HCC_cancer/test1_median.csv",
        "test2": "./dataset/HCC_cancer/test2_median.csv",
        "test3": "./dataset/HCC_cancer/test3_median.csv",
    }


    dfs = {k: pd.read_csv(v, index_col=0) for k, v in paths.items()}

    # First 2 columns = metadata (duration + event), rest = gene expression
    top_cols = dfs["train"].columns[2:]
    metadata = {k: df.iloc[:, :2] for k, df in dfs.items()}
    X_np = {k: df[top_cols].to_numpy() for k, df in dfs.items()}


    # Graph Construction
    graphs = {
        k: graph_construction(X_np[k], n_neighbors=n_neighbors).to(device)
        for k in dfs.keys()
    }


    # Graph Representation Learning
    in_dim = X_np["train"].shape[1]
    encoder = SAGEEncoder_HCC(in_dim).to(device)
    gae = VGAE(encoder).to(device)
    optimizer = optim.Adam(gae.parameters(), lr=0.01, weight_decay=1e-9)

    print("====================Graph AutoEncoder Training====================")
    gae.train()
    for epoch in range(210):
        optimizer.zero_grad()
        z = gae.encode(graphs["train"].x, graphs["train"].edge_index)

        loss = gae.recon_loss(z, graphs["train"].edge_index)
        loss = loss + (1 / graphs["train"].num_nodes) * gae.kl_loss().to(device)

        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")


    # Extract Embeddings
    gae.eval()
    with torch.no_grad():
        emb_np = {
            k: gae.encode(g.x, g.edge_index).cpu().numpy()
            for k, g in graphs.items()
        }

    emb_dim = emb_np["train"].shape[1]
    emb_cols = [f"emb_{i}" for i in range(emb_dim)]
    emb_df = {k: pd.DataFrame(v, columns=emb_cols) for k, v in emb_np.items()}

    # Align testing set embeddings to training mean
    emb_df["train"], [emb_df["test1"], emb_df["test2"], emb_df["test3"]] = centre_to_train_mean(emb_df["train"], [emb_df["test1"], emb_df["test2"], emb_df["test3"]])


    # Build final DataFrames
    full_df = {
        k: pd.concat(
            [metadata[k].reset_index(drop=True), emb_df[k].reset_index(drop=True)],
            axis=1,
        )
        for k in dfs.keys()
    }


    # Cox model & Feature Selection
    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(full_df["train"], duration_col=duration_col, event_col=event_col)

    summary = cph.summary
    top_features = summary["p"].nsmallest(top_k).index
    print("\n".join(map(str, top_features)))

    # Keep metadata + top 15 embedding features
    train_cols = list(full_df["train"].columns[:2]) + list(top_features)
    selected_train = full_df["train"][train_cols]

    cph2 = CoxPHFitter(penalizer=0.1)
    cph2.fit(selected_train, duration_col=duration_col, event_col=event_col)


    # Evaluation
    c_index_list = []
    for name in ["test1", "test2", "test3"]:
        print(name)
        c_index = cindex(cph2, full_df[name], duration_col, event_col)
        print(c_index)
        c_index_list.append(c_index)

    mean_acc = np.mean(c_index_list)
    std_acc = np.std(c_index_list)
    worst_acc = min(c_index_list)

    print("mean_acc", mean_acc)
    print("std_acc", std_acc)
    print("worst_acc", worst_acc)


if __name__ == "__main__":
    main()
