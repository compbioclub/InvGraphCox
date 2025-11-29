import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from lifelines import CoxPHFitter

from torch_geometric.nn import VGAE
from utils import set_seed, cindex, centre_to_train_mean, build_preproc
from model.Encoder import Breast_clinical_os
from Preprocessing import graph_construction, graph_from_numeric_matrix



duration_col = 'Survival.months'
event_col = 'Survival.status'


def main():
    seed_value = 42
    n_neighbors = 15
    set_seed(seed_value)

    device = torch.device("cuda")


    paths = {
        "train": "./dataset/clinical_data/breast_cancer/breast_train_survival.csv",
        "test0": "./dataset/clinical_data/breast_cancer/breast_test0_survival.csv",
        "test1": "./dataset/clinical_data/breast_cancer/breast_test1_survival.csv",
        "test2": "./dataset/clinical_data/breast_cancer/breast_test2_survival.csv"
    }

    cont_cols = [
        "Age at Diagnosis",
        "Lymph nodes examined positive",
        "Mutation Count",
        "Nottingham prognostic index",
        "Tumor Size",
        "Tumor Stage",
    ]

    binary_cols = [
        "Type of Breast Surgery_Breast Conserving",
        "Chemotherapy_No",
        "ER status measured by IHC_Positve",
        "ER Status_Positive",
        "HER2 Status_Negative",
        "Hormone Therapy_Yes",
        "Inferred Menopausal State_Pre",
        "Primary Tumor Laterality_Right",
        "PR Status_Negative",
        "Radio Therapy_No",
    ]

    onehot_groups = {
        "Cellularity": [
            "Cellularity_High",
            "Cellularity_Low",
            "Cellularity_Moderate",
        ],
        "Neoplasm Histologic Grade": [
            "Neoplasm Histologic Grade_1.0",
            "Neoplasm Histologic Grade_2.0",
            "Neoplasm Histologic Grade_3.0",
        ],
        "HER2 status measured by SNP6": [
            "HER2 status measured by SNP6_Gain",
            "HER2 status measured by SNP6_Loss",
            "HER2 status measured by SNP6_Neutral",
        ],
        "3-Gene classifier subtype": [
            "3-Gene classifier subtype_ER+/HER2- High Prolif",
            "3-Gene classifier subtype_ER+/HER2- Low Prolif",
            "3-Gene classifier subtype_ER-/HER2-",
            "3-Gene classifier subtype_HER2+",
        ],
    }
    onehot_cols = sum(onehot_groups.values(), [])


    dfs = {k: pd.read_csv(v, index_col=0) for k, v in paths.items()}

    drop_cols = ["Recurr.months", "Recurr.status", "Cohort"]
    dfs = {k: df.drop(columns=drop_cols) for k, df in dfs.items()}

    # First 2 columns = metadata (duration + event), rest = gene expression
    top_cols = dfs["train"].columns[2:]
    metadata = {k: df.iloc[:, :2] for k, df in dfs.items()}

    X_pd = {
        k: pd.concat([metadata[k], df[top_cols]], axis=1)
        for k, df in dfs.items()
    }

    X_train_pd = X_pd["train"]
    X_test0_pd = X_pd["test0"]
    X_test1_pd = X_pd["test1"]
    X_test2_pd = X_pd["test2"]


    exclude_cols = [
        "Unnamed: 0",
        "Survival.status",
        "Survival.months",
        "Reccur.status",
        "Reccur.months",
    ]
    feature_cols = [c for c in X_train_pd.columns if c not in set(exclude_cols)]

    preproc = build_preproc(cont_cols, binary_cols, onehot_cols, drop_constant=True)

    x_train = preproc.fit_transform(X_train_pd[feature_cols])
    x_test0 = preproc.transform(X_test0_pd[feature_cols])
    x_test1 = preproc.transform(X_test1_pd[feature_cols])
    x_test2 = preproc.transform(X_test2_pd[feature_cols])


    train_graph, sigma = graph_from_numeric_matrix(
        x_train, n_neighbors=n_neighbors, sigma=None, add_self_loops=False
    )
    test0_graph, _ = graph_from_numeric_matrix(
        x_test0, n_neighbors=n_neighbors, sigma=sigma, add_self_loops=False
    )
    test1_graph, _ = graph_from_numeric_matrix(
        x_test1, n_neighbors=n_neighbors, sigma=sigma, add_self_loops=False
    )
    test2_graph, _ = graph_from_numeric_matrix(
        x_test2, n_neighbors=n_neighbors, sigma=sigma, add_self_loops=False
    )

    train_graph = train_graph.to(device)
    test0_graph = test0_graph.to(device)
    test1_graph = test1_graph.to(device)
    test2_graph = test2_graph.to(device)


    in_dim = train_graph.x.shape[1]
    encoder = Breast_clinical_os(in_dim=in_dim).to(device)
    gae = VGAE(encoder).to(device)

    optimizer = optim.Adam(gae.parameters(), lr=2.8717308063831805e-05, weight_decay=1.43416488942446e-07)

    gae.train()
    encoder.train()
    for epoch in range(210):
        optimizer.zero_grad()
        z = gae.encode(train_graph.x, train_graph.edge_index)
        loss = gae.recon_loss(z, train_graph.edge_index)
        loss = loss + (1 / train_graph.num_nodes) * gae.kl_loss().to(device)
        loss.backward()
        optimizer.step()


    gae.eval()
    encoder.eval()
    with torch.no_grad():
        emb = {
            "train": gae.encode(train_graph.x, train_graph.edge_index),
            "test0": gae.encode(test0_graph.x, test0_graph.edge_index),
            "test1": gae.encode(test1_graph.x, test1_graph.edge_index),
            "test2": gae.encode(test2_graph.x, test2_graph.edge_index),
        }

    emb_np = {k: v.cpu().numpy() for k, v in emb.items()}
    emb_dim = emb_np["train"].shape[1]
    emb_cols = [f"emb_{i}" for i in range(emb_dim)]

    emb_df = {k: pd.DataFrame(v, columns=emb_cols) for k, v in emb_np.items()}


    final_df = {}
    for split in ["train", "test0", "test1", "test2"]:
        meta = metadata[split].reset_index(drop=True)
        emb_part = emb_df[split].reset_index(drop=True)
        final_df[split] = pd.concat([meta, emb_part], axis=1)

    X_train_df = final_df["train"]
    X_test0_df = final_df["test0"]
    X_test1_df = final_df["test1"]
    X_test2_df = final_df["test2"]


    cph = CoxPHFitter(penalizer=0.0001531227607272157)
    cph.fit(X_train_df, duration_col=duration_col, event_col=event_col)

    c_index_list = []

    c_index1 = cindex(cph, X_test1_df, duration_col, event_col)
    print(f"test1 {c_index1}")
    c_index_list.append(c_index1)

    c_index2 = cindex(cph, X_test2_df, duration_col, event_col)
    print(f"test2 {c_index2}")
    c_index_list.append(c_index2)

    mean_acc = np.mean(c_index_list)
    std_acc = np.std(c_index_list)
    worst_acc = min(c_index_list)

    print("mean_acc", mean_acc)
    print("std_acc", std_acc)
    print("worst_acc", worst_acc)


if __name__ == "__main__":
    main()
