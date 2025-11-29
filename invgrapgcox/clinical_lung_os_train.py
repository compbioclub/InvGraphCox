import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from lifelines import CoxPHFitter

from torch_geometric.nn import VGAE
from utils import set_seed, cindex, centre_to_train_mean, build_preproc
from model.Encoder import Lung_clinical_os
from Preprocessing import graph_construction, graph_from_dataframe_B_gower



duration_col = 'Survival.months'
event_col = 'Survival.status'


def main():
    seed_value = 42
    n_neighbors = 13
    set_seed(seed_value)

    device = torch.device("cuda")


    paths = {
        "train": "./dataset/clinical_data/lung_cancer/train_pd.csv",
        "test0": "./dataset/clinical_data/lung_cancer/test0_pd.csv",
        "test1": "./dataset/clinical_data/lung_cancer/test1_pd.csv",
        "test2": "./dataset/clinical_data/lung_cancer/test2_pd.csv",
        "test3": "./dataset/clinical_data/lung_cancer/test3_pd.csv",
        "test4": "./dataset/clinical_data/lung_cancer/test4_pd.csv",
        "test5": "./dataset/clinical_data/lung_cancer/test5_pd.csv",
        "test6": "./dataset/clinical_data/lung_cancer/test6_pd.csv",
        "test7": "./dataset/clinical_data/lung_cancer/test7_pd.csv"
    }

    binary_cols = [
        "Sex",
        "Location",
        "阻塞性肺炎/肺不张Obst pn\nor plugging",
        "CT Kernel",
        "Multiplicity多重性",
        "Effusion胸膜积液",
        "cM",
        "pM",
        "post-op CTx术后CT",
        "post-op RTx",
    ]

    cont_cols = [
        "Age",
        "CT value \nMean",
        "CT value \nStd Dev",
        "CT value \nP.major",
        "CT V ratio",
        "pos_LN",
        "total_LN",
        "LN ratio淋巴结比例",
        "cT",
        "cN",
        "pT病例T阶段",
        "pN病理N阶段",
        "FVC用力肺活量 %PRED",
        "FEV1 %PRED",
        "FEV1一秒用力呼气量/FVC 用力肺活量(%) ",
        "CEA癌胚抗原",
    ]

    onehot_groups = {
        "Necrosis坏死": [
            "Necrosis坏死_0",
            "Necrosis坏死_1",
            "Necrosis坏死_2",
        ],
        "Underlying \nlung": [
            "Underlying \nlung_0",
            "Underlying \nlung_1",
            "Underlying \nlung_2",
            "Underlying \nlung_3",
            "Underlying \nlung_4",
        ],
        "Bronchoscopy 支气管镜检": [
            "Bronchoscopy 支气管镜检_1",
            "Bronchoscopy 支气管镜检_2",
            "Bronchoscopy 支气管镜检_3",
            "Bronchoscopy 支气管镜检_4",
        ],
        "Differentiation分化": [
            "Differentiation分化_1",
            "Differentiation分化_2",
            "Differentiation分化_3",
            "Differentiation分化_4",
        ],
        "Smoking state": [
            "Smoking state_0",
            "Smoking state_1",
            "Smoking state_2",
            "Smoking state_3",
        ],
        "op type手术类型": [
            "op type手术类型_1",
            "op type手术类型_2",
            "op type手术类型_3",
            "op type手术类型_4",
            "op type手术类型_5",
        ],
    }
    onehot_cols = sum(onehot_groups.values(), [])

    dfs = {k: pd.read_csv(p, index_col=0) for k, p in paths.items()}


    drop_cols = ["Reccur.status", "Reccur.months"]
    dfs = {k: df.drop(columns=drop_cols) for k, df in dfs.items()}


    metadata = {k: df.iloc[:, -2:] for k, df in dfs.items()}
    feature_cols = dfs["train"].columns[:-2]

    X_pd = {k: df for k, df in dfs.items()}  # full DF including metadata


    exclude_cols = ["Unnamed: 0", "Survival.status", "Survival.months", "Reccur.status", "Reccur.months"]
    feature_cols = [c for c in feature_cols if c not in set(exclude_cols)]

    preproc = build_preproc(cont_cols, binary_cols, onehot_cols, drop_constant=True)

    x_train = preproc.fit_transform(X_pd["train"][feature_cols])

    x_test = {
        k: preproc.transform(X_pd[k][feature_cols])
        for k in dfs.keys() if k != "train"
    }


    train_graph = graph_from_dataframe_B_gower(
        df_features=X_pd["train"][feature_cols],
        X_node=x_train,
        n_neighbors=n_neighbors,
    )

    graphs = {"train": train_graph}
    for k in dfs.keys():
        if k == "train":
            continue
        graphs[k] = graph_from_dataframe_B_gower(
            df_features=X_pd[k][feature_cols],
            X_node=x_test[k],
            n_neighbors=n_neighbors,
        )

    for k in graphs:
        graphs[k] = graphs[k].to(device)


    in_dim = train_graph.x.shape[1]
    encoder = Lung_clinical_os(in_dim=in_dim).to(device)
    gae = VGAE(encoder).to(device)

    optimizer = optim.Adam(gae.parameters(), lr=0.000961477554774501, weight_decay=0.0004897874644841262)

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
        emb_tensors = {
            k: gae.encode(g.x, g.edge_index)
            for k, g in graphs.items()
        }

    emb_np = {k: v.cpu().numpy() for k, v in emb_tensors.items()}
    emb_dim = emb_np["train"].shape[1]
    emb_cols = [f"emb_{i}" for i in range(emb_dim)]

    emb_df = {k: pd.DataFrame(v, columns=emb_cols) for k, v in emb_np.items()}


    final_df = {}
    for k in dfs.keys():
        meta = metadata[k].reset_index(drop=True)
        emb_part = emb_df[k].reset_index(drop=True)
        final_df[k] = pd.concat([meta, emb_part], axis=1)

    X_train_df = final_df["train"]


    eval_splits = ["test0", "test1", "test4", "test5", "test6", "test7"]


    cph = CoxPHFitter(penalizer=0.01931179021907528)
    cph.fit(X_train_df, duration_col=duration_col, event_col=event_col)

    c_index_list = []
    for name in eval_splits:
        c_idx = cindex(cph, final_df[name], duration_col, event_col)
        c_index_list.append(c_idx)
        print(f"{name}: {c_idx}")

    print("==================")
    mean_acc = np.mean(c_index_list)
    std_acc = np.std(c_index_list)
    worst_acc = min(c_index_list)
    print(mean_acc)
    print(std_acc)
    print(worst_acc)


if __name__ == "__main__":
    main()
