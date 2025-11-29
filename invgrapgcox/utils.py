import random
import numpy as np
import torch
from lifelines.utils import concordance_index
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold



def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def cindex(cph, selected_data, duration_col, event_col):
    c_index = concordance_index(selected_data[duration_col], -cph.predict_partial_hazard(selected_data),
                                selected_data[event_col])

    return c_index


def centre_to_train_mean(Z_train, Z_eval_list):
    mu_train = Z_train.mean(axis=0)
    Z_train_aligned = Z_train - (Z_train.mean(axis=0) - mu_train)
    Z_eval_aligned = []
    for Z in Z_eval_list:
        mu_c = Z.mean(axis=0)
        Z_eval_aligned.append(Z - (mu_c - mu_train))

    return Z_train_aligned, Z_eval_aligned


def build_preproc(cont_cols, binary_cols, onehot_cols, drop_constant=True):
    cont_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="mean")),
        ("sc", StandardScaler()),
    ])
    bin_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
    ])
    oh_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
    ])

    coltx = ColumnTransformer(
        transformers=[
            ("cont", cont_pipe, cont_cols),
            ("bin", bin_pipe, binary_cols),
            ("oh", oh_pipe, onehot_cols),
        ],
        remainder="drop",
        sparse_threshold=0.0,
        verbose_feature_names_out=True,
    )

    steps = [("cols", coltx)]
    if drop_constant:
        steps.append(("vt", VarianceThreshold(threshold=0.0)))
    preproc = Pipeline(steps)

    def get_feature_names_out():
        names = preproc.named_steps["cols"].get_feature_names_out()
        if "vt" in preproc.named_steps:
            mask = preproc.named_steps["vt"].get_support()
            names = names[mask]
        return names

    setattr(preproc, "get_feature_names_out", get_feature_names_out)
    return preproc
