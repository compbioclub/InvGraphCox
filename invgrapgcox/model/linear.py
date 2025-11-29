from lifelines import LogLogisticAFTFitter, WeibullAFTFitter, LogNormalAFTFitter
from lifelines import CoxPHFitter
import pandas as pd
import numpy as np
from sklearn import linear_model

def get_algorithm_class(algorithm_name):
    if algorithm_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]




def Weibull_AFT(df, duration_col, event_col):
    model = WeibullAFTFitter(penalizer=0.01, fit_intercept=False)
    model.fit(df, duration_col=duration_col, event_col=event_col)
    return model


def LogLogistic_AFT(df, duration_col, event_col):
    model = LogLogisticAFTFitter(penalizer=0.01, fit_intercept=False)
    model.fit(df, duration_col=duration_col, event_col=event_col)
    return model


def LogNormal_AFT(df, duration_col, event_col):
    model = LogNormalAFTFitter(penalizer=0.01, fit_intercept=False)
    model.fit(df, duration_col=duration_col, event_col=event_col)
    return model


def Cox(df, duration_col, event_col):
    model = CoxPHFitter(penalizer=0.1)
    model.fit(df, duration_col=duration_col, event_col=event_col)
    return model


def Weighted_cox(X, duration_col, event_col, W, pen, **options):
    columns = X.columns
    all_X = np.concatenate((X, W), axis=1)

    all_X = pd.DataFrame(all_X, columns=list(columns)+["Weights"])
    cph = CoxPHFitter(penalizer=pen)
    cph.fit(all_X, duration_col=duration_col, event_col=event_col, weights_col="Weights")

    return cph
