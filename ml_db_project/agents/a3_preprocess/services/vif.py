from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor


def compute_vif(X: pd.DataFrame) -> pd.DataFrame:
    """
    VIF корректно считается на числовых признаках без NaN.
    """
    X_ = X.copy()
    X_ = X_.replace([np.inf, -np.inf], np.nan).dropna(axis=0)

    if X_.shape[0] < 5 or X_.shape[1] == 0:
        return pd.DataFrame(columns=["feature", "vif"])

    vals = X_.values
    vifs = []
    for i, col in enumerate(X_.columns):
        vifs.append((col, float(variance_inflation_factor(vals, i))))

    out = pd.DataFrame(vifs, columns=["feature", "vif"]).sort_values("vif", ascending=False)
    return out
