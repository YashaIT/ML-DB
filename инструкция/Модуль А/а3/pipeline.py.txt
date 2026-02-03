from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE, ADASYN


def build_preprocess_pipeline(
    feature_cols: list[str],
    balance: str | None,
    random_state: int = 42,
) -> Pipeline:
    """
    Pipeline только на предобработку + (опционально) балансировку.
    Модель будет обучаться уже в модуле 2.x/3.x.
    """
    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    pre = ColumnTransformer(
        transformers=[("num", num_pipe, feature_cols)],
        remainder="drop",
    )

    if balance is None:
        return Pipeline(steps=[("preprocess", pre)])

    if balance == "smote":
        sampler = SMOTE(random_state=random_state)
    elif balance == "adasyn":
        sampler = ADASYN(random_state=random_state)
    else:
        raise ValueError("balance must be one of: None, smote, adasyn")

    # imblearn Pipeline, чтобы sampler работал
    return ImbPipeline(steps=[("preprocess", pre), ("sampler", sampler)])
