from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier


def rf_importance(X: pd.DataFrame, y: pd.Series, random_state: int = 42) -> pd.DataFrame:
    model = RandomForestClassifier(
        n_estimators=300,
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced",
    )
    model.fit(X, y)
    imp = pd.DataFrame(
        {"feature": X.columns, "importance": model.feature_importances_.astype(float)}
    ).sort_values("importance", ascending=False)
    return imp


def mi_importance(X: pd.DataFrame, y: pd.Series, random_state: int = 42) -> pd.DataFrame:
    mi = mutual_info_classif(X, y, random_state=random_state)
    imp = pd.DataFrame({"feature": X.columns, "importance": mi.astype(float)}).sort_values(
        "importance", ascending=False
    )
    return imp


def pick_top_features(imp: pd.DataFrame, top_k: int) -> list[str]:
    top_k = max(1, int(top_k))
    return imp.head(top_k)["feature"].tolist()
