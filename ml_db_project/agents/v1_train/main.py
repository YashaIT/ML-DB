from __future__ import annotations

import argparse
import os
import json
from datetime import datetime, timezone

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sqlalchemy import text

from shared.db import get_session
from shared.logging import setup_logger

logger = setup_logger("agent_v1_train")

OUT_DIR = "agents/v1_train/artifacts"


FEATURES = [
    "length_km",
    "mean_ele",
    "mean_slope",
    "mean_green",
    "frac_forest",
    "frac_water",
    "frac_road",
    "frac_building",
]


def load_dataset() -> pd.DataFrame:
    sess = get_session()
    q = """
    SELECT
      a.track_id,
      a.length_km,
      a.mean_ele,
      a.mean_slope,
      a.mean_green,
      a.frac_forest,
      a.frac_water,
      a.frac_road,
      a.frac_building,
      l.label
    FROM (
      SELECT
        dp.track_id,
        COUNT(*) * 0.02 AS length_km,
        AVG(dp.ele) AS mean_ele,
        AVG(dp.slope_deg) AS mean_slope,
        AVG(dp.green_index) AS mean_green,
        AVG(dp.frac_vegetation) AS frac_forest,
        AVG(dp.frac_water) AS frac_water,
        AVG(dp.frac_road) AS frac_road,
        AVG(dp.frac_building) AS frac_building
      FROM dataset_points dp
      GROUP BY dp.track_id
    ) a
    JOIN track_labels l ON l.track_id = a.track_id
    """
    df = pd.read_sql(text(q), sess.bind)
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["rf", "logreg"], default="rf")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    df = load_dataset()
    logger.info(f"dataset loaded: {df.shape}")

    if df.shape[0] < 5:
        logger.warning("Too few samples for training")
        return

    X = df[FEATURES].copy()
    y = df["label"].astype(str)

    X = X.fillna(X.median())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    if args.model == "rf":
        model = RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced",
        )
        pipe = Pipeline(
            [
                ("model", model),
            ]
        )
    else:
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=2000)),
            ]
        )

    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")

    logger.info(f"accuracy={acc:.3f} macro_f1={f1:.3f}")

    report = classification_report(y_test, y_pred, output_dict=True)

    meta = {
        "model": args.model,
        "features": FEATURES,
        "rows": int(df.shape[0]),
        "accuracy": acc,
        "macro_f1": f1,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    with open(os.path.join(OUT_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    with open(os.path.join(OUT_DIR, "classification_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    logger.info("artifacts saved")


if __name__ == "__main__":
    main()
