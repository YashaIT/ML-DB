from __future__ import annotations

import argparse
import os
import json
from datetime import datetime, timezone

import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

from sqlalchemy import text

from shared.db import get_session
from shared.logging import setup_logger

logger = setup_logger("agent_v4_predict")

OUT_DIR = "agents/v4_predict/artifacts"

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

MODEL_PATH = "agents/v4_predict/artifacts/model_rf.pkl"


def load_one_track(track_id: int) -> pd.DataFrame:
    sess = get_session()
    q = """
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
    WHERE dp.track_id = :track_id
    GROUP BY dp.track_id
    """
    df = pd.read_sql(text(q), sess.bind, params={"track_id": track_id})
    return df


def load_training_set() -> pd.DataFrame:
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
    return pd.read_sql(text(q), sess.bind)


def train_and_save_model() -> dict:
    os.makedirs(OUT_DIR, exist_ok=True)

    df = load_training_set()
    if df.empty:
        raise RuntimeError("No labeled tracks found in track_labels")

    X = df[FEATURES].fillna(df[FEATURES].median(numeric_only=True))
    y = df["label"].astype(str)

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
    )
    model.fit(X, y)

    joblib.dump(model, MODEL_PATH)

    meta = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "rows": int(df.shape[0]),
        "unique_classes": int(y.nunique()),
        "features": FEATURES,
        "model_path": MODEL_PATH,
    }
    with open(os.path.join(OUT_DIR, "train_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return meta


def predict(track_id: int) -> dict:
    if not os.path.exists(MODEL_PATH):
        train_and_save_model()

    model = joblib.load(MODEL_PATH)

    df_one = load_one_track(track_id)
    if df_one.empty:
        raise RuntimeError(f"Track {track_id} not found in dataset_points")

    X = df_one[FEATURES].fillna(0.0)
    pred = model.predict(X)[0]

    out = {
        "track_id": track_id,
        "predicted_label": str(pred),
        "features": {k: float(df_one.iloc[0][k]) for k in FEATURES},
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, f"prediction_track_{track_id}.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--track_id", type=int, required=True)
    args = ap.parse_args()

    out = predict(args.track_id)
    logger.info(f"predicted: track_id={out['track_id']} label={out['predicted_label']}")


if __name__ == "__main__":
    main()
