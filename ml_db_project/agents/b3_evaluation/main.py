from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from sqlalchemy import text

from shared.db import get_session
from shared.logging import setup_logger

logger = setup_logger("agent_b3_evaluation")


OUT_DIR = "agents/b3_evaluation/artifacts"


FEATURES = [
    "length_km",
    "ele_gain_m",
    "mean_slope",
    "mean_green",
    "frac_forest",
    "frac_water",
    "frac_road",
    "frac_building",
]


def load_data() -> pd.DataFrame:
    sess = get_session()

    q = """
    SELECT
      a.track_id,
      a.length_km,
      a.ele_gain_m,
      a.mean_slope,
      a.mean_green,
      a.frac_forest,
      a.frac_water,
      a.frac_road,
      a.frac_building,
      l.label,
      l.method
    FROM (
      SELECT
        dp.track_id,
        COUNT(*) * 0.02 as length_km,
        0.0 as ele_gain_m,
        AVG(dp.slope_deg) as mean_slope,
        AVG(dp.green_index) as mean_green,
        AVG(dp.frac_water) as frac_water,
        AVG(dp.frac_vegetation) as frac_forest,
        AVG(dp.frac_road) as frac_road,
        AVG(dp.frac_building) as frac_building
      FROM dataset_points dp
      GROUP BY dp.track_id
    ) a
    LEFT JOIN track_labels l ON l.track_id = a.track_id
    """
    df = pd.read_sql(text(q), sess.bind)
    return df


def evaluate(df: pd.DataFrame) -> dict:
    out: dict = {}

    if df.empty or df.shape[0] < 3:
        return {"status": "too_few_tracks"}

    X = df[FEATURES].copy()
    for c in FEATURES:
        X[c] = X[c].astype(float).fillna(float(X[c].median()))

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X.values)

    # кластеризацию не пересчитываем — используем label как прокси
    # silhouette считаем по псевдо-кластерам (label)
    labels = df["label"].astype("category").cat.codes.values

    if len(np.unique(labels)) > 1:
        sil = float(silhouette_score(Xs, labels))
        out["silhouette"] = sil
    else:
        out["silhouette"] = None

    # согласованность меток
    vc = df["label"].value_counts(normalize=True).to_dict()
    out["label_distribution"] = vc

    # конфликты cluster/manual
    if "method" in df.columns:
        total = int(df.shape[0])
        manual = df[df["method"] == "manual"]
        auto = df[df["method"] == "cluster"]

        out["n_tracks"] = total
        out["manual_labeled"] = int(manual.shape[0])
        out["auto_labeled"] = int(auto.shape[0])
    else:
        out["n_tracks"] = int(df.shape[0])

    return out


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    df = load_data()
    logger.info(f"tracks loaded: {len(df)}")

    metrics = evaluate(df)
    metrics["created_at"] = datetime.now(timezone.utc).isoformat()

    # сохраняем таблицу
    csv_path = os.path.join(OUT_DIR, "tracks_with_labels.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8")

    # сохраняем json с метриками
    json_path = os.path.join(OUT_DIR, "labeling_evaluation.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    logger.info(f"saved: {csv_path}")
    logger.info(f"saved: {json_path}")


if __name__ == "__main__":
    main()
