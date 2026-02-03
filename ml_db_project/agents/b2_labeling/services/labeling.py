from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from sqlalchemy import text

from shared.db import engine, get_session
from shared.models import Base
from shared.b2_models import TrackLabel


@dataclass
class LabelingConfig:
    k: int = 3
    active_n: int = 10
    out_dir: str = "agents/b2_labeling/artifacts"


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


def ensure_db() -> None:
    Base.metadata.create_all(bind=engine)


def load_track_aggregates() -> pd.DataFrame:
    """
    Агрегаты по трекам прямо из dataset_points (без зависимостей от дашборда).
    """
    sess = get_session()

    q = """
    SELECT
      track_id,
      COUNT(*) as n_points,
      AVG(slope_deg) as mean_slope,
      AVG(green_index) as mean_green,
      AVG(frac_water) as mean_frac_water,
      AVG(frac_vegetation) as mean_frac_vegetation,
      AVG(frac_road) as mean_frac_road,
      AVG(frac_building) as mean_frac_building,
      AVG(ele) as mean_ele,
      MIN(ts) as ts_min,
      MAX(ts) as ts_max
    FROM dataset_points
    GROUP BY track_id
    """
    df = pd.read_sql(text(q), sess.bind)

    # length_km + ele_gain_m посчитаем грубо по track_points (если нужно) — тут сделаем минимально:
    # length_km = n_points * 0.02 как суррогат, чтобы кластеризация работала даже без времени/расстояний.
    # Если хочешь “по-взрослому” — можно посчитать haversine по точкам, но это дольше.
    df["length_km"] = (df["n_points"].astype(float) * 0.02).clip(lower=0.1)

    # ele_gain_m — суррогат через дисперсию высоты
    df["ele_gain_m"] = (df["mean_ele"].astype(float).fillna(0.0) * 0.0)  # оставим 0, если ele плохой

    # доли по land_type из dataset_points
    q2 = """
    SELECT track_id, land_type, COUNT(*) as cnt
    FROM dataset_points
    GROUP BY track_id, land_type
    """
    lt = pd.read_sql(text(q2), sess.bind)
    if not lt.empty:
        total = lt.groupby("track_id")["cnt"].sum().rename("total").reset_index()
        lt = lt.merge(total, on="track_id", how="left")
        lt["share"] = lt["cnt"] / lt["total"]

        piv = lt.pivot_table(index="track_id", columns="land_type", values="share", fill_value=0.0).reset_index()
        # нормализуем имена
        for col in ["forest", "water", "road_area", "urban"]:
            if col not in piv.columns:
                piv[col] = 0.0
        piv = piv.rename(
            columns={
                "forest": "frac_forest",
                "water": "frac_water",
                "road_area": "frac_road",
                "urban": "frac_building",
            }
        )
        df = df.merge(piv[["track_id", "frac_forest", "frac_water", "frac_road", "frac_building"]], on="track_id", how="left")

    # если нет land_type — заполним нулями
    for c in ["frac_forest", "frac_water", "frac_road", "frac_building"]:
        if c not in df.columns:
            df[c] = 0.0
        df[c] = df[c].fillna(0.0).astype(float)

    # mean_green/mean_slope
    df["mean_green"] = df["mean_green"].astype(float).fillna(0.0)
    df["mean_slope"] = df["mean_slope"].astype(float).fillna(0.0)

    return df


def cluster_and_label(df: pd.DataFrame, cfg: LabelingConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Возвращает:
      - df_tracks с cluster, pseudo_label, confidence
      - df_active с “спорными” треками для ручной проверки
    """
    if df.empty:
        return df, pd.DataFrame()

    X = df[FEATURES].copy()
    for c in FEATURES:
        if X[c].isna().all():
            X[c] = 0.0
        else:
            X[c] = X[c].fillna(float(X[c].median())).astype(float)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X.values)

    k = max(2, min(int(cfg.k), int(len(df))))
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    clusters = km.fit_predict(Xs)

    # расстояния до центров для "uncertainty"
    dists = km.transform(Xs)  # shape: (n, k)
    order = np.argsort(dists, axis=1)
    best = dists[np.arange(len(df)), order[:, 0]]
    second = dists[np.arange(len(df)), order[:, 1]]
    margin = second - best  # чем меньше, тем менее уверенно

    out = df.copy()
    out["cluster"] = clusters
    out["confidence"] = (margin / (margin.max() + 1e-9)).clip(0.0, 1.0)

    # псевдо-метки по правилам (простые, но понятные)
    def pseudo_label(row: pd.Series) -> str:
        if row["frac_forest"] > 0.5 and row["mean_green"] > 0.2:
            return "forest_route"
        if row["frac_water"] > 0.3:
            return "water_route"
        if row["frac_building"] > 0.3 or row["frac_road"] > 0.3:
            return "urban_route"
        return "mixed_route"

    out["label"] = out.apply(pseudo_label, axis=1)
    out["method"] = "cluster"

    # active learning: берём самые маленькие margin
    out["uncertainty"] = 1.0 - out["confidence"]
    df_active = out.sort_values("uncertainty", ascending=False).head(int(cfg.active_n)).copy()
    df_active["review_label"] = df_active["label"]  # колонка для правки человеком

    return out, df_active


def save_labels_to_db(df_labels: pd.DataFrame) -> int:
    """
    Сохраняет метки в таблицу track_labels (перезаписывает по track_id).
    """
    if df_labels.empty:
        return 0

    ensure_db()
    sess = get_session()

    # удаляем старые метки этих треков
    ids = df_labels["track_id"].astype(int).tolist()

# SQLite-safe: удаляем по одному (на объёмах этого задания достаточно)
    for tid in ids:
        sess.execute(text("DELETE FROM track_labels WHERE track_id = :tid"), {"tid": int(tid)})

        sess.commit()

    inserted = 0
    for _, r in df_labels.iterrows():
        sess.add(
            TrackLabel(
                track_id=int(r["track_id"]),
                label=str(r["label"]),
                method=str(r.get("method", "cluster")),
                confidence=float(r["confidence"]) if "confidence" in r and pd.notna(r["confidence"]) else None,
                created_at=datetime.now(timezone.utc),
            )
        )
        inserted += 1

    sess.commit()
    return inserted


def export_active_csv(df_active: pd.DataFrame, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cols = ["track_id", "cluster", "label", "confidence", "uncertainty", "review_label"]
    for c in cols:
        if c not in df_active.columns:
            df_active[c] = None
    df_active[cols].to_csv(out_path, index=False, encoding="utf-8")


def import_review_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # review_label -> label + method=manual
    df_out = df.copy()
    df_out["label"] = df_out["review_label"].astype(str)
    df_out["method"] = "manual"
    return df_out[["track_id", "label", "method", "confidence"]]
