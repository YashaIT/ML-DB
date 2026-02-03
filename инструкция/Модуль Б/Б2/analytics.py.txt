from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import networkx as nx
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from agents.b1_dashboard.services.data import haversine_m

try:
    from statsmodels.tsa.seasonal import STL
except Exception:  # если вдруг нет statsmodels
    STL = None


@dataclass
class TrackAgg:
    track_id: int
    n_points: int
    length_km: float
    ele_gain_m: float | None
    mean_slope: float | None
    mean_green: float | None
    frac_forest: float | None
    frac_water: float | None
    frac_road: float | None
    frac_building: float | None
    ts_coverage: str  # "ok" | "missing"


def aggregate_tracks(df: pd.DataFrame) -> pd.DataFrame:
    """
    df: dataset_points (может содержать разные треки)
    Возвращает агрегаты по каждому треку для кластеризации/сравнений.
    """
    if df.empty:
        return pd.DataFrame()

    df = df.copy()
    df["ts"] = pd.to_datetime(df["ts"], errors="coerce", utc=True)

    rows: list[dict] = []

    for tid, g in df.groupby("track_id"):
        g = g.sort_values(["ts", "track_point_id"]) if g["ts"].notna().any() else g.sort_values(["track_point_id"])

        n_points = int(g.shape[0])

        # длина по последовательным точкам
        length_m = 0.0
        prev = None
        for _, r in g.iterrows():
            if prev is not None:
                length_m += haversine_m(float(prev["lat"]), float(prev["lon"]), float(r["lat"]), float(r["lon"]))
            prev = r
        length_km = length_m / 1000.0

        # набор высоты (если есть ele)
        ele_gain = None
        if g["ele"].notna().any():
            diffs = g["ele"].astype(float).diff()
            ele_gain = float(diffs[diffs > 0].sum()) if diffs is not None else None

        ts_ok = "ok" if g["ts"].notna().any() else "missing"

        # доли типов местности (простая интерпретация)
        land = g["land_type"].fillna("unknown").astype(str).str.lower()
        frac_forest = float((land == "forest").mean()) if n_points else None
        frac_water = float((land == "water").mean()) if n_points else None
        frac_road = float((land == "road_area").mean()) if n_points else None
        frac_building = float((land == "urban").mean()) if n_points else None

        rows.append(
            TrackAgg(
                track_id=int(tid),
                n_points=n_points,
                length_km=float(length_km),
                ele_gain_m=ele_gain,
                mean_slope=float(g["slope_deg"].dropna().astype(float).mean()) if g["slope_deg"].notna().any() else None,
                mean_green=float(g["green_index"].dropna().astype(float).mean()) if g["green_index"].notna().any() else None,
                frac_forest=frac_forest,
                frac_water=frac_water,
                frac_road=frac_road,
                frac_building=frac_building,
                ts_coverage=ts_ok,
            ).__dict__
        )

    return pd.DataFrame(rows)


def cluster_tracks(agg_df: pd.DataFrame, k: int = 3) -> pd.DataFrame:
    """
    Кластеризация треков по агрегатам. Работает даже если часть фич отсутствует.
    """
    if agg_df.empty:
        return agg_df

    df = agg_df.copy()

    feats = ["length_km", "ele_gain_m", "mean_slope", "mean_green", "frac_forest", "frac_water", "frac_road", "frac_building"]
    X = df[feats].copy()

    # заполнение пропусков медианами
    for c in feats:
        if X[c].isna().all():
            X[c] = 0.0
        else:
            X[c] = X[c].astype(float).fillna(float(X[c].median()))

    # нормализация + KMeans
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X.values)

    k = max(2, min(int(k), int(len(df))))  # чтобы не упасть
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    df["cluster"] = km.fit_predict(Xs)

    return df


def popularity_heatmap(df_track: pd.DataFrame, grid_m: int = 250) -> pd.DataFrame:
    """
    “Популярность” как плотность посещений: считаем количество точек в ячейках сетки.
    Для закрытия пункта про популярные зоны.
    """
    if df_track.empty:
        return pd.DataFrame()

    # грубая метрическая сетка через приведение к “условным метрам”
    lat = df_track["lat"].astype(float).values
    lon = df_track["lon"].astype(float).values

    lat0 = float(np.nanmean(lat))
    m_per_deg_lat = 111_320.0
    m_per_deg_lon = 111_320.0 * math.cos(math.radians(lat0))

    x_m = (lon - lon.min()) * m_per_deg_lon
    y_m = (lat - lat.min()) * m_per_deg_lat

    gx = (x_m // grid_m).astype(int)
    gy = (y_m // grid_m).astype(int)

    out = pd.DataFrame({"gx": gx, "gy": gy, "lat": lat, "lon": lon})
    heat = (
        out.groupby(["gx", "gy"])
        .agg(count=("gx", "size"), lat=("lat", "mean"), lon=("lon", "mean"))
        .reset_index()
        .sort_values("count", ascending=False)
    )
    return heat


def build_track_graph(df_track: pd.DataFrame, every_n: int = 5) -> tuple[nx.DiGraph, pd.DataFrame]:
    """
    Граф маршрута: узлы — точки (с прореживанием every_n), рёбра — переходы.
    Вес ребра — расстояние (м). Для визуализации выдаём edge list.
    """
    G = nx.DiGraph()

    if df_track.empty:
        return G, pd.DataFrame()

    df = df_track.copy()
    df["ts"] = pd.to_datetime(df["ts"], errors="coerce", utc=True)

    df = df.sort_values(["ts", "track_point_id"]) if df["ts"].notna().any() else df.sort_values(["track_point_id"])
    df = df.reset_index(drop=True)

    # прореживание
    idxs = list(range(0, len(df), max(1, int(every_n))))
    if idxs[-1] != len(df) - 1:
        idxs.append(len(df) - 1)

    nodes = df.loc[idxs, ["track_point_id", "lat", "lon", "ele"]].copy()
    nodes["node"] = range(len(nodes))

    for _, r in nodes.iterrows():
        G.add_node(int(r["node"]), track_point_id=int(r["track_point_id"]), lat=float(r["lat"]), lon=float(r["lon"]), ele=float(r["ele"]) if pd.notna(r["ele"]) else None)

    edges = []
    for i in range(len(nodes) - 1):
        a = nodes.iloc[i]
        b = nodes.iloc[i + 1]
        dist = haversine_m(float(a["lat"]), float(a["lon"]), float(b["lat"]), float(b["lon"]))
        G.add_edge(int(a["node"]), int(b["node"]), weight=float(dist))
        edges.append({"src": int(a["node"]), "dst": int(b["node"]), "dist_m": float(dist)})

    return G, pd.DataFrame(edges)


def stl_activity(df_track: pd.DataFrame) -> pd.DataFrame:
    """
    STL по активности (кол-во точек по минутам).
    Работает только если есть ts и есть statsmodels.
    """
    if STL is None:
        return pd.DataFrame({"status": ["no_statsmodels"]})

    if df_track.empty:
        return pd.DataFrame({"status": ["empty"]})

    tmp = df_track.copy()
    tmp["ts"] = pd.to_datetime(tmp["ts"], errors="coerce", utc=True)
    tmp = tmp.dropna(subset=["ts"])
    if tmp.empty:
        return pd.DataFrame({"status": ["no_ts"]})

    # временной ряд: точки в минуту
    s = tmp.set_index("ts").resample("1min").size().astype(float)
    if s.sum() < 10:
        return pd.DataFrame({"status": ["too_few_events"]})

    # период 60 минут как базовый (можно менять)
    period = 60 if len(s) >= 120 else max(2, len(s) // 2)
    res = STL(s, period=period, robust=True).fit()

    out = pd.DataFrame(
        {
            "ts": s.index,
            "observed": s.values,
            "trend": res.trend,
            "seasonal": res.seasonal,
            "resid": res.resid,
        }
    )
    return out
