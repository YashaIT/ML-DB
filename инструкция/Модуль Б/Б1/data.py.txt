from __future__ import annotations

import math
from datetime import datetime
import pandas as pd
from sqlalchemy import text

from shared.db import get_session


EARTH_R = 6371000.0


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p = math.pi / 180.0
    a = 0.5 - math.cos((lat2 - lat1) * p) / 2 + math.cos(lat1 * p) * math.cos(lat2 * p) * (1 - math.cos((lon2 - lon1) * p)) / 2
    return 2 * EARTH_R * math.asin(math.sqrt(max(0.0, a)))


def load_dataset_points(region: str | None = None) -> pd.DataFrame:
    sess = get_session()

    q = """
    SELECT
      dp.track_id,
      dp.track_point_id,
      dp.region,
      dp.ts,
      dp.lat,
      dp.lon,
      dp.ele,
      dp.slope_deg,
      dp.dist_to_settlement_m,
      dp.green_index,
      dp.temp_c,
      dp.wind_ms,
      dp.humidity_pct,
      dp.precipitation_mm,
      dp.land_type,
      dp.frac_road,
      dp.frac_building,
      dp.frac_water,
      dp.frac_vegetation
    FROM dataset_points dp
    """
    params = {}
    if region and region != "ALL":
        q += " WHERE dp.region = :region"
        params["region"] = region

    df = pd.read_sql(text(q), sess.bind, params=params)
    return df


def load_tracks() -> pd.DataFrame:
    sess = get_session()
    df = pd.read_sql(text("SELECT id, name, region, started_at FROM tracks ORDER BY id"), sess.bind)
    return df


def compute_speed_series(df: pd.DataFrame) -> pd.DataFrame:
    """
    df: dataset_points for one track, sorted by ts (or track_point_id as fallback)
    returns rows with segment speed (m/s) between consecutive points
    """
    if df.empty:
        return df

    # ensure time
    df = df.copy()
    # ts может быть строкой из sqlite -> парсим
    df["ts"] = pd.to_datetime(df["ts"], errors="coerce", utc=True)

    # сортировка: по времени если есть, иначе по track_point_id
    if df["ts"].notna().any():
        df = df.sort_values(["ts", "track_point_id"])
    else:
        df = df.sort_values(["track_point_id"])

    speeds = []
    prev = None
    for _, row in df.iterrows():
        if prev is None:
            prev = row
            continue

        d = haversine_m(float(prev["lat"]), float(prev["lon"]), float(row["lat"]), float(row["lon"]))
        dt = None
        if pd.notna(prev["ts"]) and pd.notna(row["ts"]):
            dt = (row["ts"] - prev["ts"]).total_seconds()

        v = None
        if dt and dt > 0:
            v = d / dt

        speeds.append(
            {
                "track_id": int(row["track_id"]),
                "track_point_id": int(row["track_point_id"]),
                "ts": row["ts"].to_pydatetime() if pd.notna(row["ts"]) else None,
                "dist_m": float(d),
                "dt_s": float(dt) if dt else None,
                "speed_ms": float(v) if v is not None else None,
            }
        )
        prev = row

    return pd.DataFrame(speeds)
