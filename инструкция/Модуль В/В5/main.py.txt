from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is in PYTHONPATH so "shared" imports work under streamlit run
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sqlalchemy import text
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from shared.db import get_session
from shared.logging import setup_logger

logger = setup_logger("agent_v5_risk_viz")


# ----------------------------
# Risk scoring (0..1)
# ----------------------------
def robust_minmax(s: pd.Series, q_lo: float = 0.05, q_hi: float = 0.95) -> pd.Series:
    s = s.astype(float)
    if not s.notna().any():
        return pd.Series(np.zeros(len(s)), index=s.index)

    lo = float(s.quantile(q_lo))
    hi = float(s.quantile(q_hi))
    if hi - lo < 1e-9:
        return pd.Series(np.zeros(len(s)), index=s.index)

    x = (s.clip(lo, hi) - lo) / (hi - lo)
    return x.clip(0.0, 1.0)


def compute_risk(df: pd.DataFrame) -> pd.DataFrame:
    """
    Композитный риск 0..1 для визуализации/сравнения.
    Компоненты:
      - slope_deg: уклон
      - precipitation_mm: осадки
      - wind_ms: ветер
      - frac_water: вода рядом
      - dist_to_settlement_m: удалённость от населённых пунктов
    """
    out = df.copy()

    cols = ["slope_deg", "precipitation_mm", "wind_ms", "frac_water", "dist_to_settlement_m"]
    for c in cols:
        if c not in out.columns:
            out[c] = np.nan
        out[c] = pd.to_numeric(out[c], errors="coerce")

    slope = out["slope_deg"].fillna(out["slope_deg"].median() if out["slope_deg"].notna().any() else 0.0)
    prec = out["precipitation_mm"].fillna(0.0)
    wind = out["wind_ms"].fillna(out["wind_ms"].median() if out["wind_ms"].notna().any() else 0.0)
    water = out["frac_water"].fillna(0.0)
    dist = out["dist_to_settlement_m"].fillna(
        out["dist_to_settlement_m"].median() if out["dist_to_settlement_m"].notna().any() else 0.0
    )

    s_slope = robust_minmax(slope)
    s_prec = robust_minmax(prec)
    s_wind = robust_minmax(wind)
    s_water = robust_minmax(water)
    s_dist = robust_minmax(dist)

    out["risk_score"] = (
        0.30 * s_slope
        + 0.25 * s_prec
        + 0.15 * s_wind
        + 0.15 * s_dist
        + 0.15 * s_water
    ).clip(0.0, 1.0)

    out["risk_level"] = pd.cut(
        out["risk_score"],
        bins=[-0.001, 0.33, 0.66, 1.001],
        labels=["low", "medium", "high"],
    ).astype(str)

    return out


# ----------------------------
# Data loading
# ----------------------------
def load_points(limit_points: int = 40000) -> pd.DataFrame:
    """
    dataset_points + region из tracks.
    limit_points нужен, чтобы 3D не тормозил.
    """
    sess = get_session()
    q = """
    SELECT
      dp.track_id,
      dp.track_point_id,
      dp.ts,
      dp.lat,
      dp.lon,
      dp.ele,
      dp.slope_deg,
      dp.dist_to_settlement_m,
      dp.green_index,
      dp.temp_c,
      dp.precipitation_mm,
      dp.humidity_pct,
      dp.wind_ms,
      dp.frac_water,
      dp.frac_vegetation,
      dp.frac_road,
      dp.frac_building,
      t.region
    FROM dataset_points dp
    JOIN tracks t ON t.id = dp.track_id
    ORDER BY dp.track_id, dp.ts
    LIMIT :lim
    """
    df = pd.read_sql(text(q), sess.bind, params={"lim": int(limit_points)})

    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"], errors="coerce", utc=True)

    df["region"] = df["region"].fillna("UNKNOWN").astype(str)

    df["ele"] = pd.to_numeric(df["ele"], errors="coerce")
    if df["ele"].isna().all():
        df["ele"] = 0.0
    else:
        df["ele"] = df["ele"].fillna(df["ele"].median())

    return df


def ensure_time(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if out["ts"].isna().all():
        base = pd.Timestamp("2026-01-01", tz="UTC")
        out = out.sort_values(["track_id", "track_point_id"])
        out["ts"] = base + pd.to_timedelta(np.arange(len(out)), unit="m")
    return out


def region_timeseries(df: pd.DataFrame, freq: str = "D") -> pd.DataFrame:
    out = ensure_time(df)
    out["tbin"] = out["ts"].dt.floor(freq)
    agg = (
        out.groupby(["region", "tbin"], as_index=False)
        .agg(
            risk_mean=("risk_score", "mean"),
            risk_p90=("risk_score", lambda x: float(np.nanpercentile(x, 90))),
            points=("risk_score", "count"),
        )
        .sort_values(["region", "tbin"])
    )
    return agg


# ----------------------------
# Forecasting
# ----------------------------
def forecast_region_series(ts_df: pd.DataFrame, value_col: str, horizon: int) -> pd.DataFrame:
    """
    Прогноз по регионам (ExponentialSmoothing).
    Возвращает: region, tbin, y, kind=history|forecast
    """
    rows = []

    for region, g in ts_df.groupby("region"):
        g = g.sort_values("tbin")
        y = g[value_col].astype(float).values
        t = g["tbin"].values

        for ti, yi in zip(t, y):
            rows.append({"region": region, "tbin": pd.Timestamp(ti), "y": float(yi), "kind": "history"})

        if len(y) < 3 or horizon <= 0:
            continue

        try:
            model = ExponentialSmoothing(y, trend="add", seasonal=None, initialization_method="estimated")
            fit = model.fit(optimized=True)
            yhat = fit.forecast(horizon)
        except Exception:
            yhat = np.full(horizon, float(np.nanmean(y)))

        step = pd.Timedelta(days=1)
        if len(g) >= 2:
            step = g["tbin"].iloc[1] - g["tbin"].iloc[0]
            if step <= pd.Timedelta(0):
                step = pd.Timedelta(days=1)

        last_t = pd.Timestamp(g["tbin"].iloc[-1])
        for i in range(horizon):
            rows.append({"region": region, "tbin": last_t + step * (i + 1), "y": float(yhat[i]), "kind": "forecast"})

    return pd.DataFrame(rows)


# ----------------------------
# Plot helpers
# ----------------------------
def plot_3d_points(df: pd.DataFrame, max_points: int = 15000) -> go.Figure:
    d = df.copy()
    if len(d) > max_points:
        d = d.sample(max_points, random_state=42)

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=d["lon"],
                y=d["lat"],
                z=d["ele"],
                mode="markers",
                marker=dict(
                    size=3,
                    color=d["risk_score"],
                    colorscale="Viridis",
                    opacity=0.85,
                    colorbar=dict(title="risk"),
                ),
                text=[
                    f"region={r}<br>risk={rs:.2f}<br>ts={t}"
                    for r, rs, t in zip(d["region"], d["risk_score"], d["ts"])
                ],
                hoverinfo="text",
            )
        ]
    )
    fig.update_layout(
        title="3D risk map (lon, lat, elevation)",
        scene=dict(xaxis_title="lon", yaxis_title="lat", zaxis_title="elevation"),
        margin=dict(l=0, r=0, t=40, b=0),
        height=650,
    )
    return fig


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit_points", type=int, default=40000)
    ap.add_argument("--max_3d_points", type=int, default=15000)
    ap.add_argument("--freq", type=str, default="D", choices=["H", "D", "W"])
    ap.add_argument("--forecast_horizon", type=int, default=7)
    args, _ = ap.parse_known_args()

    st.set_page_config(page_title="Risk forecast viz", layout="wide")
    st.title("Risk: 3D map + dynamics + forecast")

    df = load_points(limit_points=args.limit_points)
    st.caption(f"Loaded points: {len(df)}")

    regions = ["ALL"] + sorted(df["region"].unique().tolist())
    region = st.sidebar.selectbox("Region", regions, index=0)

    df = compute_risk(df)

    if region != "ALL":
        df_view = df[df["region"] == region].copy()
    else:
        df_view = df.copy()

    st.subheader("3D map")
    st.plotly_chart(plot_3d_points(df_view, max_points=args.max_3d_points), use_container_width=True)

    st.subheader("Risk over time")
    ts = region_timeseries(df, freq=args.freq)
    if region != "ALL":
        ts = ts[ts["region"] == region].copy()

    fc = forecast_region_series(ts, value_col="risk_mean", horizon=args.forecast_horizon)
    if not fc.empty:
        fig_line = px.line(
            fc,
            x="tbin",
            y="y",
            color="region" if region == "ALL" else None,
            line_dash="kind",
            title="Risk mean: history + forecast",
        )
        st.plotly_chart(fig_line, use_container_width=True)
    else:
        st.info("Forecast skipped: not enough time bins for forecasting.")

    fig_p90 = px.line(
        ts,
        x="tbin",
        y="risk_p90",
        color="region" if region == "ALL" else None,
        title="Risk p90 over time",
    )
    st.plotly_chart(fig_p90, use_container_width=True)

    if region == "ALL":
        ts_anim = ts.copy()
        ts_anim["tbin_str"] = ts_anim["tbin"].astype(str)
        fig_anim = px.bar(
            ts_anim,
            x="region",
            y="risk_mean",
            animation_frame="tbin_str",
            title="Risk mean by regions (animated by time)",
        )
        st.plotly_chart(fig_anim, use_container_width=True)

    st.subheader("Summary metrics")
    c1, c2, c3 = st.columns(3)
    c1.metric("Mean risk", f"{df_view['risk_score'].mean():.3f}")
    c2.metric("P90 risk", f"{np.nanpercentile(df_view['risk_score'], 90):.3f}")
    c3.metric("High risk share", f"{(df_view['risk_level'] == 'high').mean():.2%}")

    st.caption(
        "Risk score = composite(slope, precipitation, wind, distance_to_settlement, nearby water). "
        "Forecast = ExponentialSmoothing on aggregated regional time series."
    )


if __name__ == "__main__":
    main()
