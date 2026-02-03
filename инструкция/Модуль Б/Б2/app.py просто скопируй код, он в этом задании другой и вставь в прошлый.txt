from __future__ import annotations

import os
import pandas as pd
import streamlit as st
import plotly.express as px
from streamlit_autorefresh import st_autorefresh

from agents.b1_dashboard.services.auth import get_role
from agents.b1_dashboard.services.data import load_tracks, load_dataset_points, compute_speed_series
from agents.b1_dashboard.services.external import fetch_current_weather, fetch_emergency_stub
from agents.b1_dashboard.services.alerts import recompute_speed_drop_alerts, load_alerts

from agents.b1_dashboard.services.analytics import (
    aggregate_tracks,
    cluster_tracks,
    popularity_heatmap,
    build_track_graph,
    stl_activity,
)

st.set_page_config(page_title="Trails Dashboard", layout="wide")
st.title("Аналитика туристических маршрутов (Module B)")

# автообновление
refresh_sec = st.sidebar.slider("Auto refresh (seconds)", 5, 60, 15, 5)
st_autorefresh(interval=refresh_sec * 1000, key="autorefresh")

# доступ
st.sidebar.subheader("Access")
pw = st.sidebar.text_input("Password", type="password")
role = get_role(pw)
st.sidebar.write(f"Role: **{role}**")

if role == "none":
    st.warning("Нужен пароль (viewer или admin).")
    st.stop()

# данные
tracks = load_tracks()
regions = ["ALL"] + sorted([r for r in tracks["region"].dropna().unique().tolist() if str(r).strip()])

region = st.sidebar.selectbox("Region", regions, index=0)

track_ids = tracks["id"].tolist()
track_id = st.sidebar.selectbox("Track", track_ids, index=0)

df = load_dataset_points(region=None if region == "ALL" else region)
df_track = df[df["track_id"] == track_id].copy()

# KPI
c1, c2, c3, c4 = st.columns(4)
c1.metric("Tracks in DB", int(tracks.shape[0]))
c2.metric("Dataset points", int(df.shape[0]))
c3.metric("Points (selected track)", int(df_track.shape[0]))
c4.metric("Regions", int(len(regions) - 1))

st.divider()

# внешние источники
st.subheader("Внешние источники (по возможности)")
if not df_track.empty:
    lat0 = float(df_track.iloc[0]["lat"])
    lon0 = float(df_track.iloc[0]["lon"])
    colw1, colw2 = st.columns([2, 3])
    with colw1:
        try:
            w = fetch_current_weather(lat0, lon0)
            st.success("Погода сейчас (Open-Meteo)")
            st.json(w)
        except Exception as e:
            st.warning(f"Погода недоступна: {e}")
    with colw2:
        st.info("Чрезвычайные ситуации/инциденты (заглушка)")
        st.json(fetch_emergency_stub())
else:
    st.info("Нет точек для выбранного трека, внешние источники не запрашиваем.")

st.divider()

# карта
st.subheader("Карта трека")
if df_track.empty:
    st.info("Выбранный трек не имеет dataset_points.")
else:
    fig = px.scatter_mapbox(
        df_track,
        lat="lat",
        lon="lon",
        hover_data=["ts", "ele", "land_type"],
        zoom=10,
        height=450,
    )
    fig.update_layout(mapbox_style="open-street-map", margin={"r": 0, "t": 0, "l": 0, "b": 0})
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# быстрые графики
st.subheader("Быстрые метрики")
if not df_track.empty:
    left, right = st.columns(2)

    with left:
        st.caption("Распределение типов местности")
        land = df_track["land_type"].fillna("unknown").value_counts().reset_index()
        land.columns = ["land_type", "count"]
        st.plotly_chart(px.bar(land, x="land_type", y="count"), use_container_width=True)

    with right:
        st.caption("Высота по времени")
        tmp = df_track.copy()
        tmp["ts"] = pd.to_datetime(tmp["ts"], errors="coerce", utc=True)
        tmp = tmp.dropna(subset=["ts"])
        if tmp.empty:
            st.info("Нет ts для графика.")
        else:
            st.plotly_chart(px.line(tmp.sort_values("ts"), x="ts", y="ele"), use_container_width=True)

st.divider()

# алертинг
st.subheader("Алертинг: резкое падение активности (через скорость сегментов)")
if df_track.empty:
    st.info("Нет данных для алертинга.")
else:
    speed_df = compute_speed_series(df_track)
    th = st.slider("Speed threshold (m/s)", 0.1, 3.0, 0.5, 0.1)

    if role == "admin":
        if st.button("Пересчитать оповещения по треку"):
            n = recompute_speed_drop_alerts(track_id=track_id, speed_df=speed_df, threshold_ms=th)
            st.success(f"Inserted alerts: {n}")

    st.caption("Скорость по сегментам")
    if not speed_df.empty and speed_df["speed_ms"].notna().any():
        st.plotly_chart(px.line(speed_df, x="track_point_id", y="speed_ms"), use_container_width=True)
    else:
        st.info("Скорость не рассчитана (нет времени или слишком мало точек).")

    st.caption("Последние алерты")
    alerts_df = load_alerts(track_id=track_id)
    st.dataframe(alerts_df, use_container_width=True, height=250)

st.divider()
st.header("2.2 Расширенная аналитика")

tab1, tab2, tab3, tab4 = st.tabs(
    ["Кластеры треков", "Популярные зоны", "Граф маршрута", "Временная аналитика (STL)"]
)

with tab1:
    st.subheader("Кластеризация маршрутов (KMeans по агрегатам)")
    agg = aggregate_tracks(df if region == "ALL" else df[df["region"] == region])
    if agg.empty:
        st.info("Нет данных для агрегации/кластеризации.")
    else:
        k = st.slider("k (кол-во кластеров)", 2, 6, 3, 1)
        clustered = cluster_tracks(agg, k=k)
        st.dataframe(clustered.sort_values("cluster"), use_container_width=True, height=280)

        if clustered.shape[0] >= 3:
            fig = px.scatter(
                clustered,
                x="length_km",
                y="mean_green",
                color="cluster",
                hover_data=["track_id", "n_points", "ts_coverage"],
                title="Кластеры: length_km vs mean_green",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Слишком мало треков для наглядной визуализации.")

with tab2:
    st.subheader("Популярные зоны: плотность точек на сетке")
    if df_track.empty:
        st.info("Нет точек для выбранного трека.")
    else:
        grid_m = st.slider("Размер ячейки (м)", 100, 1000, 250, 50)
        heat = popularity_heatmap(df_track, grid_m=grid_m)
        if heat.empty:
            st.info("Не удалось построить теплокарту.")
        else:
            st.caption("Топ-20 ячеек по числу точек")
            st.dataframe(heat.head(20), use_container_width=True, height=260)

            fig = px.scatter_mapbox(
                heat.head(200),
                lat="lat",
                lon="lon",
                size="count",
                hover_data=["count", "gx", "gy"],
                zoom=10,
                height=450,
                title="Теплокарта популярности (по плотности точек)",
            )
            fig.update_layout(mapbox_style="open-street-map", margin={"r": 0, "t": 40, "l": 0, "b": 0})
            st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("Граф маршрута")
    if df_track.empty:
        st.info("Нет точек для графа.")
    else:
        every_n = st.slider("Прореживание (каждая N-я точка)", 1, 50, 5, 1)
        G, edges = build_track_graph(df_track, every_n=every_n)
        st.write(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")

        if edges.empty:
            st.info("Граф пустой (слишком мало точек).")
        else:
            st.caption("Edge list (первые 20)")
            st.dataframe(edges.head(20), use_container_width=True)

            total_dist = float(edges["dist_m"].sum())
            st.metric("Длина по графу (м)", f"{total_dist:.0f}")

            nodes = pd.DataFrame(
                [{"node": n, "lat": G.nodes[n]["lat"], "lon": G.nodes[n]["lon"]} for n in G.nodes]
            ).sort_values("node")

            fig = px.line_mapbox(nodes, lat="lat", lon="lon", zoom=10, height=450, title="Граф (узлы) как линия")
            fig.update_layout(mapbox_style="open-street-map", margin={"r": 0, "t": 40, "l": 0, "b": 0})
            st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.subheader("Временная аналитика (STL по активности)")
    st.caption("Работает только если в dataset_points есть ts. Если ts нет — это показывается явно.")
    if df_track.empty:
        st.info("Нет точек.")
    else:
        out = stl_activity(df_track)
        if "status" in out.columns:
            st.info(f"STL недоступен: {out.iloc[0]['status']}")
        else:
            st.dataframe(out.head(20), use_container_width=True, height=220)
            fig1 = px.line(out, x="ts", y="observed", title="Observed (точек/мин)")
            st.plotly_chart(fig1, use_container_width=True)
            fig2 = px.line(out, x="ts", y="trend", title="Trend")
            st.plotly_chart(fig2, use_container_width=True)
            fig3 = px.line(out, x="ts", y="seasonal", title="Seasonal")
            st.plotly_chart(fig3, use_container_width=True)
            fig4 = px.line(out, x="ts", y="resid", title="Residual")
            st.plotly_chart(fig4, use_container_width=True)
