from __future__ import annotations

import pandas as pd
import folium


def build_tracks_map(df: pd.DataFrame, out_html: str) -> None:
    """
    Рисуем треки полилиниями на folium.
    Берём точки из dataset_points и группируем по track_id.
    """
    work = df.dropna(subset=["lat", "lon"])
    if work.empty:
        # создаём пустую карту
        m = folium.Map(location=[55.75, 37.62], zoom_start=4)
        m.save(out_html)
        return

    # центр карты по медиане
    center_lat = float(work["lat"].median())
    center_lon = float(work["lon"].median())
    m = folium.Map(location=[center_lat, center_lon], zoom_start=10)

    for track_id, g in work.groupby("track_id"):
        g = g.sort_values("track_point_id")  # если seq нет в df
        coords = list(zip(g["lat"].astype(float), g["lon"].astype(float)))
        if len(coords) < 2:
            continue

        region = str(g["region"].dropna().iloc[0]) if g["region"].notna().any() else "unknown"
        folium.PolyLine(
            coords,
            tooltip=f"track_id={track_id}, region={region}, points={len(coords)}",
            weight=3,
            opacity=0.9,
        ).add_to(m)

    m.save(out_html)
