from __future__ import annotations

import os
import math
from PIL import Image, ImageDraw

from shared.db import get_session
from shared.models import Track, TrackPoint


def _norm_points(latlons: list[tuple[float, float]], size: int, pad: int = 12) -> list[tuple[int, int]]:
    lats = [p[0] for p in latlons]
    lons = [p[1] for p in latlons]
    min_lat, max_lat = min(lats), max(lats)
    min_lon, max_lon = min(lons), max(lons)

    # защита от нулевого диапазона
    dlat = max(max_lat - min_lat, 1e-9)
    dlon = max(max_lon - min_lon, 1e-9)

    w = size - 2 * pad
    h = size - 2 * pad

    pts = []
    for lat, lon in latlons:
        x = (lon - min_lon) / dlon
        y = (lat - min_lat) / dlat

        # y инвертируем для картинки
        px = int(pad + x * w)
        py = int(pad + (1.0 - y) * h)
        pts.append((px, py))
    return pts


def render_track_images(out_dir: str, max_tracks: int = 20, size: int = 512) -> list[str]:
    """
    Создаёт PNG-картинки треков (полилинии) из БД.
    Это и будет "изображение маршрута" для аугментации.
    """
    os.makedirs(out_dir, exist_ok=True)
    s = get_session()

    tracks = s.query(Track).order_by(Track.id.desc()).limit(max_tracks).all()
    produced: list[str] = []

    for t in tracks:
        pts = (
            s.query(TrackPoint)
            .filter(TrackPoint.track_id == t.id)
            .order_by(TrackPoint.seq)
            .all()
        )
        if len(pts) < 2:
            continue

        latlons = [(float(p.lat), float(p.lon)) for p in pts]
        xy = _norm_points(latlons, size=size)

        img = Image.new("RGB", (size, size), (255, 255, 255))
        dr = ImageDraw.Draw(img)

        # линия трека
        dr.line(xy, fill=(0, 0, 0), width=3)

        # старт/финиш
        dr.ellipse((xy[0][0]-5, xy[0][1]-5, xy[0][0]+5, xy[0][1]+5), outline=(0, 180, 0), width=3)
        dr.ellipse((xy[-1][0]-5, xy[-1][1]-5, xy[-1][0]+5, xy[-1][1]+5), outline=(180, 0, 0), width=3)

        fn = os.path.join(out_dir, f"track_{t.id}.png")
        img.save(fn)
        produced.append(fn)

    return produced
