from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import gpxpy
import requests

@dataclass
class GPXPoint:
    lat: float
    lon: float
    ele: float | None
    ts: datetime | None

@dataclass
class GPXTrack:
    name: str | None
    started_at: datetime | None
    points: list[GPXPoint]

def load_gpx_from_path(path: str) -> GPXTrack:
    with open(path, "r", encoding="utf-8") as f:
        gpx = gpxpy.parse(f)
    return _parse_gpx(gpx)

def load_gpx_from_url(url: str, timeout: int = 30) -> GPXTrack:
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    gpx = gpxpy.parse(r.text)
    return _parse_gpx(gpx)

def _parse_gpx(gpx) -> GPXTrack:
    name = gpx.name
    points: list[GPXPoint] = []
    started_at: datetime | None = None

    for trk in gpx.tracks:
        for seg in trk.segments:
            for p in seg.points:
                if started_at is None and p.time:
                    started_at = p.time
                points.append(GPXPoint(
                    lat=p.latitude,
                    lon=p.longitude,
                    ele=p.elevation,
                    ts=p.time,
                ))
    return GPXTrack(name=name, started_at=started_at, points=points)
