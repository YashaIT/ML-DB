from __future__ import annotations
from pathlib import Path

import argparse
from datetime import datetime, timezone

from shared.logging import setup_logger
from shared.db import engine, get_session
from shared.models import Base, Track, TrackPoint, WeatherPoint, TrackImage
from shared.config import settings
from shared.utils import deg2num

from agents.a1_ingest.providers.gpx import load_gpx_from_path, load_gpx_from_url
from agents.a1_ingest.services.weather import fetch_historical_weather_open_meteo
from agents.a1_ingest.services.imagery import download_tile, tile_path

logger = setup_logger("agent_a1_ingest")

def ensure_db():
    Base.metadata.create_all(bind=engine)

def is_stale(dt: datetime, ttl_hours: int) -> bool:
    age = datetime.now(timezone.utc) - dt.replace(tzinfo=timezone.utc)
    return age.total_seconds() > ttl_hours * 3600

def run(source: str, ref: str, region: str | None):
    ensure_db()
    sess = get_session()

    logger.info(f"ingest start: source={source} ref={ref}")

    if source == "gpx_file":
        gpx = load_gpx_from_path(ref)
    elif source == "gpx_url":
        gpx = load_gpx_from_url(ref)
    else:
        raise ValueError("source must be gpx_file|gpx_url")

    track = (
        sess.query(Track)
        .filter(Track.source == source, Track.source_ref == ref)
        .one_or_none()
    )

    now = datetime.utcnow()
    if track is None:
        track = Track(
            source=source,
            source_ref=ref,
            name=gpx.name,
            started_at=gpx.started_at,
            region=region,
            created_at=now,
            updated_at=now,
        )
        sess.add(track)
        sess.flush()
        logger.info(f"track created: id={track.id}")
    else:
        track.updated_at = now
        logger.info(f"track exists: id={track.id}")

    existing_points = (
        sess.query(TrackPoint)
        .filter(TrackPoint.track_id == track.id)
        .count()
    )

    if existing_points == 0:
        for i, p in enumerate(gpx.points):
            sess.add(TrackPoint(
                track_id=track.id,
                seq=i,
                lat=p.lat,
                lon=p.lon,
                ele=p.ele,
                ts=p.ts,
            ))
        sess.commit()
        logger.info(f"points inserted: {len(gpx.points)}")
    else:
        logger.info(f"points already in db: {existing_points}")

    points = (
        sess.query(TrackPoint)
        .filter(TrackPoint.track_id == track.id)
        .order_by(TrackPoint.seq)
        .all()
    )

    step = max(1, len(points) // 100)  # чтобы не делать слишком много запросов
    zoom = settings.tile_zoom

    for p in points[::step]:
        # Погода
        if p.ts:
            if p.weather is None or is_stale(p.weather.fetched_at, settings.data_ttl_hours):
                try:
                    w = fetch_historical_weather_open_meteo(p.lat, p.lon, p.ts)
                    if w:
                        wp = p.weather or WeatherPoint(track_point_id=p.id)
                        wp.temp_c = w.get("temp_c")
                        wp.humidity_pct = w.get("humidity_pct")
                        wp.precipitation_mm = w.get("precipitation_mm")
                        wp.wind_ms = w.get("wind_ms")
                        wp.provider = w.get("provider", "open_meteo")
                        wp.fetched_at = datetime.utcnow()
                        sess.add(wp)
                        sess.commit()
                except Exception as e:
                    logger.warning(f"weather failed point_id={p.id}: {e}")

        # Тайлы
        x, y = deg2num(p.lat, p.lon, zoom)

        for kind, tpl in [("topo", settings.topo_tile_url), ("sat", settings.sat_tile_url)]:
            out = tile_path(kind, zoom, x, y)
            need = True
            if out.exists():
                mtime = datetime.fromtimestamp(out.stat().st_mtime, tz=timezone.utc)
                need = is_stale(mtime, settings.data_ttl_hours)

            if need:
                try:
                    download_tile(tpl, zoom, x, y, out)
                    sess.add(TrackImage(
                        track_id=track.id,
                        kind=kind,
                        z=zoom, x=x, y=y,
                        file_path=str(out),
                        fetched_at=datetime.utcnow(),
                    ))
                    sess.commit()
                except Exception as e:
                    logger.warning(f"tile failed {kind} zxy={zoom}/{x}/{y}: {e}")

    logger.info(f"ingest done: track_id={track.id}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, choices=["gpx_file", "gpx_url"])
    ap.add_argument("--ref", required=True, help="path to .gpx / folder with .gpx / url")
    ap.add_argument("--region", default=None)
    args = ap.parse_args()

    if args.source == "gpx_file":
        p = Path(args.ref)

        if p.is_dir():
            files = sorted(p.glob("*.gpx"))
            if not files:
                raise FileNotFoundError(f"No .gpx files in folder: {p}")

            for f in files:
                run("gpx_file", str(f), args.region)
        else:
            run("gpx_file", str(p), args.region)

    else:
        # gpx_url
        run("gpx_url", args.ref, args.region)

if __name__ == "__main__":
    main()
