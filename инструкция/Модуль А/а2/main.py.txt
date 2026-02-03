from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import text

from shared.db import engine, get_session
from shared.logging import setup_logger
from shared.models import Base, Track, TrackPoint, WeatherPoint
from shared.dataset_models import DatasetPoint
from shared.config import settings

from agents.a2_features.services.geo_features import slope_deg
from agents.a2_features.services.context_from_tiles import extract_context
from agents.a2_features.services.settlements import distance_to_nearest_settlement_m

logger = setup_logger("agent_a2_features")


# ---------------------------
# Вспомогательные структуры
# ---------------------------

@dataclass
class WeatherRow:
    tp_id: int
    seq: int
    ts: Optional[datetime]
    temp_c: Optional[float]
    precipitation_mm: Optional[float]
    humidity_pct: Optional[float]
    wind_ms: Optional[float]


def ensure_db() -> None:
    """Создаём таблицы, если их ещё нет."""
    Base.metadata.create_all(bind=engine)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _ensure_ts(track: Track, points: List[TrackPoint]) -> Tuple[datetime, List[Optional[datetime]]]:
    """
    Гарантируем, что ts будет НЕ NULL.
    Логика:
      - base_ts = track.started_at, если есть
      - иначе base_ts = первая ненулевая TrackPoint.ts
      - иначе base_ts = текущий момент UTC
    Если у конкретной точки ts NULL — ставим base_ts + seq секунд.
    Это не “истинное” время из GPX, но позволяет:
      - строить графики,
      - сопоставлять погоду “по времени”,
      - не иметь NULL в датасете.
    """
    base_ts = track.started_at
    if base_ts is None:
        for p in points:
            if p.ts is not None:
                base_ts = p.ts
                break
    if base_ts is None:
        base_ts = _utc_now()

    # приводим base_ts к aware UTC (если вдруг naive)
    if base_ts.tzinfo is None:
        base_ts = base_ts.replace(tzinfo=timezone.utc)
    else:
        base_ts = base_ts.astimezone(timezone.utc)

    out: List[Optional[datetime]] = []
    for p in points:
        if p.ts is not None:
            ts = p.ts
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            else:
                ts = ts.astimezone(timezone.utc)
            out.append(ts)
        else:
            # fallback: base + seq секунд
            out.append(base_ts + timedelta(seconds=int(p.seq or 0)))
    return base_ts, out


def _detect_weather_fk() -> str:
    """
    Выбираем реальную колонку WeatherPoint, которая ссылается на TrackPoint:
    - track_point_id (если есть)
    - иначе point_id (частый вариант)
    """
    if hasattr(WeatherPoint, "track_point_id"):
        return "track_point_id"
    if hasattr(WeatherPoint, "point_id"):
        return "point_id"
    # если ни одной — значит модель отличается, и надо смотреть схему БД
    raise RuntimeError("WeatherPoint has no track_point_id/point_id. Check shared.models and DB schema.")


def _load_weather_rows(sess, track_id: int, weather_fk: str) -> List[WeatherRow]:
    """
    Грузим погоду для трека одним запросом.
    Важно: join делаем по реальному FK (weather_fk).
    """
    # В SQLite удобнее читать через raw SQL, чтобы не зависеть от relationship.
    q = f"""
    SELECT
      wp.{weather_fk} AS tp_id,
      tp.seq AS seq,
      tp.ts  AS ts,
      wp.temp_c AS temp_c,
      wp.precipitation_mm AS precipitation_mm,
      wp.humidity_pct AS humidity_pct,
      wp.wind_ms AS wind_ms
    FROM weather_points wp
    JOIN track_points tp ON tp.id = wp.{weather_fk}
    WHERE tp.track_id = :track_id
    """
    rows = sess.execute(text(q), {"track_id": track_id}).mappings().all()

    out: List[WeatherRow] = []
    for r in rows:
        out.append(
            WeatherRow(
                tp_id=int(r["tp_id"]),
                seq=int(r["seq"] or 0),
                ts=r["ts"],
                temp_c=r["temp_c"],
                precipitation_mm=r["precipitation_mm"],
                humidity_pct=r["humidity_pct"],
                wind_ms=r["wind_ms"],
            )
        )
    return out


def _nearest_weather_for_point(
    p_seq: int,
    p_ts: Optional[datetime],
    exact_by_tp: Dict[int, WeatherRow],
    rows: List[WeatherRow],
) -> Optional[WeatherRow]:
    """
    Погода для точки:
      1) если есть точное совпадение по tp_id
      2) иначе ближайшая по времени (если p_ts есть)
      3) иначе ближайшая по seq
    rows обычно небольшие, но даже если 1000 — нормально.
    """
    # exact — самый быстрый
    # (будет работать только если в weather_points реально есть связка на tp_id)
    if p_seq is None:
        p_seq = 0

    # time-based / seq-based
    best: Optional[WeatherRow] = None
    best_d: Optional[float] = None

    if p_ts is not None:
        # nearest by time
        for r in rows:
            if r.ts is None:
                continue
            ts = r.ts
            # приводим к UTC aware
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            else:
                ts = ts.astimezone(timezone.utc)

            d = abs((ts - p_ts).total_seconds())
            if best is None or d < best_d:
                best = r
                best_d = d
        return best

    # fallback: nearest by seq
    for r in rows:
        d = abs(int(r.seq) - int(p_seq))
        if best is None or d < best_d:
            best = r
            best_d = float(d)
    return best


def run(
    radius_m: int,
    skip_context: bool,
    skip_settlement: bool,
    context_step: int = 10,
    settlement_step: Optional[int] = None,
    batch_size: int = 500,
) -> None:
    """
    Главный цикл агента A2: превращаем track_points -> dataset_points.

    Ускорения:
      - одним запросом берём существующие dataset_points для трека (set)
      - погоду берём одним запросом на трек
      - контекст (OSM/тайлы) считаем раз в context_step точек и переиспользуем
      - settlement считаем ещё реже (settlement_step), + кэш по округлённым координатам
      - вставка bulk пачками (batch_size)

    Важно: OSM/контекст НЕ отключаем — он работает через extract_context(), как требуется.
    """
    ensure_db()
    sess = get_session()

    if settlement_step is None:
        # обычно settlement тяжелее контекста из тайлов => считаем ещё реже
        settlement_step = max(1, context_step * 2)

    tracks = sess.query(Track).all()
    logger.info(
        f"tracks found: {len(tracks)}; radius_m={radius_m}; "
        f"skip_context={skip_context}; skip_settlement={skip_settlement}; "
        f"context_step={context_step}; settlement_step={settlement_step}; batch_size={batch_size}"
    )

    weather_fk = _detect_weather_fk()
    logger.info(f"weather FK detected: {weather_fk}")

    for track in tracks:
        # 1) грузим точки трека
        points: List[TrackPoint] = (
            sess.query(TrackPoint)
            .filter(TrackPoint.track_id == track.id)
            .order_by(TrackPoint.seq)
            .all()
        )
        logger.info(f"track_id={track.id} points={len(points)}")
        if not points:
            continue

        # 2) гарантируем ts (чтобы не было NULL)
        _, ts_list = _ensure_ts(track, points)

        # 3) загружаем погоду на трек одним запросом
        weather_rows = _load_weather_rows(sess, track.id, weather_fk)

        # 4) быстрый exact-словарь по tp_id (если FK реально соответствует track_points.id)
        exact_by_tp: Dict[int, WeatherRow] = {r.tp_id: r for r in weather_rows}

        # 5) берём уже существующие dataset_points для трека одним запросом
        existing_tp_ids = {
            int(x[0])
            for x in sess.execute(
                text("SELECT track_point_id FROM dataset_points WHERE track_id = :tid"),
                {"tid": track.id},
            ).all()
        }

        # 6) кэши для тяжёлых внешних вызовов (OSM/settlement)
        cached_ctx = None
        settlement_cache: Dict[Tuple[int, int], Optional[float]] = {}
        cached_settlement_val: Optional[float] = None

        inserted = 0
        buffer: List[Dict[str, Any]] = []

        # ---------------------------
        # Главный цикл по точкам трека
        # ---------------------------
        for i, p in enumerate(points):
            if p.id in existing_tp_ids:
                continue

            # prev/next для уклона (локальная геометрия)
            prev_p = points[i - 1] if i - 1 >= 0 else None
            next_p = points[i + 1] if i + 1 < len(points) else None
            sdeg = slope_deg(prev_p, next_p)

            ts = ts_list[i]  # уже гарантированно НЕ None

            # ---------------------------
            # Контекст (OSM/тайлы)
            # ---------------------------
            if skip_context:
                land_type = None
                objects = []
                frac_road = frac_building = frac_water = frac_vegetation = None
                green_index = None
            else:
                # считаем контекст раз в context_step точек
                if cached_ctx is None or (context_step > 0 and (i % context_step == 0)):
                    # Здесь подключается OSM/тайловая логика через extract_context()
                    cached_ctx = extract_context(p.lat, p.lon, radius_m=radius_m)

                land_type = cached_ctx.land_type
                objects = cached_ctx.objects
                frac_road = cached_ctx.frac_road
                frac_building = cached_ctx.frac_building
                frac_water = cached_ctx.frac_water
                frac_vegetation = cached_ctx.frac_vegetation
                green_index = cached_ctx.green_index

            # ---------------------------
            # Settlement (внешний API/поиск населённых пунктов)
            # ---------------------------
            dist_set: Optional[float] = None
            if not skip_settlement:
                # считаем ещё реже, иначе ловим TimeoutError как у тебя на скрине
                if (i % settlement_step) == 0 or cached_settlement_val is None:
                    # кэш по “клетке”: округлим координаты до ~100м–200м (1e-3 ≈ 111м)
                    key = (int(p.lat * 1000), int(p.lon * 1000))
                    if key in settlement_cache:
                        cached_settlement_val = settlement_cache[key]
                    else:
                        try:
                            cached_settlement_val = distance_to_nearest_settlement_m(p.lat, p.lon)
                        except Exception as e:
                            logger.warning(f"settlement failed: track={track.id} tp={p.seq} err={repr(e)}")
                            cached_settlement_val = None
                        settlement_cache[key] = cached_settlement_val
                dist_set = cached_settlement_val

            # ---------------------------
            # Погода
            # ---------------------------
            w = exact_by_tp.get(p.id)  # exact match по tp_id
            if w is None and weather_rows:
                w = _nearest_weather_for_point(int(p.seq or 0), ts, exact_by_tp, weather_rows)

            temp_c = w.temp_c if w else None
            precipitation_mm = w.precipitation_mm if w else None
            humidity_pct = w.humidity_pct if w else None
            wind_ms = w.wind_ms if w else None

            # ---------------------------
            # Готовим строку датасета (в buffer)
            # ---------------------------
            buffer.append(
                {
                    "track_id": track.id,
                    "track_point_id": p.id,
                    "region": track.region,
                    "ts": ts,
                    "lat": p.lat,
                    "lon": p.lon,
                    "ele": p.ele,
                    "radius_m": radius_m,
                    "slope_deg": sdeg,
                    "dist_to_settlement_m": dist_set,
                    "green_index": green_index,
                    "temp_c": temp_c,
                    "precipitation_mm": precipitation_mm,
                    "humidity_pct": humidity_pct,
                    "wind_ms": wind_ms,
                    "land_type": land_type,
                    "objects_json": json.dumps(objects, ensure_ascii=False),
                    "frac_road": frac_road,
                    "frac_building": frac_building,
                    "frac_water": frac_water,
                    "frac_vegetation": frac_vegetation,
                    "created_at": _utc_now(),
                }
            )
            inserted += 1

            # ---------------------------
            # Bulk insert пачками
            # ---------------------------
            if len(buffer) >= batch_size:
                sess.bulk_insert_mappings(DatasetPoint, buffer)
                sess.commit()
                buffer.clear()
                logger.info(f"track_id={track.id} committed inserted={inserted}")

        # дописываем хвост буфера
        if buffer:
            sess.bulk_insert_mappings(DatasetPoint, buffer)
            sess.commit()
            buffer.clear()

        logger.info(f"track_id={track.id} done inserted={inserted}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--radius_m", type=int, default=settings.radius_m)
    ap.add_argument("--skip_context", action="store_true")
    ap.add_argument("--skip_settlement", action="store_true")
    ap.add_argument("--context_step", type=int, default=10)
    ap.add_argument("--settlement_step", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=500)
    args = ap.parse_args()

    if args.radius_m < 200 or args.radius_m > 1000:
        raise ValueError("radius_m must be in 200..1000")
    if args.context_step < 0:
        raise ValueError("context_step must be >= 0")
    if args.batch_size < 50:
        raise ValueError("batch_size must be >= 50")

    settlement_step = None if args.settlement_step <= 0 else args.settlement_step

    run(
        radius_m=args.radius_m,
        skip_context=args.skip_context,
        skip_settlement=args.skip_settlement,
        context_step=args.context_step,
        settlement_step=settlement_step,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
