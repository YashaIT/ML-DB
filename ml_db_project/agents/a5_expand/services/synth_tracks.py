from __future__ import annotations

import math
import random
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone


@dataclass
class SynthPoint:
    lat: float
    lon: float
    ele: float | None
    ts: datetime | None


def jitter_points(points: list[SynthPoint], noise_m: float = 15.0, seed: int = 42) -> list[SynthPoint]:
    """
    Лёгкая аугментация трека: добавляем шум к координатам (метры -> градусы).
    """
    rnd = random.Random(seed)

    out: list[SynthPoint] = []
    for p in points:
        # грубая конверсия: 1 град широты ~ 111_320 м
        dlat = (rnd.uniform(-noise_m, noise_m) / 111_320.0)
        # долгота зависит от широты
        denom = 111_320.0 * math.cos(math.radians(p.lat)) if abs(p.lat) < 89 else 111_320.0
        dlon = rnd.uniform(-noise_m, noise_m) / max(1e-9, denom)

        out.append(SynthPoint(lat=p.lat + dlat, lon=p.lon + dlon, ele=p.ele, ts=p.ts))
    return out


def inject_time(points: list[SynthPoint], start: datetime, step_sec: int = 5) -> list[SynthPoint]:
    """
    Если нет времени — проставим равномерно.
    """
    out = []
    t = start
    for p in points:
        out.append(SynthPoint(lat=p.lat, lon=p.lon, ele=p.ele, ts=t))
        t = t + timedelta(seconds=step_sec)
    return out


def extreme_weather_profile(seed: int = 7) -> dict:
    """
    Синтетическая “экстремальная погода”.
    Это не API, а сценарная генерация, чтобы будущие модели видели редкие случаи.
    """
    rnd = random.Random(seed)

    # набор сценариев
    scenarios = [
        {"temp_c": rnd.uniform(-35, -20), "wind_ms": rnd.uniform(10, 25), "humidity_pct": rnd.uniform(40, 70), "precipitation_mm": rnd.uniform(0, 2)},
        {"temp_c": rnd.uniform(30, 42), "wind_ms": rnd.uniform(0, 8), "humidity_pct": rnd.uniform(20, 45), "precipitation_mm": 0.0},
        {"temp_c": rnd.uniform(-5, 10), "wind_ms": rnd.uniform(5, 18), "humidity_pct": rnd.uniform(85, 100), "precipitation_mm": rnd.uniform(8, 40)},
    ]
    sc = rnd.choice(scenarios)
    sc["scenario"] = "extreme_weather"
    return sc


def obstacle_flags(seed: int = 13) -> list[str]:
    rnd = random.Random(seed)
    possible = ["fallen_tree", "flooded_segment", "rockfall", "closed_bridge", "mud"]
    k = rnd.randint(1, 3)
    return rnd.sample(possible, k)
