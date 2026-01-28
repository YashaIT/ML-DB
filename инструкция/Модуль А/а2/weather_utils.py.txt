from __future__ import annotations

def nearest_weather_by_time(point, weather_points):
    """
    weather_points: list[WeatherPoint]
    Требует point.ts.
    """
    if not weather_points or point.ts is None:
        return None

    best = None
    best_dt = None

    for w in weather_points:
        if w.point is None or w.point.ts is None:
            continue
        dt = abs((w.point.ts - point.ts).total_seconds())
        if best is None or dt < best_dt:
            best = w
            best_dt = dt

    return best


def nearest_weather_by_seq(point, weather_points):
    """
    Fallback если нет времени: выбираем ближайшую погодную точку по seq в рамках трека.
    """
    if not weather_points:
        return None
    if point is None:
        return None

    best = None
    best_d = None

    for w in weather_points:
        if w.point is None:
            continue
        d = abs(w.point.seq - point.seq)
        if best is None or d < best_d:
            best = w
            best_d = d

    return best
