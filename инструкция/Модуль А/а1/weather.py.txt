from __future__ import annotations

from datetime import datetime
import requests

def fetch_historical_weather_open_meteo(lat: float, lon: float, ts: datetime) -> dict:
    date = ts.date().isoformat()
    url = (
        "https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={lat}&longitude={lon}"
        f"&start_date={date}&end_date={date}"
        "&hourly=temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m"
        "&timezone=UTC"
    )
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    data = r.json()

    hourly = data.get("hourly", {})
    times = hourly.get("time", [])
    if not times:
        return {}

    # Берём ближайший час (упрощённо)
    target_hour = ts.replace(minute=0, second=0, microsecond=0).strftime("%Y-%m-%dT%H")
    idx = 0
    for i, t in enumerate(times):
        if t.startswith(target_hour):
            idx = i
            break

    def pick(key: str):
        arr = hourly.get(key, [])
        return arr[idx] if idx < len(arr) else None

    return {
        "temp_c": pick("temperature_2m"),
        "humidity_pct": pick("relative_humidity_2m"),
        "precipitation_mm": pick("precipitation"),
        "wind_ms": pick("wind_speed_10m"),
        "provider": "open_meteo",
    }
