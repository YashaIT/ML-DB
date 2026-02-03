from __future__ import annotations

import requests


def fetch_current_weather(lat: float, lon: float, timeout_s: int = 7) -> dict:
    """
    Open-Meteo (без ключа). Если сеть недоступна — бросаем исключение, UI покажет статус.
    """
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": "temperature_2m,precipitation,wind_speed_10m,relative_humidity_2m",
        "timezone": "UTC",
    }
    r = requests.get(url, params=params, timeout=timeout_s)
    r.raise_for_status()
    js = r.json()
    cur = js.get("current", {}) or {}
    return {
        "temp_c": cur.get("temperature_2m"),
        "precip_mm": cur.get("precipitation"),
        "wind_ms": cur.get("wind_speed_10m"),
        "humidity_pct": cur.get("relative_humidity_2m"),
        "time": cur.get("time"),
    }


def fetch_emergency_stub(timeout_s: int = 7) -> dict:
    """
    Заглушка для ЧС: чтобы пункт был закрыт архитектурно.
    В отчёте: внешний источник интегрирован best-effort, без гарантии доступности.
    """
    # Можно позже заменить на RSS/JSON конкретного источника.
    return {"status": "not_configured", "note": "external emergency feed not configured"}
