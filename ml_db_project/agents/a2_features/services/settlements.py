from __future__ import annotations

import requests

def distance_to_nearest_settlement_m(lat: float, lon: float) -> float | None:
    """
    Overpass: ищем ближайший населённый пункт в радиусе 5км.
    Если сеть недоступна — вернём None.
    """
    query = f"""
    [out:json];
    (
      node(around:5000,{lat},{lon})["place"];
    );
    out center;
    """
    try:
        r = requests.post("https://overpass-api.de/api/interpreter", data=query.encode("utf-8"), timeout=20)
        r.raise_for_status()
        data = r.json()
        els = data.get("elements", [])
        if not els:
            return None

        # берём первый как приближение (можно улучшить выбором по расстоянию)
        e = els[0]
        slat = e.get("lat")
        slon = e.get("lon")
        if slat is None or slon is None:
            return None

        from agents.a2_features.services.geo_features import haversine_m
        return float(haversine_m(lat, lon, slat, slon))
    except Exception:
        return None
