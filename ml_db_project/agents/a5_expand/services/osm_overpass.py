from __future__ import annotations

import time
import requests


OVERPASS_URL = "https://overpass-api.de/api/interpreter"


def fetch_hiking_ways(bbox: tuple[float, float, float, float], limit: int = 10) -> list[dict]:
    """
    bbox: (south, west, north, east)
    Возвращает ways с геометрией (список lat/lon).
    """
    south, west, north, east = bbox
    q = f"""
    [out:json][timeout:25];
    (
      way["highway"="path"]["sac_scale"]({south},{west},{north},{east});
      way["highway"="footway"]({south},{west},{north},{east});
    );
    out geom;
    """
    r = requests.post(OVERPASS_URL, data=q.encode("utf-8"), headers={"Content-Type": "text/plain"})
    r.raise_for_status()
    data = r.json()

    ways = []
    for el in data.get("elements", []):
        if el.get("type") != "way":
            continue
        geom = el.get("geometry") or []
        if len(geom) < 30:
            continue
        tags = el.get("tags", {})
        ways.append(
            {
                "osm_id": el.get("id"),
                "tags": tags,
                "geometry": geom,
            }
        )
        if len(ways) >= limit:
            break

    # небольшой троттлинг
    time.sleep(1.0)
    return ways
