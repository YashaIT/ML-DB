from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import json


from shared.config import settings
from shared.utils import deg2num
from agents.a1_ingest.services.imagery import download_tile

LAND_TYPES = {
    "urban",
    "forest",
    "water",
    "road_area",
    "mixed",
    "unknown",
    "river_or_lake",
}

@dataclass
class ContextResult:
    land_type: str
    objects: list[str]
    frac_road: float | None
    frac_building: float | None
    frac_water: float | None
    frac_vegetation: float | None
    green_index: float | None

def _tile_url(kind: str) -> str:
    return settings.topo_tile_url if kind == "topo" else settings.sat_tile_url

def _tile_ext(kind: str) -> str:
    return "png" if kind == "topo" else "jpg"

def _tile_path(kind: str, z: int, x: int, y: int) -> Path:
    ext = _tile_ext(kind)
    return Path("data/images") / kind / str(z) / str(x) / f"{y}.{ext}"

def _meters_per_pixel(lat: float, zoom: int) -> float:
    # приближение
    return 156543.03392 * math.cos(math.radians(lat)) / (2 ** zoom)

def _read_image(path: Path):
    from PIL import Image
    import numpy as np
    img = Image.open(path).convert("RGB")
    return np.array(img)

def extract_context(lat: float, lon: float, radius_m: int, zoom: int | None = None) -> ContextResult:
    """
    Базовый анализ окружения:
    - качаем несколько тайлов topo + sat вокруг точки
    - считаем прокси-доли классов по простым порогам RGB
    """
    zoom = zoom or settings.tile_zoom
    x0, y0 = deg2num(lat, lon, zoom)

    mpp = _meters_per_pixel(lat, zoom)
    px_radius = max(1, int(radius_m / mpp))
    # тайл 256px, берём окружение до 1 тайла в каждую сторону в большинстве случаев
    tile_radius = max(1, int(px_radius / 256) + 1)

    sat_imgs = []
    topo_imgs = []

    for dx in range(-tile_radius, tile_radius + 1):
        for dy in range(-tile_radius, tile_radius + 1):
            x = x0 + dx
            y = y0 + dy

            for kind, store in [("sat", sat_imgs), ("topo", topo_imgs)]:
                path = _tile_path(kind, zoom, x, y)
                if not path.exists():
                    try:
                        download_tile(_tile_url(kind), zoom, x, y, path)
                    except Exception:
                        continue
                try:
                    store.append(_read_image(path))
                except Exception:
                    continue

    # если нет картинок — возвращаем пусто
    if not sat_imgs:
        return ContextResult(
            land_type="unknown",
            objects=[],
            frac_road=None, frac_building=None, frac_water=None, frac_vegetation=None,
            green_index=None,
        )

    # объединяем статистику по всем sat тайлам
    import numpy as np
    all_pixels = np.concatenate([img.reshape(-1, 3) for img in sat_imgs], axis=0).astype(np.float32) / 255.0
    r, g, b = all_pixels[:, 0], all_pixels[:, 1], all_pixels[:, 2]

    # очень простые маски:
    water = (b > 0.45) & (b > r + 0.05) & (b > g + 0.05)
    vegetation = (g > 0.40) & (g > r + 0.05) & (g > b + 0.05)
    bright = (r + g + b) / 3.0
    gray = (abs(r - g) < 0.06) & (abs(r - b) < 0.06) & (abs(g - b) < 0.06)

    buildings = gray & (bright > 0.70) & (~vegetation) & (~water)
    roads = gray & (bright > 0.40) & (bright < 0.70) & (~vegetation) & (~water)

    n = float(all_pixels.shape[0])
    frac_water = float(water.sum() / n)
    frac_veg = float(vegetation.sum() / n)
    frac_build = float(buildings.sum() / n)
    frac_road = float(roads.sum() / n)

    # тип местности
    if frac_water > 0.20:
        land = "water"
    elif frac_veg > 0.35 and frac_build < 0.12:
        land = "forest"
    elif frac_build > 0.18:
        land = "urban"
    elif frac_road > 0.15:
        land = "road_area"
    else:
        land = "mixed"

    objs = []
    if frac_water > 0.03: objs.append("water")
    if frac_veg > 0.05: objs.append("vegetation")
    if frac_build > 0.03: objs.append("building")
    if frac_road > 0.04: objs.append("road")

    # green index (прокси)
    from agents.a2_features.services.geo_features import green_index_from_rgb
    green_idx = green_index_from_rgb(sat_imgs[0])

    if land not in LAND_TYPES:
        land = "unknown"

    return ContextResult(
        land_type=land,
        objects=objs,
        frac_road=frac_road,
        frac_building=frac_build,
        frac_water=frac_water,
        frac_vegetation=frac_veg,
        green_index=green_idx,
    )
