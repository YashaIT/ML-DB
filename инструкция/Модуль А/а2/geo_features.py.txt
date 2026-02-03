from __future__ import annotations

import math

def haversine_m(lat1, lon1, lat2, lon2) -> float:
    R = 6371000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dl/2)**2
    return 2 * R * math.asin(math.sqrt(a))

def slope_deg(prev_point, next_point) -> float | None:
    if prev_point is None or next_point is None:
        return None
    if prev_point.ele is None or next_point.ele is None:
        return None
    d = haversine_m(prev_point.lat, prev_point.lon, next_point.lat, next_point.lon)
    if d < 1.0:
        return None
    dh = next_point.ele - prev_point.ele
    return math.degrees(math.atan2(dh, d))

def green_index_from_rgb(img_rgb) -> float:
    """
    Прокси “зелени” по RGB (не настоящий NDVI).
    img_rgb: numpy array HxWx3 uint8
    """
    import numpy as np
    rgb = img_rgb.astype(np.float32) / 255.0
    r = rgb[..., 0]
    g = rgb[..., 1]
    b = rgb[..., 2]
    exg = 2*g - r - b  # Excess Green
    return float(np.clip(exg.mean(), -1.0, 1.0))
