from __future__ import annotations

import pandas as pd

from shared.db import get_session
from shared.dataset_models import DatasetPoint


# какие поля считаем числовыми признаками
FEATURE_COLS = [
    "ele",
    "slope_deg",
    "dist_to_settlement_m",
    "green_index",
    "temp_c",
    "precipitation_mm",
    "humidity_pct",
    "wind_ms",
    "frac_road",
    "frac_building",
    "frac_water",
    "frac_vegetation",
]

META_COLS = [
    "id",
    "track_id",
    "track_point_id",
    "region",
    "ts",
    "lat",
    "lon",
    "radius_m",
    "land_type",
]


def load_dataset_points() -> pd.DataFrame:
    s = get_session()
    rows = s.query(DatasetPoint).all()
    if not rows:
        return pd.DataFrame()

    data = []
    for r in rows:
        d = {}
        for c in META_COLS + FEATURE_COLS:
            d[c] = getattr(r, c)
        data.append(d)

    df = pd.DataFrame(data)
    return df
