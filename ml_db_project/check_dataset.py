from shared.db import get_session
from shared.dataset_models import DatasetPoint

s = get_session()
cnt = s.query(DatasetPoint).count()
print("dataset_points:", cnt)

row = s.query(DatasetPoint).first()
print("first:", None if row is None else {
    "track_id": row.track_id,
    "ts": row.ts,
    "lat": row.lat,
    "lon": row.lon,
    "ele": row.ele,
    "radius_m": row.radius_m,
    "slope_deg": row.slope_deg,
    "green_index": row.green_index,
    "land_type": row.land_type,
    "objects_json": row.objects_json,
    "temp_c": row.temp_c,
})
