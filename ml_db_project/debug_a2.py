from shared.db import get_session
from shared.models import TrackPoint, Track, WeatherPoint
from agents.a2_features.services.context_from_tiles import extract_context

s = get_session()

tp = s.query(TrackPoint).first()
t = s.query(Track).filter(Track.id == tp.track_id).one()

print("track_id:", t.id, "region:", t.region, "started_at:", t.started_at)
print("point:", tp.id, tp.lat, tp.lon, "ts:", tp.ts)

wp_cnt = s.query(WeatherPoint).count()
print("weather_points:", wp_cnt)

ctx = extract_context(tp.lat, tp.lon, radius_m=500)
print("ctx:", ctx)
