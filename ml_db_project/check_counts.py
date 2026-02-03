from shared.db import get_session
from shared.models import Track, TrackPoint, WeatherPoint

s = get_session()
print("tracks:", s.query(Track).count())
print("track_points:", s.query(TrackPoint).count())
print("weather_points:", s.query(WeatherPoint).count())
