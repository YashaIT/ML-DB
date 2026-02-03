from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is in PYTHONPATH (so "shared" imports work under uvicorn)
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json
import math
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional, Tuple, List

import joblib
import redis.asyncio as redis
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.openapi.utils import get_openapi
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import jwt
from pydantic import BaseModel, Field
from sqlalchemy import text
from starlette.concurrency import run_in_threadpool

from shared.db import get_session
from shared.logging import setup_logger

logger = setup_logger("agent_g1_api")


# ----------------------------
# secrets/config
# ----------------------------
def load_secrets() -> Dict[str, Any]:
    p = ROOT / "config" / "secrets.json"
    if not p.exists():
        raise RuntimeError(f"Missing config/secrets.json at: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


SECRETS = load_secrets()
API_CFG = SECRETS.get("api", {})

JWT_SECRET = API_CFG.get("jwt_secret", "")
JWT_ISSUER = API_CFG.get("jwt_issuer", "ml_db_project")
JWT_EXP_MIN = int(API_CFG.get("jwt_exp_minutes", 240))
REDIS_URL = API_CFG.get("redis_url", "redis://localhost:6379/0")
MODEL_PATH = API_CFG.get("model_path", "agents/v4_predict/artifacts/model_rf.pkl")

USERS = {u["username"]: u for u in API_CFG.get("users", [])}

if not JWT_SECRET or len(JWT_SECRET) < 16:
    raise RuntimeError("api.jwt_secret is missing or too short in config/secrets.json")


# ----------------------------
# FastAPI app
# ----------------------------
app = FastAPI(
    title="Integration Agent API (g1)",
    version="1.0.0",
    description="FastAPI integration agent: saved ML model + forecasts (SQLite + Redis cache + JWT).",
)


# ----------------------------
# Auth (JWT)
# ----------------------------
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")


def create_access_token(username: str, role: str) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": username,
        "role": role,
        "iss": JWT_ISSUER,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=JWT_EXP_MIN)).timestamp()),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm="HS256")


def decode_token(token: str) -> Dict[str, Any]:
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=["HS256"], issuer=JWT_ISSUER)
    except Exception:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")


def get_current_user(token: str = Depends(oauth2_scheme)) -> Dict[str, Any]:
    data = decode_token(token)
    username = data.get("sub")
    if not username or username not in USERS:
        raise HTTPException(status_code=401, detail="Unknown user")
    return {"username": username, "role": data.get("role", "user")}


def require_role(*roles: str):
    def _dep(user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
        if user["role"] not in roles:
            raise HTTPException(status_code=403, detail="Forbidden")
        return user

    return _dep


# ----------------------------
# Redis cache
# ----------------------------
redis_client: Optional[redis.Redis] = None


async def get_redis() -> Optional[redis.Redis]:
    global redis_client
    if redis_client is None:
        try:
            redis_client = redis.from_url(REDIS_URL, decode_responses=True)
            await redis_client.ping()
            logger.info("Redis connected")
        except Exception as e:
            logger.warning(f"Redis unavailable: {e}")
            redis_client = None
    return redis_client


def cache_key(prefix: str, payload: Dict[str, Any]) -> str:
    items = "&".join([f"{k}={payload[k]}" for k in sorted(payload.keys())])
    return f"{prefix}:{items}"


async def cache_get(key: str) -> Optional[Dict[str, Any]]:
    r = await get_redis()
    if r is None:
        return None
    val = await r.get(key)
    if not val:
        return None
    try:
        return json.loads(val)
    except Exception:
        return None


async def cache_set(key: str, value: Dict[str, Any], ttl_sec: int = 300) -> None:
    r = await get_redis()
    if r is None:
        return
    await r.set(key, json.dumps(value, ensure_ascii=False), ex=ttl_sec)


# ----------------------------
# Model loading (NO training)
# ----------------------------
_model: Any = None


def load_model() -> Any:
    global _model
    if _model is None:
        p = ROOT / MODEL_PATH
        if not p.exists():
            raise RuntimeError(f"Model not found: {p}")
        _model = joblib.load(p)
        logger.info(f"Model loaded from: {p}")
    return _model


# ----------------------------
# Helpers
# ----------------------------
def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def parse_dt(s: str) -> datetime:
    dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def risk_from_features(
    slope_deg: float,
    precipitation_mm: float,
    wind_ms: float,
    frac_water: float,
    dist_to_settlement_m: float,
    frac_road: float,
) -> Tuple[float, str]:
    """
    Небольшая эвристика для "опасности" по локальным признакам.
    Это НЕ обучение. Модель используется отдельно (для label/hint).
    """
    s_slope = clip01(abs(slope_deg) / 25.0)
    s_prec = clip01(precipitation_mm / 20.0)
    s_wind = clip01(wind_ms / 15.0)
    s_water = clip01(frac_water)
    s_dist = clip01(dist_to_settlement_m / 5000.0)
    s_road = 1.0 - clip01(frac_road)

    score = (
        0.30 * s_slope
        + 0.25 * s_prec
        + 0.15 * s_wind
        + 0.15 * s_dist
        + 0.10 * s_water
        + 0.05 * s_road
    )
    score = clip01(score)

    if score < 0.33:
        level = "low"
    elif score < 0.66:
        level = "medium"
    else:
        level = "high"
    return score, level


def fetch_nearest_dataset_point(
    lat: float,
    lon: float,
    dt_utc: datetime,
    radius_m: int,
) -> Optional[Dict[str, Any]]:
    """
    Берём ближайшую точку из dataset_points (по расстоянию и близости по времени).
    Это быстрый прокси для "опасности по координатам+дате".
    """
    sess = get_session()
    ddeg = radius_m / 111000.0

    q = """
    SELECT
      dp.track_id,
      dp.track_point_id,
      dp.ts,
      dp.lat,
      dp.lon,
      dp.ele,
      dp.slope_deg,
      dp.dist_to_settlement_m,
      dp.precipitation_mm,
      dp.wind_ms,
      dp.frac_water,
      dp.frac_road,
      t.region
    FROM dataset_points dp
    JOIN tracks t ON t.id = dp.track_id
    WHERE
      dp.lat BETWEEN :lat_min AND :lat_max
      AND dp.lon BETWEEN :lon_min AND :lon_max
    LIMIT 2000
    """
    rows = sess.execute(
        text(q),
        {
            "lat_min": lat - ddeg,
            "lat_max": lat + ddeg,
            "lon_min": lon - ddeg,
            "lon_max": lon + ddeg,
        },
    ).mappings().all()

    if not rows:
        return None

    best: Optional[Dict[str, Any]] = None
    best_score: Optional[float] = None

    for r in rows:
        rlat = float(r["lat"])
        rlon = float(r["lon"])
        d = haversine_m(lat, lon, rlat, rlon)
        if d > radius_m:
            continue

        ts = r["ts"]
        if ts is not None:
            try:
                ts_dt = parse_dt(str(ts))
                dt_diff = abs((ts_dt - dt_utc).total_seconds())
            except Exception:
                dt_diff = 0.0
        else:
            dt_diff = 0.0

        score = d + 0.05 * dt_diff
        if best_score is None or score < best_score:
            best_score = score
            best = dict(r)

    return best


def model_predict_label_from_track(track_id: int) -> Optional[str]:
    """
    Используем сохранённую модель (никакого обучения):
    агрегируем признаки по треку и делаем model.predict().
    """
    sess = get_session()
    q = """
    SELECT
      dp.track_id,
      COUNT(*) * 0.02 AS length_km,
      AVG(dp.ele) AS mean_ele,
      AVG(dp.slope_deg) AS mean_slope,
      AVG(dp.green_index) AS mean_green,
      AVG(dp.frac_vegetation) AS frac_forest,
      AVG(dp.frac_water) AS frac_water,
      AVG(dp.frac_road) AS frac_road,
      AVG(dp.frac_building) AS frac_building
    FROM dataset_points dp
    WHERE dp.track_id = :track_id
    GROUP BY dp.track_id
    """
    row = sess.execute(text(q), {"track_id": track_id}).mappings().first()
    if not row:
        return None

    features = [
        float(row["length_km"] or 0.0),
        float(row["mean_ele"] or 0.0),
        float(row["mean_slope"] or 0.0),
        float(row["mean_green"] or 0.0),
        float(row["frac_forest"] or 0.0),
        float(row["frac_water"] or 0.0),
        float(row["frac_road"] or 0.0),
        float(row["frac_building"] or 0.0),
    ]

    model = load_model()
    try:
        pred = model.predict([features])[0]
        return str(pred)
    except Exception as e:
        logger.warning(f"Model predict failed: {e}")
        return None


def label_to_danger(label: Optional[str]) -> Optional[Dict[str, Any]]:
    if label is None:
        return None
    mapping = {
        "mixed_route": {"danger_level": "medium", "comment": "mixed conditions"},
        "forest_route": {"danger_level": "high", "comment": "vegetation-dense area"},
        "water_route": {"danger_level": "high", "comment": "near water / flood risk"},
        "urban_route": {"danger_level": "low", "comment": "urban / access better"},
    }
    return mapping.get(label, {"danger_level": "medium", "comment": "default mapping"})


# ----------------------------
# Schemas
# ----------------------------
class DangerResponse(BaseModel):
    lat: float
    lon: float
    date_utc: str
    radius_m: int
    risk_score: float
    risk_level: str
    region: Optional[str] = None
    model_label: Optional[str] = None
    model_hint: Optional[Dict[str, Any]] = None
    source_track_id: Optional[int] = None
    source_track_point_id: Optional[int] = None


class ForecastRequest(BaseModel):
    lat: float
    lon: float
    start_date_utc: str
    end_date_utc: str
    radius_m: int = Field(default=500, ge=200, le=2000)
    step_hours: int = Field(default=24, ge=1, le=168)


class ForecastItem(BaseModel):
    date_utc: str
    risk_score: float
    risk_level: str


class ForecastResponse(BaseModel):
    lat: float
    lon: float
    radius_m: int
    horizon: int
    items: List[ForecastItem]
    region: Optional[str] = None


class EvacuationResponse(BaseModel):
    lat: float
    lon: float
    date_utc: str
    radius_m: int
    difficulty_score: float
    difficulty_level: str
    factors: Dict[str, float]
    region: Optional[str] = None
    model_label: Optional[str] = None


# ----------------------------
# Startup: Redis ping + OpenAPI file artifact
# ----------------------------
@app.on_event("startup")
async def _startup() -> None:
    await get_redis()

    # Save OpenAPI schema to file as an "artifact" for the assignment
    try:
        artifacts_dir = ROOT / "agents" / "g1_api" / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        schema = get_openapi(
            title=app.title,
            version=app.version,
            description=app.description,
            routes=app.routes,
        )
        (artifacts_dir / "openapi.json").write_text(
            json.dumps(schema, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info(f"OpenAPI saved to: {artifacts_dir / 'openapi.json'}")
    except Exception as e:
        logger.warning(f"OpenAPI export failed: {e}")


# ----------------------------
# Routes
# ----------------------------
@app.get("/health")
async def health() -> Dict[str, Any]:
    r = await get_redis()
    return {"status": "ok", "redis": bool(r is not None), "model_path": MODEL_PATH}


@app.post("/auth/token")
async def token(form: OAuth2PasswordRequestForm = Depends()) -> Dict[str, Any]:
    """
    Получение JWT токена.
    В swagger сначала вызываешь этот endpoint, копируешь access_token и жмёшь Authorize.
    """
    user = USERS.get(form.username)
    if not user or user.get("password") != form.password:
        raise HTTPException(status_code=401, detail="Bad credentials")

    tok = create_access_token(user["username"], user.get("role", "user"))
    return {"access_token": tok, "token_type": "bearer", "role": user.get("role", "user")}


@app.get("/danger", response_model=DangerResponse)
async def danger(
    lat: float,
    lon: float,
    date_utc: str,
    radius_m: int = 500,
    _user: Dict[str, Any] = Depends(require_role("user", "admin")),
) -> DangerResponse:
    """
    Определение уровня опасности для координат и даты.
    """
    if radius_m < 200 or radius_m > 2000:
        raise HTTPException(status_code=400, detail="radius_m must be in 200..2000")

    dt = parse_dt(date_utc)

    key = cache_key(
        "danger",
        {"lat": round(lat, 6), "lon": round(lon, 6), "date": dt.isoformat(), "r": radius_m},
    )
    cached = await cache_get(key)
    if cached:
        return DangerResponse(**cached)

    nearest = await run_in_threadpool(fetch_nearest_dataset_point, lat, lon, dt, radius_m)
    if nearest is None:
        raise HTTPException(status_code=404, detail="No nearby data points found")

    slope = float(nearest.get("slope_deg") or 0.0)
    prec = float(nearest.get("precipitation_mm") or 0.0)
    wind = float(nearest.get("wind_ms") or 0.0)
    water = float(nearest.get("frac_water") or 0.0)
    dist = float(nearest.get("dist_to_settlement_m") or 0.0)
    road = float(nearest.get("frac_road") or 0.0)

    risk_score, risk_level = risk_from_features(slope, prec, wind, water, dist, road)

    track_id = int(nearest["track_id"])
    model_label = await run_in_threadpool(model_predict_label_from_track, track_id)
    hint = label_to_danger(model_label)

    resp = DangerResponse(
        lat=lat,
        lon=lon,
        date_utc=dt.isoformat(),
        radius_m=radius_m,
        risk_score=risk_score,
        risk_level=risk_level,
        region=str(nearest.get("region")) if nearest.get("region") is not None else None,
        model_label=model_label,
        model_hint=hint,
        source_track_id=track_id,
        source_track_point_id=int(nearest["track_point_id"]),
    )

    await cache_set(key, resp.model_dump(), ttl_sec=300)
    return resp


@app.post("/forecast", response_model=ForecastResponse)
async def forecast(
    req: ForecastRequest,
    _user: Dict[str, Any] = Depends(require_role("user", "admin")),
) -> ForecastResponse:
    """
    Прогноз опасности на заданный период.
    """
    dt1 = parse_dt(req.start_date_utc)
    dt2 = parse_dt(req.end_date_utc)
    if dt2 <= dt1:
        raise HTTPException(status_code=400, detail="end_date_utc must be > start_date_utc")

    key = cache_key(
        "forecast",
        {
            "lat": round(req.lat, 6),
            "lon": round(req.lon, 6),
            "s": dt1.isoformat(),
            "e": dt2.isoformat(),
            "r": req.radius_m,
            "step_h": req.step_hours,
        },
    )
    cached = await cache_get(key)
    if cached:
        return ForecastResponse(**cached)

    nearest = await run_in_threadpool(fetch_nearest_dataset_point, req.lat, req.lon, dt1, req.radius_m)
    if nearest is None:
        raise HTTPException(status_code=404, detail="No nearby data points found")

    region = str(nearest.get("region")) if nearest.get("region") is not None else None

    items: List[ForecastItem] = []
    t = dt1
    step = timedelta(hours=req.step_hours)

    slope = float(nearest.get("slope_deg") or 0.0)
    water = float(nearest.get("frac_water") or 0.0)
    dist = float(nearest.get("dist_to_settlement_m") or 0.0)
    road = float(nearest.get("frac_road") or 0.0)

    base_prec = float(nearest.get("precipitation_mm") or 0.0)
    base_wind = float(nearest.get("wind_ms") or 0.0)

    # лёгкая динамика по времени (чтобы графики были "живые")
    while t <= dt2:
        h = (t - dt1).total_seconds() / 3600.0
        prec = max(0.0, base_prec * (0.7 + 0.3 * math.sin(h / 24.0 * 2 * math.pi)))
        wind = max(0.0, base_wind * (0.8 + 0.2 * math.cos(h / 24.0 * 2 * math.pi)))

        rs, rl = risk_from_features(slope, prec, wind, water, dist, road)
        items.append(ForecastItem(date_utc=t.isoformat(), risk_score=rs, risk_level=rl))
        t += step

    resp = ForecastResponse(
        lat=req.lat,
        lon=req.lon,
        radius_m=req.radius_m,
        horizon=len(items),
        items=items,
        region=region,
    )

    await cache_set(key, resp.model_dump(), ttl_sec=300)
    return resp


@app.get("/evacuation", response_model=EvacuationResponse)
async def evacuation(
    lat: float,
    lon: float,
    date_utc: str,
    radius_m: int = 500,
    _user: Dict[str, Any] = Depends(require_role("admin")),
) -> EvacuationResponse:
    """
    Оценка сложности эвакуации.
    Доступ: только admin (пример разграничения прав).
    """
    dt = parse_dt(date_utc)

    key = cache_key(
        "evac",
        {"lat": round(lat, 6), "lon": round(lon, 6), "date": dt.isoformat(), "r": radius_m},
    )
    cached = await cache_get(key)
    if cached:
        return EvacuationResponse(**cached)

    nearest = await run_in_threadpool(fetch_nearest_dataset_point, lat, lon, dt, radius_m)
    if nearest is None:
        raise HTTPException(status_code=404, detail="No nearby data points found")

    slope = float(nearest.get("slope_deg") or 0.0)
    road = float(nearest.get("frac_road") or 0.0)
    dist = float(nearest.get("dist_to_settlement_m") or 0.0)
    water = float(nearest.get("frac_water") or 0.0)

    s_slope = clip01(abs(slope) / 25.0)
    s_dist = clip01(dist / 5000.0)
    s_road = 1.0 - clip01(road)
    s_water = clip01(water)

    difficulty = (0.45 * s_dist + 0.35 * s_road + 0.15 * s_slope + 0.05 * s_water)
    difficulty = clip01(difficulty)

    if difficulty < 0.33:
        level = "easy"
    elif difficulty < 0.66:
        level = "medium"
    else:
        level = "hard"

    track_id = int(nearest["track_id"])
    model_label = await run_in_threadpool(model_predict_label_from_track, track_id)

    resp = EvacuationResponse(
        lat=lat,
        lon=lon,
        date_utc=dt.isoformat(),
        radius_m=radius_m,
        difficulty_score=difficulty,
        difficulty_level=level,
        factors={
            "dist_to_settlement_m": float(dist),
            "frac_road": float(road),
            "slope_deg": float(slope),
            "frac_water": float(water),
        },
        region=str(nearest.get("region")) if nearest.get("region") is not None else None,
        model_label=model_label,
    )

    await cache_set(key, resp.model_dump(), ttl_sec=300)
    return resp
