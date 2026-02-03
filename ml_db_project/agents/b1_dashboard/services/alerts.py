from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
from sqlalchemy import text

from shared.db import engine, get_session
from shared.models import Base
from shared.b_models import Alert


def ensure_db() -> None:
    Base.metadata.create_all(bind=engine)


def recompute_speed_drop_alerts(
    track_id: int,
    speed_df: pd.DataFrame,
    threshold_ms: float,
    min_segments: int = 3,
) -> int:
    """
    Пишем алерты speed_drop, если скорость ниже threshold_ms.
    """
    ensure_db()
    sess = get_session()

    if speed_df.empty or "speed_ms" not in speed_df.columns:
        return 0

    sdf = speed_df.dropna(subset=["speed_ms"]).copy()
    if len(sdf) < min_segments:
        return 0

    # удаляем старые алерты этого типа по треку (чтобы кнопка "пересчитать" была идемпотентной)
    sess.execute(
        text("DELETE FROM alerts WHERE track_id = :tid AND kind = 'speed_drop'"),
        {"tid": track_id},
    )
    sess.commit()

    inserted = 0
    for _, row in sdf.iterrows():
        v = float(row["speed_ms"])
        if v < threshold_ms:
            a = Alert(
                track_id=int(row["track_id"]),
                track_point_id=int(row["track_point_id"]) if pd.notna(row["track_point_id"]) else None,
                ts=row["ts"],
                kind="speed_drop",
                severity=2 if v < threshold_ms * 0.5 else 1,
                value=v,
                threshold=threshold_ms,
                message=f"Speed drop: {v:.2f} m/s < {threshold_ms:.2f} m/s",
                created_at=datetime.now(timezone.utc),
            )
            sess.add(a)
            inserted += 1

    sess.commit()
    return inserted


def load_alerts(track_id: int | None = None) -> pd.DataFrame:
    sess = get_session()
    q = "SELECT id, track_id, track_point_id, ts, kind, severity, value, threshold, message, created_at FROM alerts"
    params = {}
    if track_id is not None:
        q += " WHERE track_id = :tid"
        params["tid"] = track_id
    q += " ORDER BY created_at DESC LIMIT 200"
    return pd.read_sql(text(q), sess.bind, params=params)
