from __future__ import annotations

from datetime import datetime
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy import Integer, Float, String, DateTime, ForeignKey, Text

from shared.models import Base

class DatasetPoint(Base):
    __tablename__ = "dataset_points"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    track_id: Mapped[int] = mapped_column(ForeignKey("tracks.id"), index=True)
    track_point_id: Mapped[int] = mapped_column(ForeignKey("track_points.id"), unique=True, index=True)

    region: Mapped[str | None] = mapped_column(String(128), nullable=True)
    ts: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    lat: Mapped[float] = mapped_column(Float)
    lon: Mapped[float] = mapped_column(Float)
    ele: Mapped[float | None] = mapped_column(Float, nullable=True)

    radius_m: Mapped[int] = mapped_column(Integer)

    # производные признаки
    slope_deg: Mapped[float | None] = mapped_column(Float, nullable=True)
    dist_to_settlement_m: Mapped[float | None] = mapped_column(Float, nullable=True)
    green_index: Mapped[float | None] = mapped_column(Float, nullable=True)

    # погода (денормализуем для удобства ML)
    temp_c: Mapped[float | None] = mapped_column(Float, nullable=True)
    precipitation_mm: Mapped[float | None] = mapped_column(Float, nullable=True)
    humidity_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    wind_ms: Mapped[float | None] = mapped_column(Float, nullable=True)

    # результат окружения
    land_type: Mapped[str | None] = mapped_column(String(32), nullable=True)
    objects_json: Mapped[str | None] = mapped_column(Text, nullable=True)

    # доли классов (опционально)
    frac_road: Mapped[float | None] = mapped_column(Float, nullable=True)
    frac_building: Mapped[float | None] = mapped_column(Float, nullable=True)
    frac_water: Mapped[float | None] = mapped_column(Float, nullable=True)
    frac_vegetation: Mapped[float | None] = mapped_column(Float, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
