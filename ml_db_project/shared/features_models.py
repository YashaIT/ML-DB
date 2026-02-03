from __future__ import annotations

from datetime import datetime
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy import Integer, Float, String, DateTime, ForeignKey

from shared.models import Base

class TrackFeatures(Base):
    __tablename__ = "track_features"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    track_id: Mapped[int] = mapped_column(ForeignKey("tracks.id"), unique=True)

    # агрегаты
    length_km: Mapped[float | None] = mapped_column(Float)
    points_count: Mapped[int] = mapped_column(Integer)

    # тип местности
    area_type: Mapped[str] = mapped_column(String(32))  # urban | forest | water | mixed

    # плотность объектов
    road_density: Mapped[float | None] = mapped_column(Float)
    building_density: Mapped[float | None] = mapped_column(Float)
    water_density: Mapped[float | None] = mapped_column(Float)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
