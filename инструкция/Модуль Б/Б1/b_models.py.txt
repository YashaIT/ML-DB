from __future__ import annotations

from datetime import datetime
from sqlalchemy import String, Integer, Float, DateTime, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column

from shared.models import Base


class Alert(Base):
    __tablename__ = "alerts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    track_id: Mapped[int] = mapped_column(Integer, index=True)
    track_point_id: Mapped[int | None] = mapped_column(Integer, ForeignKey("track_points.id"), nullable=True)

    ts: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    kind: Mapped[str] = mapped_column(String(64), index=True)  # speed_drop, etc
    severity: Mapped[int] = mapped_column(Integer, default=1)  # 1..5

    value: Mapped[float | None] = mapped_column(Float, nullable=True)
    threshold: Mapped[float | None] = mapped_column(Float, nullable=True)

    message: Mapped[str] = mapped_column(String(512), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
