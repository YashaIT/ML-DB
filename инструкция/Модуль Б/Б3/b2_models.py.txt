from __future__ import annotations

from datetime import datetime
from sqlalchemy import Integer, String, DateTime, Float
from sqlalchemy.orm import Mapped, mapped_column

from shared.models import Base


class TrackLabel(Base):
    __tablename__ = "track_labels"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    track_id: Mapped[int] = mapped_column(Integer, index=True)
    label: Mapped[str] = mapped_column(String(64), index=True)

    method: Mapped[str] = mapped_column(String(32), default="cluster")  # cluster | manual | active
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
