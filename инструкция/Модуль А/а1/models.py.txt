from __future__ import annotations

from datetime import datetime
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy import String, Integer, Float, DateTime, ForeignKey, UniqueConstraint, Text

class Base(DeclarativeBase):
    pass

class Track(Base):
    __tablename__ = "tracks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    source: Mapped[str] = mapped_column(String(32))
    source_ref: Mapped[str] = mapped_column(Text)

    name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    region: Mapped[str | None] = mapped_column(String(128), nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    points: Mapped[list["TrackPoint"]] = relationship(
        back_populates="track", cascade="all, delete-orphan"
    )

class TrackPoint(Base):
    __tablename__ = "track_points"
    __table_args__ = (UniqueConstraint("track_id", "seq", name="uq_track_point_seq"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    track_id: Mapped[int] = mapped_column(ForeignKey("tracks.id"), index=True)
    seq: Mapped[int] = mapped_column(Integer)

    lat: Mapped[float] = mapped_column(Float)
    lon: Mapped[float] = mapped_column(Float)
    ele: Mapped[float | None] = mapped_column(Float, nullable=True)
    ts: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    track: Mapped["Track"] = relationship(back_populates="points")
    weather: Mapped["WeatherPoint | None"] = relationship(
        back_populates="point", cascade="all, delete-orphan", uselist=False
    )

class WeatherPoint(Base):
    __tablename__ = "weather_points"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    track_point_id: Mapped[int] = mapped_column(ForeignKey("track_points.id"), unique=True)

    temp_c: Mapped[float | None] = mapped_column(Float, nullable=True)
    precipitation_mm: Mapped[float | None] = mapped_column(Float, nullable=True)
    humidity_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    wind_ms: Mapped[float | None] = mapped_column(Float, nullable=True)

    provider: Mapped[str] = mapped_column(String(64), default="open_meteo")
    fetched_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    point: Mapped["TrackPoint"] = relationship(back_populates="weather")

class TrackImage(Base):
    __tablename__ = "track_images"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    track_id: Mapped[int] = mapped_column(ForeignKey("tracks.id"), index=True)

    kind: Mapped[str] = mapped_column(String(16))  # topo|sat
    z: Mapped[int] = mapped_column(Integer)
    x: Mapped[int] = mapped_column(Integer)
    y: Mapped[int] = mapped_column(Integer)

    file_path: Mapped[str] = mapped_column(Text)
    fetched_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
