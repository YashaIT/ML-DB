from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class FieldDoc:
    name: str
    description: str
    unit: Optional[str] = None
    example: Optional[str] = None


def dataset_dictionary() -> list[FieldDoc]:
    return [
        FieldDoc("id", "Идентификатор записи в dataset_points", unit=None),
        FieldDoc("track_id", "Идентификатор трека", unit=None),
        FieldDoc("track_point_id", "Идентификатор точки трека", unit=None),
        FieldDoc("region", "Регион трека (задаётся при ingest)", unit=None),
        FieldDoc("ts", "Дата и время точки (UTC). Если в GPX нет времени — может быть NULL/подстановка started_at трека", unit="ISO 8601"),
        FieldDoc("lat", "Широта точки", unit="deg"),
        FieldDoc("lon", "Долгота точки", unit="deg"),
        FieldDoc("ele", "Высота точки", unit="m"),
        FieldDoc("radius_m", "Радиус анализа окружения точки", unit="m"),
        FieldDoc("slope_deg", "Крутизна склона (градиент высоты) по соседним точкам", unit="deg"),
        FieldDoc("dist_to_settlement_m", "Расстояние до ближайшего населённого пункта (если включено)", unit="m"),
        FieldDoc("green_index", "Индекс зелёных насаждений (прокси по RGB, т.к. NDVI требует NIR)", unit="0..1"),
        FieldDoc("temp_c", "Температура воздуха (историческая/приближённая по ближайшей точке)", unit="°C"),
        FieldDoc("precipitation_mm", "Осадки", unit="mm"),
        FieldDoc("humidity_pct", "Относительная влажность", unit="%"),
        FieldDoc("wind_ms", "Скорость ветра", unit="m/s"),
        FieldDoc("land_type", "Тип местности (категориальный признак)", unit=None),
        FieldDoc("objects_json", "Список ключевых объектов вокруг точки (дорога/вода/растительность/застройка)", unit="json list[str]"),
        FieldDoc("frac_road", "Доля пикселей, классифицированных как дорога в окрестности", unit="0..1"),
        FieldDoc("frac_building", "Доля пикселей застройки в окрестности", unit="0..1"),
        FieldDoc("frac_water", "Доля пикселей воды в окрестности", unit="0..1"),
        FieldDoc("frac_vegetation", "Доля пикселей растительности в окрестности", unit="0..1"),
    ]


def render_markdown_table(docs: list[FieldDoc]) -> str:
    lines = []
    lines.append("| Поле | Описание | Ед. изм. |")
    lines.append("|---|---|---|")
    for d in docs:
        unit = d.unit or ""
        lines.append(f"| `{d.name}` | {d.description} | {unit} |")
    return "\n".join(lines)
