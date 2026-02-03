from __future__ import annotations

from pathlib import Path
import requests

def download_tile(url_template: str, z: int, x: int, y: int, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    url = url_template.format(z=z, x=x, y=y)
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    out_path.write_bytes(r.content)

def tile_path(kind: str, z: int, x: int, y: int) -> Path:
    ext = "png" if kind == "topo" else "jpg"
    return Path("data/images") / kind / str(z) / str(x) / f"{y}.{ext}"
