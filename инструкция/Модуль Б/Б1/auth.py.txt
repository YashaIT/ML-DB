from __future__ import annotations

import json
import hmac
from pathlib import Path


def _load_secrets() -> dict:
    """
    Читает config/secrets.json.
    Если файла нет — возвращает пустой словарь (тогда доступ будет закрыт).
    """
    path = Path("config") / "secrets.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def get_role(password: str | None) -> str:
    """
    roles:
      - admin: может пересчитывать алерты и смотреть расширенные блоки
      - viewer: read-only
    """
    secrets = _load_secrets()
    admin_pw = str(secrets.get("admin_password", "") or "")
    viewer_pw = str(secrets.get("viewer_password", "") or "")

    if password is None:
        return "none"

    if admin_pw and hmac.compare_digest(password, admin_pw):
        return "admin"

    if viewer_pw and hmac.compare_digest(password, viewer_pw):
        return "viewer"

    return "none"
