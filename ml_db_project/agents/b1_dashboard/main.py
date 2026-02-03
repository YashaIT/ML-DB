from __future__ import annotations

import os
import sys
import subprocess


def main() -> None:
    """
    Запуск одной командой:
      python -m agents.b1_dashboard.main
    """
    app_path = os.path.join("agents", "b1_dashboard", "app.py")
    cmd = [sys.executable, "-m", "streamlit", "run", app_path, "--server.headless=true"]
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
