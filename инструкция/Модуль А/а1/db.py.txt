from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from shared.config import settings

Path("data").mkdir(parents=True, exist_ok=True)

engine = create_engine(f"sqlite:///{settings.sqlite_path}", echo=False, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)

def get_session():
    return SessionLocal()
