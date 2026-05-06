from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from config import settings

engine = create_engine(
    settings.DATABASE_URL,
    pool_pre_ping=True,
    pool_recycle=3600,
    pool_size=10,
    max_overflow=20,
)

SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


def get_db():
    """FastAPI dependency - cung cấp DB session."""
    db: Session = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def fetch_df(query: str, params: dict = None):
    """Tiện ích đọc dữ liệu thẳng ra pandas DataFrame."""
    import pandas as pd
    with engine.connect() as conn:
        return pd.read_sql(text(query), conn, params=params)
