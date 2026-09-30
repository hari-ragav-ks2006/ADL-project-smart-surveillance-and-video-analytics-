"""
Database Schema and SQLAlchemy Session Management for Surveillance Logs.
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Generator, Optional
import json

def get_utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)

from sqlalchemy import (
    Column,
    DateTime,
    Integer,
    String,
    Text,
    create_engine,
    desc,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker


class Base(DeclarativeBase):
    pass


class AlertRecord(Base):
    """Stores every triggered surveillance alert."""

    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=get_utc_now, index=True)
    alert_type = Column(String(64), nullable=False, index=True)
    severity = Column(String(16), nullable=False, index=True)  # LOW, MEDIUM, HIGH, CRITICAL
    module_name = Column(String(64), nullable=False)
    zone_name = Column(String(64), nullable=True)
    tracked_ids = Column(String(256), default="[]")  # JSON list of int IDs
    message = Column(String(512), nullable=False)
    resolution_status = Column(String(16), default="ACTIVE", index=True)  # ACTIVE, ESCALATED, RESOLVED
    resolved_at = Column(DateTime, nullable=True)
    screenshot_path = Column(String(512), nullable=True)
    details_json = Column(Text, default="{}")

    def to_dict(self):
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "alert_type": self.alert_type,
            "severity": self.severity,
            "module_name": self.module_name,
            "zone_name": self.zone_name,
            "tracked_ids": json.loads(self.tracked_ids or "[]"),
            "message": self.message,
            "resolution_status": self.resolution_status,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "screenshot_path": self.screenshot_path,
            "details": json.loads(self.details_json or "{}"),
        }


class SystemLogRecord(Base):
    """Stores system operational events and dispatches."""

    __tablename__ = "system_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=get_utc_now, index=True)
    event_type = Column(String(64), nullable=False)
    message = Column(String(512), nullable=False)

    def to_dict(self):
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "event_type": self.event_type,
            "message": self.message,
        }


class DatabaseManager:
    """Manages SQLite database connections and table lifecycle."""

    def __init__(self, db_path: str = "logs/surveillance.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db_url = f"sqlite:///{self.db_path.as_posix()}"
        self.engine = create_engine(
            self.db_url,
            connect_args={"check_same_thread": False},
            pool_pre_ping=True,
        )
        self.SessionFactory = sessionmaker(
            autocommit=False, autoflush=False, bind=self.engine
        )
        self.init_db()

    def init_db(self):
        """Create all tables if they do not exist."""
        Base.metadata.create_all(bind=self.engine)

    def get_session(self) -> Session:
        """Provide a new SQLAlchemy session."""
        return self.SessionFactory()


# Helper factory
def get_db_manager(db_path: str = "logs/surveillance.db") -> DatabaseManager:
    return DatabaseManager(db_path=db_path)
