"""Persistencia común local/Neon; PostgreSQL obligatorio en producción."""
import os
from pathlib import Path
from sqlalchemy import create_engine, Float, Integer, String, DateTime, func, Index
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

ROOT=Path(__file__).resolve().parents[1]
url=os.getenv('DATABASE_URL')
if not url:
    if os.getenv('APP_ENV','development').lower()=='production':
        raise RuntimeError('DATABASE_URL es obligatorio en producción (Neon PostgreSQL)')
    url=f'sqlite:///{ROOT / "local-dev.sqlite3"}'
if url.startswith('postgres://'):
    url='postgresql'+url[len('postgres'):]
if url.startswith('postgresql://'):
    url='postgresql+psycopg://'+url[len('postgresql://'):]
kwargs={'pool_pre_ping':True}
if url.startswith('sqlite:'):
    kwargs['connect_args']={'check_same_thread':False}
engine=create_engine(url,**kwargs)
SessionLocal=sessionmaker(bind=engine,autoflush=False,autocommit=False)

class Base(DeclarativeBase): pass

class TelemetryRecord(Base):
    __tablename__='telemetry'
    id: Mapped[int]=mapped_column(Integer,primary_key=True,autoincrement=True)
    robot_id: Mapped[str]=mapped_column(String(20),nullable=False,index=True)
    timestamp: Mapped[float]=mapped_column(Float,nullable=False)
    x: Mapped[float]=mapped_column(Float,nullable=False)
    y: Mapped[float]=mapped_column(Float,nullable=False)
    speed: Mapped[float]=mapped_column(Float,nullable=False)
    battery: Mapped[float]=mapped_column(Float,nullable=False)
    motor_temp: Mapped[float]=mapped_column(Float,nullable=False)
    vibration: Mapped[float]=mapped_column(Float,nullable=False)
    motor_current: Mapped[float]=mapped_column(Float,nullable=False)
    obstacle_distance: Mapped[float]=mapped_column(Float,nullable=False)
    predicted_status: Mapped[str | None]=mapped_column(String(10),nullable=True)
    confidence: Mapped[float | None]=mapped_column(Float,nullable=True)
    received_at: Mapped[str]=mapped_column(DateTime(timezone=True),server_default=func.now())

Index('idx_telemetry_robot_pk',TelemetryRecord.robot_id,TelemetryRecord.id)
