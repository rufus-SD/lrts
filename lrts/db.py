from __future__ import annotations

from pathlib import Path

from sqlmodel import SQLModel
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

from lrts.config import DATABASE_URL, DB_PATH

_engine = None
_async_session = None


def _get_engine():
    global _engine
    if _engine is None:
        Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
        _engine = create_async_engine(DATABASE_URL, echo=False)
    return _engine


def _get_session_factory():
    global _async_session
    if _async_session is None:
        _async_session = sessionmaker(
            _get_engine(), class_=AsyncSession, expire_on_commit=False
        )
    return _async_session


def use_local_db(directory: Path) -> None:
    """Switch to a project-local DB at <directory>/.lrts/lrts.db."""
    global _engine, _async_session
    db_dir = directory / ".lrts"
    db_dir.mkdir(exist_ok=True)
    db_path = db_dir / "lrts.db"
    url = f"sqlite+aiosqlite:///{db_path}"
    _engine = create_async_engine(url, echo=False)
    _async_session = sessionmaker(
        _engine, class_=AsyncSession, expire_on_commit=False
    )


class _SessionProxy:
    """Proxy that always uses the current session factory."""

    def __call__(self):
        return _get_session_factory()()


async_session = _SessionProxy()


async def init_db() -> None:
    async with _get_engine().begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)


async def get_session() -> AsyncSession:  # type: ignore[misc]
    async with _get_session_factory()() as session:
        yield session
