"""Destructive, development-only reset for repeatable hackathon demos.

The reset is called by the Docker entrypoint only when
DEMO_RESET_DATABASE_ON_START=true. Never enable it for real patient data.
"""
from __future__ import annotations

import os
import shutil
import time
from pathlib import Path

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from app.core.config import settings
from app.db.session import Base, engine
from app.models import models  # noqa: F401 - registers model metadata


def main() -> None:
    enabled = os.getenv("DEMO_RESET_DATABASE_ON_START", "false").strip().casefold() == "true"
    environment = os.getenv("ENVIRONMENT", settings.environment).strip().casefold()
    if not enabled:
        print("Demo database reset disabled; keeping existing database and files.", flush=True)
        return
    if environment not in {"development", "test", "demo"}:
        raise RuntimeError("Refusing to reset the database unless ENVIRONMENT is development, test, or demo.")

    # Compose restart order/health timing can vary. Wait for PostgreSQL instead of exiting
    # immediately if it is still starting.
    last_error: Exception | None = None
    for _ in range(60):
        try:
            with engine.connect() as connection:
                connection.execute(text("SELECT 1"))
            last_error = None
            break
        except SQLAlchemyError as exc:
            last_error = exc
            time.sleep(1)
    if last_error is not None:
        raise RuntimeError("PostgreSQL did not become available for the demo reset.") from last_error

    print("DEMO MODE: wiping the public PostgreSQL schema and uploaded medical files for a clean session.", flush=True)
    # Drop schema instead of metadata tables so stale/mismatched prototype schemas are
    # removed consistently. This intentionally deletes all tables/data in the public schema.
    with engine.begin() as connection:
        connection.execute(text("DROP SCHEMA IF EXISTS public CASCADE"))
        connection.execute(text("CREATE SCHEMA public"))
    Base.metadata.create_all(bind=engine)

    upload_root = Path(settings.upload_dir).expanduser()
    if upload_root.exists():
        for child in upload_root.iterdir():
            if child.is_dir() and not child.is_symlink():
                shutil.rmtree(child)
            else:
                child.unlink(missing_ok=True)
    upload_root.mkdir(parents=True, exist_ok=True)
    print("Demo database and uploaded files reset successfully. Register a fresh demo account.", flush=True)


if __name__ == "__main__":
    main()
