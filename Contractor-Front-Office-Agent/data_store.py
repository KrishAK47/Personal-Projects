"""CSV-backed storage for schedule, jobs and customers."""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
SCHEDULE_CSV = DATA_DIR / "schedule.csv"
JOBS_CSV = DATA_DIR / "jobs.csv"
CUSTOMERS_CSV = DATA_DIR / "customers.csv"

SCHEDULE_COLUMNS = ["slot_id", "date", "time", "duration_hours", "status", "job_id"]
JOBS_COLUMNS = [
    "job_id", "customer_name", "address", "phone", "email", "job_type", "urgency",
    "description", "slot_id", "status", "created_date", "notes",
]
CUSTOMERS_COLUMNS = [
    "customer_id", "name", "phone", "email", "address", "total_jobs", "last_job_date", "notes",
]

_FILES = {
    SCHEDULE_CSV: SCHEDULE_COLUMNS,
    JOBS_CSV: JOBS_COLUMNS,
    CUSTOMERS_CSV: CUSTOMERS_COLUMNS,
}


def _load(path: Path) -> pd.DataFrame:
    columns = _FILES[path]
    if not path.exists():
        return pd.DataFrame(columns=columns)
    # Everything as strings so IDs, phones and times never get mangled.
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    for col in columns:
        if col not in df.columns:
            df[col] = ""
    return df[columns]


def _save(df: pd.DataFrame, path: Path) -> None:
    DATA_DIR.mkdir(exist_ok=True)
    df.to_csv(path, index=False)


def load_schedule() -> pd.DataFrame:
    return _load(SCHEDULE_CSV)


def load_jobs() -> pd.DataFrame:
    return _load(JOBS_CSV)


def load_customers() -> pd.DataFrame:
    return _load(CUSTOMERS_CSV)


def save_schedule(df: pd.DataFrame) -> None:
    _save(df, SCHEDULE_CSV)


def save_jobs(df: pd.DataFrame) -> None:
    _save(df, JOBS_CSV)


def save_customers(df: pd.DataFrame) -> None:
    _save(df, CUSTOMERS_CSV)


def next_id(df: pd.DataFrame, column: str, prefix: str, width: int) -> str:
    """Next sequential ID like C004 or J1005, based on the highest existing one."""
    numbers = [
        int(m.group(1))
        for value in df[column]
        if (m := re.fullmatch(rf"{prefix}(\d+)", str(value)))
    ]
    start = 10 ** (width - 1) if width > 3 else 1
    return f"{prefix}{(max(numbers) + 1) if numbers else start:0{width}d}"


def normalize_phone(phone: str) -> str:
    """Digits only, dropping a leading North American country code."""
    digits = re.sub(r"\D", "", phone or "")
    if len(digits) == 11 and digits.startswith("1"):
        digits = digits[1:]
    return digits


def format_phone(phone: str) -> str:
    digits = normalize_phone(phone)
    if len(digits) == 10:
        return f"{digits[:3]}-{digits[3:6]}-{digits[6:]}"
    return phone.strip()


def now() -> datetime:
    return datetime.now()
