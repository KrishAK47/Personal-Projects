"""Generate sample schedule, jobs and customers CSVs relative to today.

Usage:
    python seed_data.py           # only creates files that don't exist yet
    python seed_data.py --reset   # overwrite all data with fresh sample data
"""

from __future__ import annotations

import sys
from datetime import date, timedelta

import pandas as pd

import data_store as ds
import store

SLOT_TIMES = ["09:00", "11:00", "14:00", "16:00"]
SLOT_HOURS = "2"
WORKDAYS = 10


def _workdays(start: date, count: int) -> list[date]:
    """Today (even on a weekend, so the demo always has a "today") plus the following weekdays."""
    days, d = [], start
    while len(days) < count:
        if d == start or d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)
    return days


def build_sample_data(today: date | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    today = today or ds.now().date()
    days = _workdays(today, WORKDAYS)

    schedule = pd.DataFrame(
        [
            {
                "slot_id": f"S{i + 1:03d}",
                "date": d.isoformat(),
                "time": t,
                "duration_hours": SLOT_HOURS,
                "status": "available",
                "job_id": "",
            }
            for i, (d, t) in enumerate((d, t) for d in days for t in SLOT_TIMES)
        ],
        columns=ds.SCHEDULE_COLUMNS,
    )

    customers = pd.DataFrame(
        [
            ["C001", "Maria Lopez", "416-555-0198", "maria.lopez@example.com",
             "88 Queen St W, Toronto", "4", (today - timedelta(days=21)).isoformat(),
             "Prefers morning appointments. Has a dog."],
            ["C002", "David Chen", "905-555-0142", "dchen@example.com",
             "12 Elm Ave, Mississauga", "1", (today - timedelta(days=90)).isoformat(), ""],
            ["C003", "Priya Patel", "647-555-0177", "priya.p@example.com",
             "450 Bloor St E, Toronto", "2", (today - timedelta(days=45)).isoformat(),
             "Side door entrance."],
            ["C004", "Sarah Thompson", "416-555-0123", "sarah.t@example.com",
             "9 Maple Cres, Toronto", "1", today.isoformat(), ""],
            ["C005", "Katherine James", "289-555-0163", "kj@example.com",
             "27 Lakeshore Rd, Oakville", "1", today.isoformat(), ""],
        ],
        columns=ds.CUSTOMERS_COLUMNS,
    )

    # Pre-booked jobs so the dashboard has something to show: 2 today (1 emergency),
    # 3 awaiting confirmation. Slot index = day * 4 + time (0=9am, 1=11am, 2=2pm, 3=4pm).
    bookings = [
        # (job_id, customer_row, slot_index, job_type, urgency, description, status, notes)
        ("J1001", 0, 0, "HVAC service", "this week", "Annual furnace tune-up before winter.",
         "confirmed", "VIP customer"),
        ("J1002", 3, 2, "Plumbing", "emergency", "Burst pipe in the basement, water spreading.",
         "pending_confirmation", "EMERGENCY; Materials: bring pipe repair kit + shop vac"),
        ("J1003", 4, 4, "Patio door repair", "this week", "Patio door and bathroom door sticking.",
         "pending_confirmation", "Crew needed: 1-person crew"),
        ("J1004", 2, 6, "Electrical fault", "this week", "Breaker trips when the dryer runs.",
         "pending_confirmation", ""),
        ("J1005", 1, 13, "Roof repair", "flexible", "A few shingles missing after the storm.",
         "confirmed", "Crew needed: 2-person crew + ladder truck"),
    ]
    jobs = []
    for job_id, cust_idx, slot_idx, job_type, urgency, desc, status, notes in bookings:
        c = customers.iloc[cust_idx]
        slot_id = schedule.at[slot_idx, "slot_id"]
        schedule.loc[slot_idx, ["status", "job_id"]] = ["booked", job_id]
        jobs.append([
            job_id, c["customer_id"], c["name"], c["address"], c["phone"], c["email"], job_type, urgency,
            desc, slot_id, status, (today - timedelta(days=1)).isoformat(), notes,
        ])
    jobs_df = pd.DataFrame(jobs, columns=ds.JOBS_COLUMNS)

    return schedule, jobs_df, customers


def seed(reset: bool = False) -> None:
    """Load fresh sample data into the active store (CSV or Supabase)."""
    db = store.get_store()
    if reset or db.is_empty():
        schedule, jobs, customers = build_sample_data()
        db.replace_all(schedule, jobs, customers)
        print(f"Loaded sample data into {db.name}: {len(schedule)} slots, "
              f"{len(jobs)} jobs, {len(customers)} customers")


_checked_on: str = ""


def ensure_data() -> None:
    """Seed if empty, and regenerate the sample data once it no longer starts today.
    This keeps the demo dashboard current; it means data resets once a day.
    Checked at most once per day per process, so it costs one query a day."""
    global _checked_on
    today = ds.now().date().isoformat()
    if _checked_on == today:
        return
    seed(reset=False)
    slots = store.get_store().slots()
    if not slots or slots[0]["date"] < today:
        seed(reset=True)
    _checked_on = today


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv(ds.DATA_DIR.parent / ".env")
    seed(reset="--reset" in sys.argv)
