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

SLOT_TIMES = ["09:00", "11:00", "14:00", "16:00"]
SLOT_HOURS = "2"
WORKDAYS = 10


def _workdays(start: date, count: int) -> list[date]:
    days, d = [], start
    while len(days) < count:
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)
    return days


def build_sample_data(today: date | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    today = today or date.today()
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
        ],
        columns=ds.CUSTOMERS_COLUMNS,
    )

    # Pre-book a few slots so the schedule looks realistic.
    bookings = [
        # (job_id, customer_row, slot_index, job_type, urgency, description, status, notes)
        ("J1001", 0, 1, "HVAC service", "this week", "Annual furnace tune-up before winter.",
         "confirmed", "VIP customer"),
        ("J1002", 2, 4, "Electrical fault", "this week", "Breaker trips when the dryer runs.",
         "pending_confirmation", ""),
        ("J1003", 1, 9, "Roof repair", "flexible", "A few shingles missing after the storm.",
         "confirmed", "Crew needed: 2-person crew + ladder truck"),
    ]
    jobs = []
    for job_id, cust_idx, slot_idx, job_type, urgency, desc, status, notes in bookings:
        c = customers.iloc[cust_idx]
        slot_id = schedule.at[slot_idx, "slot_id"]
        schedule.loc[slot_idx, ["status", "job_id"]] = ["booked", job_id]
        jobs.append([
            job_id, c["name"], c["address"], c["phone"], c["email"], job_type, urgency,
            desc, slot_id, status, (today - timedelta(days=1)).isoformat(), notes,
        ])
    jobs_df = pd.DataFrame(jobs, columns=ds.JOBS_COLUMNS)

    return schedule, jobs_df, customers


def seed(reset: bool = False) -> None:
    schedule, jobs, customers = build_sample_data()
    for df, path, save in [
        (schedule, ds.SCHEDULE_CSV, ds.save_schedule),
        (jobs, ds.JOBS_CSV, ds.save_jobs),
        (customers, ds.CUSTOMERS_CSV, ds.save_customers),
    ]:
        if reset or not path.exists():
            save(df)
            print(f"Wrote {path.relative_to(ds.DATA_DIR.parent)} ({len(df)} rows)")


def ensure_data() -> None:
    """Seed missing files, and refresh the schedule if every slot is in the past."""
    seed(reset=False)
    schedule = ds.load_schedule()
    if schedule.empty or schedule["date"].max() < date.today().isoformat():
        seed(reset=True)


if __name__ == "__main__":
    seed(reset="--reset" in sys.argv)
