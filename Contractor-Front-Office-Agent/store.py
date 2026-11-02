"""Storage backends behind one interface.

SupabaseStore (Postgres) is used when SUPABASE_URL and SUPABASE_SECRET_KEY are set;
otherwise CsvStore keeps everything in data/*.csv. Records are plain dicts of strings,
with the same fields as the CSV columns, whichever backend is in use.
"""

from __future__ import annotations

import os
import threading
from functools import lru_cache

import pandas as pd

import data_store as ds

DEMO_BUSINESS_ID = "00000000-0000-0000-0000-000000000001"


class SlotUnavailable(Exception):
    """The slot was booked by someone else (or never existed)."""


class UnknownCustomer(Exception):
    pass


# ---------- CSV ----------

class CsvStore:
    name = "CSV files"
    _lock = threading.Lock()  # Streamlit serves sessions on threads; serialize writes.

    def is_empty(self) -> bool:
        return not all(p.exists() for p in (ds.SCHEDULE_CSV, ds.JOBS_CSV, ds.CUSTOMERS_CSV))

    # Customers
    def customers(self) -> list[dict]:
        return ds.load_customers().to_dict("records")

    def get_customer(self, customer_id: str) -> dict | None:
        return next((c for c in self.customers() if c["customer_id"] == customer_id), None)

    def find_customer(self, phone: str = "", name: str = "") -> dict | None:
        """Match by phone first (any format), then by exact case-insensitive name."""
        customers = self.customers()
        digits = ds.normalize_phone(phone)
        if digits:
            match = next((c for c in customers if ds.normalize_phone(c["phone"]) == digits), None)
            if match:
                return match
        wanted = name.strip().lower()
        if wanted:
            return next((c for c in customers if c["name"].strip().lower() == wanted), None)
        return None

    def create_customer(self, name: str, phone: str, email: str = "", address: str = "") -> dict:
        with self._lock:
            customers = ds.load_customers()
            record = {
                "customer_id": ds.next_id(customers, "customer_id", "C", 3),
                "name": name, "phone": phone, "email": email, "address": address,
                "total_jobs": "0", "last_job_date": "", "notes": "",
            }
            ds.save_customers(pd.concat([customers, pd.DataFrame([record])], ignore_index=True))
            return record

    def update_customer(self, customer_id: str, fields: dict) -> dict:
        with self._lock:
            customers = ds.load_customers()
            mask = customers["customer_id"] == customer_id
            for col, value in fields.items():
                customers.loc[mask, col] = value
            ds.save_customers(customers)
            return customers[mask].iloc[0].to_dict()

    # Slots
    def slots(self, since_date: str = "") -> list[dict]:
        schedule = ds.load_schedule()
        if since_date:
            schedule = schedule[schedule["date"] >= since_date]
        return schedule.sort_values(["date", "time"]).to_dict("records")

    def get_slot(self, slot_id: str) -> dict | None:
        return next((s for s in self.slots() if s["slot_id"] == slot_id), None)

    # Jobs
    def jobs(self) -> list[dict]:
        return ds.load_jobs().to_dict("records")

    def get_job(self, job_id: str) -> dict | None:
        return next((j for j in self.jobs() if j["job_id"] == job_id), None)

    def jobs_for_customer(self, customer: dict) -> list[dict]:
        digits = ds.normalize_phone(customer["phone"])
        name = customer["name"].lower()
        return [
            j for j in self.jobs()
            if j["customer_id"] == customer["customer_id"]
            or (digits and ds.normalize_phone(j["phone"]) == digits)
            or j["customer_name"].lower() == name
        ]

    def book(self, slot_id: str, customer_id: str, job: dict, today: str) -> dict:
        """Atomically claim the slot, create the job and bump the customer's job count."""
        with self._lock:
            schedule = ds.load_schedule()
            s_mask = (schedule["slot_id"] == slot_id) & (schedule["status"] == "available")
            if not s_mask.any():
                raise SlotUnavailable(slot_id)
            customers = ds.load_customers()
            c_mask = customers["customer_id"] == customer_id
            if not c_mask.any():
                raise UnknownCustomer(customer_id)
            customer = customers[c_mask].iloc[0]

            jobs = ds.load_jobs()
            record = {
                "job_id": ds.next_id(jobs, "job_id", "J", 4),
                "customer_id": customer_id,
                "customer_name": customer["name"],
                "address": customer["address"],
                "phone": customer["phone"],
                "email": customer["email"],
                "job_type": job["job_type"],
                "urgency": job["urgency"],
                "description": job["description"],
                "slot_id": slot_id,
                "status": "pending_confirmation",
                "created_date": today,
                "notes": job.get("notes", ""),
            }
            ds.save_jobs(pd.concat([jobs, pd.DataFrame([record])], ignore_index=True))
            schedule.loc[s_mask, ["status", "job_id"]] = ["booked", record["job_id"]]
            ds.save_schedule(schedule)
            customers.loc[c_mask, "total_jobs"] = str(int(customer["total_jobs"] or 0) + 1)
            customers.loc[c_mask, "last_job_date"] = today
            ds.save_customers(customers)
            return record

    def set_job_status(self, job_id: str, status: str) -> None:
        with self._lock:
            jobs = ds.load_jobs()
            jobs.loc[jobs["job_id"] == job_id, "status"] = status
            ds.save_jobs(jobs)

    def replace_all(self, schedule: pd.DataFrame, jobs: pd.DataFrame, customers: pd.DataFrame) -> None:
        with self._lock:
            ds.save_schedule(schedule)
            ds.save_jobs(jobs)
            ds.save_customers(customers)


# ---------- Supabase ----------

def _clean(row: dict, columns: list[str]) -> dict:
    """Database row -> the same string dict CsvStore returns."""
    return {col: "" if row.get(col) is None else str(row[col]) for col in columns}


def _escape_like(value: str) -> str:
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


class SupabaseStore:
    name = "Supabase"

    def __init__(self, url: str, key: str, business_id: str):
        from supabase import create_client

        self.db = create_client(url, key)
        self.business_id = business_id

    def _table(self, name: str):
        return self.db.table(name)

    def _select(self, name: str):
        return self._table(name).select("*").eq("business_id", self.business_id)

    def _rpc(self, fn: str, **params):
        return self.db.rpc(fn, {"p_business": self.business_id, **params}).execute()

    def is_empty(self) -> bool:
        return not self._select("slots").limit(1).execute().data

    # Customers
    def customers(self) -> list[dict]:
        rows = self._select("customers").order("customer_id").execute().data
        return [_clean(r, ds.CUSTOMERS_COLUMNS) for r in rows]

    def get_customer(self, customer_id: str) -> dict | None:
        rows = self._select("customers").eq("customer_id", customer_id).limit(1).execute().data
        return _clean(rows[0], ds.CUSTOMERS_COLUMNS) if rows else None

    def find_customer(self, phone: str = "", name: str = "") -> dict | None:
        digits = ds.normalize_phone(phone)[-10:]
        if digits:
            rows = self._select("customers").eq("phone_digits", digits).limit(1).execute().data
            if rows:
                return _clean(rows[0], ds.CUSTOMERS_COLUMNS)
        if name.strip():
            rows = (self._select("customers").ilike("name", _escape_like(name.strip()))
                    .limit(1).execute().data)
            if rows:
                return _clean(rows[0], ds.CUSTOMERS_COLUMNS)
        return None

    def create_customer(self, name: str, phone: str, email: str = "", address: str = "") -> dict:
        row = self._rpc("create_customer", p_name=name, p_phone=phone,
                        p_email=email, p_address=address).data
        return _clean(row, ds.CUSTOMERS_COLUMNS)

    def update_customer(self, customer_id: str, fields: dict) -> dict:
        rows = (self._table("customers").update(fields)
                .eq("business_id", self.business_id).eq("customer_id", customer_id)
                .execute().data)
        return _clean(rows[0], ds.CUSTOMERS_COLUMNS)

    # Slots
    def slots(self, since_date: str = "") -> list[dict]:
        query = self._select("slots")
        if since_date:
            query = query.gte("date", since_date)
        rows = query.order("date").order("time").execute().data
        return [_clean(r, ds.SCHEDULE_COLUMNS) for r in rows]

    def get_slot(self, slot_id: str) -> dict | None:
        rows = self._select("slots").eq("slot_id", slot_id).limit(1).execute().data
        return _clean(rows[0], ds.SCHEDULE_COLUMNS) if rows else None

    # Jobs
    def jobs(self) -> list[dict]:
        rows = self._select("jobs").order("job_id").execute().data
        return [_clean(r, ds.JOBS_COLUMNS) for r in rows]

    def get_job(self, job_id: str) -> dict | None:
        rows = self._select("jobs").eq("job_id", job_id).limit(1).execute().data
        return _clean(rows[0], ds.JOBS_COLUMNS) if rows else None

    def jobs_for_customer(self, customer: dict) -> list[dict]:
        rows = self._select("jobs").eq("customer_id", customer["customer_id"]).order("job_id").execute().data
        return [_clean(r, ds.JOBS_COLUMNS) for r in rows]

    def book(self, slot_id: str, customer_id: str, job: dict, today: str) -> dict:
        from postgrest.exceptions import APIError

        try:
            row = self._rpc("book_job_atomic", p_slot_id=slot_id, p_customer_id=customer_id,
                            p_job=job, p_today=today).data
        except APIError as exc:
            message = str(exc)
            if "SLOT_UNAVAILABLE" in message or "jobs_one_active_per_slot" in message:
                raise SlotUnavailable(slot_id) from exc
            if "UNKNOWN_CUSTOMER" in message:
                raise UnknownCustomer(customer_id) from exc
            raise
        return _clean(row, ds.JOBS_COLUMNS)

    def set_job_status(self, job_id: str, status: str) -> None:
        (self._table("jobs").update({"status": status})
         .eq("business_id", self.business_id).eq("job_id", job_id).execute())

    def replace_all(self, schedule: pd.DataFrame, jobs: pd.DataFrame, customers: pd.DataFrame) -> None:
        records = lambda df: df.fillna("").astype(str).to_dict("records")  # noqa: E731
        self._rpc("reset_business_data", p_customers=records(customers),
                  p_slots=records(schedule), p_jobs=records(jobs))


# ---------- Selection ----------

def supabase_settings() -> tuple[str, str]:
    """(url, key) from the environment. Accepts the URL with or without a trailing /rest/v1."""
    url = os.getenv("SUPABASE_URL", "").strip().rstrip("/")
    url = url.removesuffix("/rest/v1")
    return url, os.getenv("SUPABASE_SECRET_KEY", "").strip()


@lru_cache(maxsize=1)
def _default_store():
    url, key = supabase_settings()
    if url and key:
        return SupabaseStore(url, key, os.getenv("BUSINESS_ID", DEMO_BUSINESS_ID))
    return CsvStore()


def get_store():
    """The active store. Tests replace this function to force a backend."""
    return _default_store()
