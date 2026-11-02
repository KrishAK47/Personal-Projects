"""Storage contract tests, run against every available backend.

CSV always runs. Supabase runs when SUPABASE_URL and SUPABASE_SECRET_KEY are set: each test
uses its own random business_id, so it never touches the demo data, and cleans up afterwards.
"""

from __future__ import annotations

import uuid
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import pytest

import seed_data
import store
import tools

HAS_SUPABASE = all(store.supabase_settings())


@pytest.fixture(params=["csv", pytest.param("supabase", marks=pytest.mark.supabase)])
def db(request, data_dir, monkeypatch):
    if request.param == "csv":
        backend = store.CsvStore()
    else:
        if not HAS_SUPABASE:
            pytest.skip("needs SUPABASE_URL and SUPABASE_SECRET_KEY")
        backend = store.SupabaseStore(*store.supabase_settings(), str(uuid.uuid4()))
        backend.replace_all(*seed_data.build_sample_data(today=seed_data.ds.now().date()))
    monkeypatch.setattr(store, "get_store", lambda: backend)
    yield backend
    if request.param == "supabase":
        empty = lambda cols: pd.DataFrame(columns=cols)  # noqa: E731
        backend.replace_all(empty(seed_data.ds.SCHEDULE_COLUMNS), empty(seed_data.ds.JOBS_COLUMNS),
                            empty(seed_data.ds.CUSTOMERS_COLUMNS))


JOB = {"job_type": "Plumbing", "urgency": "flexible", "description": "Dripping tap", "notes": ""}


def test_sample_data_round_trips(db):
    assert len(db.slots()) == 40
    assert len(db.jobs()) == 5
    assert len(db.customers()) == 5
    assert db.get_job("J1002")["urgency"] == "emergency"
    assert db.get_slot("S001") == {
        "slot_id": "S001", "date": "2026-10-20", "time": "09:00",
        "duration_hours": "2", "status": "booked", "job_id": "J1001",
    }


def test_find_customer_by_phone_any_format_and_by_name(db):
    assert db.find_customer(phone="+1 (416) 555-0198")["customer_id"] == "C001"
    assert db.find_customer(phone="", name="PRIYA patel")["customer_id"] == "C003"
    assert db.find_customer(phone="", name="Pri%") is None  # no wildcard matching
    assert db.find_customer(phone="111-111-1111", name="Nobody") is None


def test_create_and_update_customer(db):
    created = db.create_customer("Jo Park", "647-555-0000", "", "")
    assert created["customer_id"] == "C006"
    assert created["total_jobs"] == "0"
    updated = db.update_customer("C006", {"email": "jo@example.com"})
    assert updated["email"] == "jo@example.com"
    assert db.find_customer(phone="6475550000")["email"] == "jo@example.com"


def test_book_links_slot_job_and_customer(db):
    job = db.book("S002", "C002", JOB, today="2026-10-20")
    assert job["job_id"] == "J1006"
    assert job["customer_name"] == "David Chen"
    assert job["status"] == "pending_confirmation"
    assert db.get_slot("S002")["job_id"] == "J1006"
    assert db.get_customer("C002")["total_jobs"] == "2"
    assert [j["job_id"] for j in db.jobs_for_customer(db.get_customer("C002"))][-1] == "J1006"


def test_booked_slot_cannot_be_booked_again(db):
    db.book("S002", "C002", JOB, today="2026-10-20")
    with pytest.raises(store.SlotUnavailable):
        db.book("S002", "C003", JOB, today="2026-10-20")


def test_failed_booking_leaves_slot_free(db):
    with pytest.raises(store.UnknownCustomer):
        db.book("S002", "C999", JOB, today="2026-10-20")
    assert db.get_slot("S002")["status"] == "available"
    assert len(db.jobs()) == 5


def test_simultaneous_bookings_for_one_slot_only_one_wins(db):
    def attempt(customer_id):
        try:
            return db.book("S006", customer_id, JOB, today="2026-10-20")["job_id"]
        except store.SlotUnavailable:
            return None

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(attempt, ["C001", "C002", "C003", "C004", "C005"] * 2))

    winners = [r for r in results if r]
    assert len(winners) == 1, results
    assert db.get_slot("S006")["job_id"] == winners[0]
    assert len(db.jobs()) == 6


def test_job_status_update(db):
    db.set_job_status("J1002", "confirmed")
    assert db.get_job("J1002")["status"] == "confirmed"


def test_full_tool_flow_on_this_backend(db):
    customer = tools.collect_customer_info.invoke(
        {"name": "Lee Wong", "phone": "416 555 7777", "address": "5 Oak St, Toronto"})["customer"]
    slot = tools.check_availability.invoke({"urgency": "emergency"})["slots"][0]
    booked = tools.book_job.invoke({
        "customer_id": customer["customer_id"], "slot_id": slot["slot_id"],
        "job_type": "Plumbing", "description": "Burst pipe", "urgency": "emergency",
    })
    assert booked["status"] == "booked", booked
    confirmation = tools.generate_confirmation.invoke({"job_id": booked["job"]["job_id"]})
    assert confirmation["sms"].startswith("Hi Lee")
    assert "5 Oak St, Toronto" in confirmation["sms"]
    history = tools.get_customer_history.invoke({"phone_or_name": "4165557777"})
    assert [j["job_id"] for j in history["past_jobs"]] == [booked["job"]["job_id"]]
