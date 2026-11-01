"""Unit tests for the five agent tools and the sample data. No API key needed."""

from __future__ import annotations

from datetime import date, datetime

import pytest

import data_store as ds
import seed_data
import tools
from tools import (
    book_job,
    check_availability,
    collect_customer_info,
    generate_confirmation,
    get_customer_history,
)

pytestmark = pytest.mark.usefixtures("data_dir")


def _book(customer_id="C002", slot_id="S002", **overrides):
    args = {
        "customer_id": customer_id,
        "slot_id": slot_id,
        "job_type": "Plumbing",
        "description": "Leaking pipe under the kitchen sink",
        "urgency": "flexible",
        **overrides,
    }
    return book_job.invoke(args)


# ---------- collect_customer_info ----------

def test_new_customer_is_created_with_formatted_phone():
    result = collect_customer_info.invoke({"name": "john smith", "phone": "(647) 555 1234"})
    assert result["status"] == "new_customer"
    customer = result["customer"]
    assert customer["customer_id"] == "C006"
    assert customer["name"] == "John Smith"
    assert customer["phone"] == "647-555-1234"
    assert len(ds.load_customers()) == 6


def test_existing_customer_found_by_phone_in_any_format():
    result = collect_customer_info.invoke({"name": "M Lopez", "phone": "+1 (416) 555-0198"})
    assert result["status"] == "existing_customer"
    assert result["customer"]["customer_id"] == "C001"
    assert result["customer"]["is_vip"] is True
    assert len(ds.load_customers()) == 5


def test_existing_customer_found_by_name_when_phone_differs():
    result = collect_customer_info.invoke({"name": "priya patel", "phone": "000"})
    assert result["status"] == "existing_customer"
    assert result["customer"]["customer_id"] == "C003"


def test_existing_customer_blanks_filled_but_not_overwritten():
    collect_customer_info.invoke({"name": "J Smith", "phone": "6475551234"})
    collect_customer_info.invoke(
        {"name": "J Smith", "phone": "6475551234", "address": "123 Main St", "email": "j@x.com"}
    )
    collect_customer_info.invoke(
        {"name": "J Smith", "phone": "6475551234", "address": "999 Other Rd", "email": "other@x.com"}
    )
    row = ds.load_customers().set_index("customer_id").loc["C006"]
    assert row["address"] == "123 Main St"
    assert row["email"] == "j@x.com"


# ---------- check_availability ----------

def test_emergency_gets_earliest_open_slots():
    result = check_availability.invoke({"urgency": "water everywhere, emergency!"})
    assert result["urgency"] == "emergency"
    assert "priority_flag" in result
    # S001 (9am) and S003 (2pm) are pre-booked today.
    assert [s["slot_id"] for s in result["slots"]] == ["S002", "S004", "S006"]
    assert result["slots"][0]["display"].startswith("today")


def test_past_slots_are_never_offered(clock):
    clock.set(datetime(2026, 10, 20, 15, 0))
    result = check_availability.invoke({"urgency": "emergency"})
    assert result["slots"][0]["slot_id"] == "S004"  # 4pm today, the only one left today


def test_preferred_date_is_respected_for_non_emergency():
    result = check_availability.invoke({"urgency": "flexible", "preferred_date": "2026-10-23"})
    assert result["slots"]
    assert all(s["date"] >= "2026-10-23" for s in result["slots"])


def test_preferred_date_is_ignored_for_emergency():
    result = check_availability.invoke({"urgency": "emergency", "preferred_date": "2026-10-26"})
    assert result["slots"][0]["date"] == "2026-10-20"


def test_invalid_preferred_date_falls_back_to_earliest():
    result = check_availability.invoke({"urgency": "flexible", "preferred_date": "next tuesday"})
    assert result["slots"][0]["slot_id"] == "S002"


def test_only_available_slots_are_offered():
    booked = set(ds.load_schedule().query("status == 'booked'")["slot_id"])
    result = check_availability.invoke({"urgency": "this week"})
    assert not booked & {s["slot_id"] for s in result["slots"]}


# ---------- book_job ----------

def test_booking_creates_job_books_slot_and_updates_customer():
    result = _book(urgency="emergency")
    assert result["status"] == "booked"
    job = result["job"]
    assert job["job_id"] == "J1006"
    assert job["status"] == "pending_confirmation"
    assert job["urgency"] == "emergency"
    assert job["notes"].startswith("EMERGENCY")

    slot = ds.load_schedule().set_index("slot_id").loc["S002"]
    assert (slot["status"], slot["job_id"]) == ("booked", "J1006")
    customer = ds.load_customers().set_index("customer_id").loc["C002"]
    assert customer["total_jobs"] == "2"
    assert customer["last_job_date"] == "2026-10-20"


def test_double_booking_is_refused():
    assert _book()["status"] == "booked"
    second = _book(customer_id="C003")
    assert "error" in second
    assert (ds.load_jobs()["slot_id"] == "S002").sum() == 1


@pytest.mark.parametrize(
    "overrides, message",
    [({"customer_id": "C999"}, "Unknown customer_id"), ({"slot_id": "S999"}, "Unknown slot_id")],
)
def test_booking_with_unknown_ids_fails_cleanly(overrides, message):
    before = len(ds.load_jobs())
    result = _book(**overrides)
    assert message in result["error"]
    assert len(ds.load_jobs()) == before


def test_crew_hint_added_for_roof_jobs():
    result = _book(job_type="Roof repair", description="Shingles blown off")
    assert any("Crew needed" in flag for flag in result["flags"])


def test_third_job_makes_customer_vip():
    # Priya (C003) has 2 jobs on file; this booking is her third.
    result = _book(customer_id="C003")
    assert "VIP customer" in result["flags"]


# ---------- get_customer_history ----------

def test_history_flags_vip_and_lists_past_jobs():
    result = get_customer_history.invoke({"phone_or_name": "416 555 0198"})
    assert result["found"] and result["is_vip"]
    assert result["vip_flag"]
    assert [j["job_id"] for j in result["past_jobs"]] == ["J1001"]


def test_history_for_unknown_customer():
    assert get_customer_history.invoke({"phone_or_name": "Nobody Here"})["found"] is False


# ---------- generate_confirmation ----------

def test_confirmation_text_has_the_key_details(monkeypatch):
    monkeypatch.setenv("BUSINESS_PHONE", "416-555-0100")
    job_id = _book()["job"]["job_id"]
    result = generate_confirmation.invoke({"job_id": job_id})
    sms = result["sms"]
    assert sms.startswith("Hi David,")
    assert "today (Tue Oct 20) at 11:00am" in sms
    assert "12 Elm Ave, Mississauga" in sms
    assert "416-555-0100" in sms
    assert "Job ref:  " + job_id in result["email_body"]


def test_confirmation_does_not_mark_job_confirmed_until_sent():
    job_id = _book()["job"]["job_id"]
    generate_confirmation.invoke({"job_id": job_id})
    status = lambda: ds.load_jobs().set_index("job_id").loc[job_id, "status"]  # noqa: E731
    assert status() == "pending_confirmation"
    tools.mark_confirmation_sent(job_id)
    assert status() == "confirmed"


def test_confirmation_for_unknown_job():
    assert "error" in generate_confirmation.invoke({"job_id": "J0000"})


# ---------- urgency parsing ----------

@pytest.mark.parametrize(
    "text, expected",
    [
        ("Emergency", "emergency"),
        ("ASAP please", "emergency"),
        ("today if possible", "emergency"),
        ("sometime this week", "this week"),
        ("whenever", "flexible"),
        ("", "flexible"),
    ],
)
def test_urgency_normalization(text, expected):
    assert tools._normalize_urgency(text) == expected


# ---------- sample data ----------

def test_sample_data_matches_demo_dashboard():
    jobs = ds.load_jobs().merge(ds.load_schedule()[["slot_id", "date"]], on="slot_id")
    assert (jobs["date"] == "2026-10-20").sum() == 2
    assert (jobs["urgency"] == "emergency").sum() == 1
    assert (jobs["status"] == "pending_confirmation").sum() == 3


def test_sample_data_is_consistent():
    schedule, jobs = ds.load_schedule(), ds.load_jobs()
    booked = schedule[schedule["status"] == "booked"]
    assert set(booked["job_id"]) == set(jobs["job_id"])
    assert jobs["slot_id"].is_unique
    assert set(jobs["slot_id"]) == set(booked["slot_id"])
    assert set(jobs["customer_name"]) <= set(ds.load_customers()["name"])


def test_sample_data_always_includes_today_even_on_weekends():
    sunday = date(2026, 11, 1)
    schedule, _, _ = seed_data.build_sample_data(today=sunday)
    assert schedule["date"].iloc[0] == "2026-11-01"
    assert schedule["date"].iloc[4] == "2026-11-02"  # then Monday


def test_stale_data_is_regenerated(clock):
    clock.set(datetime(2026, 10, 27, 8, 0))
    seed_data.ensure_data()
    assert ds.load_schedule()["date"].min() == "2026-10-27"
