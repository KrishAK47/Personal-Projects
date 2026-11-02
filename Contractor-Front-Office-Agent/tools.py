"""The five LangChain tools the front office agent can call."""

from __future__ import annotations

import os
from datetime import date, datetime, timedelta

from langchain_core.tools import tool

import data_store as ds
import store

VIP_JOB_COUNT = 3
URGENCY_LEVELS = ("emergency", "this week", "flexible")

# Job types that usually need extra hands or parts. Extend this when adding trades.
CREW_HINTS = {
    "roof": "Crew needed: 2-person crew + ladder truck",
    "hvac install": "Crew needed: 2-person crew for unit install",
    "panel": "Materials: confirm panel/breaker parts before visit",
    "water heater": "Materials: may need replacement water heater",
}


def _business_name() -> str:
    return os.getenv("BUSINESS_NAME", "our team")


def _business_phone() -> str:
    return os.getenv("BUSINESS_PHONE", "[your number]")


def _normalize_urgency(urgency: str) -> str:
    u = (urgency or "").strip().lower()
    if any(word in u for word in ("emerg", "urgent", "asap", "now", "today", "high")):
        return "emergency"
    if "week" in u or "soon" in u:
        return "this week"
    return "flexible"


def _customer_record(customer: dict) -> dict:
    return {**customer, "is_vip": int(customer.get("total_jobs") or 0) >= VIP_JOB_COUNT}


def _slot_datetime(slot: dict) -> datetime:
    return datetime.fromisoformat(f"{slot['date']}T{slot['time']}")


def _friendly_day(d: date) -> str:
    today = ds.now().date()
    label = d.strftime("%a %b %d").replace(" 0", " ")
    if d == today:
        return f"today ({label})"
    if d == today + timedelta(days=1):
        return f"tomorrow ({label})"
    return label


def _friendly_time(t: str) -> str:
    return datetime.strptime(t, "%H:%M").strftime("%I:%M%p").lstrip("0").lower()


@tool
def collect_customer_info(name: str, phone: str, address: str = "", email: str = "") -> dict:
    """Look up a customer by phone (or name) and return their record, creating a new one if they
    are not on file. Call this as soon as you have the customer's name and phone number. Call it
    again later with address/email to fill in missing details on the record."""
    db = store.get_store()
    existing = db.find_customer(phone=phone, name=name)

    if existing:
        # Fill in any blanks with the new info, but never overwrite what's on file.
        updates = {
            col: value.strip()
            for col, value in (("address", address), ("email", email), ("phone", ds.format_phone(phone)))
            if value and value.strip() and not existing[col]
        }
        if updates:
            existing = db.update_customer(existing["customer_id"], updates)
        return {"status": "existing_customer", "customer": _customer_record(existing)}

    new = db.create_customer(
        name=name.strip().title() if name.islower() else name.strip(),
        phone=ds.format_phone(phone),
        email=email.strip(),
        address=address.strip(),
    )
    return {"status": "new_customer", "customer": _customer_record(new)}


@tool
def check_availability(urgency: str, preferred_date: str = "") -> dict:
    """Return up to 3 open time slots from the contractor's schedule.
    urgency: 'emergency', 'this week' or 'flexible'. Emergencies get the very next open slots.
    preferred_date: optional YYYY-MM-DD; non-emergency searches start from this date."""
    level = _normalize_urgency(urgency)
    now = ds.now()
    open_slots = [
        s for s in store.get_store().slots(since_date=now.date().isoformat())
        if s["status"] == "available" and _slot_datetime(s) > now
    ]
    if not open_slots:
        return {"urgency": level, "slots": [], "message": "No open slots on the schedule."}

    if preferred_date and level != "emergency":
        try:
            start_from = date.fromisoformat(preferred_date.strip()).isoformat()
        except ValueError:
            start_from = ""
        later = [s for s in open_slots if s["date"] >= start_from] if start_from else []
        if later:
            open_slots = later

    slots = [
        {
            "slot_id": s["slot_id"],
            "date": s["date"],
            "time": s["time"],
            "duration_hours": s["duration_hours"],
            "display": f"{_friendly_day(date.fromisoformat(s['date']))} at {_friendly_time(s['time'])}",
        }
        for s in open_slots[:3]
    ]
    result = {"urgency": level, "slots": slots}
    if level == "emergency":
        result["priority_flag"] = "EMERGENCY - offering the earliest possible slots"
    return result


@tool
def book_job(
    customer_id: str,
    slot_id: str,
    job_type: str,
    description: str,
    urgency: str,
    materials_or_crew_notes: str = "",
) -> dict:
    """Create the job record, mark the slot as booked, and update the customer's history.
    Only call this after the customer has confirmed the details and chosen a slot.
    materials_or_crew_notes: anything the contractor should bring or plan for (parts, extra crew)."""
    db = store.get_store()
    customer = db.get_customer(customer_id)
    if not customer:
        return {"error": f"Unknown customer_id {customer_id}. Call collect_customer_info first."}
    if not db.get_slot(slot_id):
        return {"error": f"Unknown slot_id {slot_id}. Call check_availability for valid slots."}

    level = _normalize_urgency(urgency)
    notes = [n for n in [materials_or_crew_notes.strip()] if n]
    lowered = f"{job_type} {description}".lower()
    notes += [hint for key, hint in CREW_HINTS.items() if key in lowered and hint not in notes]
    if level == "emergency":
        notes.insert(0, "EMERGENCY")
    if int(customer["total_jobs"] or 0) + 1 >= VIP_JOB_COUNT:
        notes.append("VIP customer")

    try:
        job = db.book(
            slot_id,
            customer_id,
            {"job_type": job_type, "urgency": level, "description": description, "notes": "; ".join(notes)},
            today=ds.now().date().isoformat(),
        )
    except store.SlotUnavailable:
        return {"error": f"Slot {slot_id} was just taken. Call check_availability again."}

    slot = db.get_slot(slot_id)
    return {
        "status": "booked",
        "job": job,
        "appointment": f"{_friendly_day(date.fromisoformat(slot['date']))} at {_friendly_time(slot['time'])}",
        "flags": notes,
    }


@tool
def get_customer_history(phone_or_name: str) -> dict:
    """Return a customer's past jobs and whether they are a VIP (3+ jobs). Use this for returning
    customers so you can reference their previous work."""
    db = store.get_store()
    customer = db.find_customer(phone=phone_or_name, name=phone_or_name)
    if not customer:
        return {"found": False, "message": "No history for this customer."}

    past = db.jobs_for_customer(customer)
    total = max(int(customer["total_jobs"] or 0), len(past))
    is_vip = total >= VIP_JOB_COUNT
    return {
        "found": True,
        "customer": _customer_record(customer),
        "total_jobs": total,
        "is_vip": is_vip,
        "vip_flag": "VIP - repeat client, prioritize and thank them for their loyalty" if is_vip else "",
        "past_jobs": [
            {k: j[k] for k in ("job_id", "job_type", "description", "status", "created_date")}
            for j in past
        ],
    }


@tool
def generate_confirmation(job_id: str) -> dict:
    """Generate a ready-to-send SMS and email confirmation for a booked job. Call this right after
    book_job succeeds, then show the SMS text to the contractor."""
    db = store.get_store()
    job = db.get_job(job_id)
    if not job:
        return {"error": f"Unknown job_id {job_id}."}
    slot = db.get_slot(job["slot_id"])
    if not slot:
        return {"error": f"Job {job_id} has no valid slot."}

    first_name = job["customer_name"].split()[0] if job["customer_name"] else "there"
    when = f"{_friendly_day(date.fromisoformat(slot['date']))} at {_friendly_time(slot['time'])}"
    trade = job["job_type"].lower()
    phone = _business_phone()

    sms = (
        f"Hi {first_name}, your {trade} appointment is confirmed for {when} at {job['address']}. "
        f"Our tech will call 30 mins before arrival. Questions? Call us at {phone}."
    )
    slot_label = datetime.fromisoformat(f"{slot['date']}T{slot['time']}").strftime("%a %b %d, %I:%M%p")
    email_subject = f"Appointment confirmed: {job['job_type']}, {slot_label.replace(' 0', ' ')}"
    email_body = (
        f"Hi {first_name},\n\n"
        f"Thanks for choosing {_business_name()}. Your appointment is booked:\n\n"
        f"  Job:      {job['job_type']}\n"
        f"  When:     {when} (about {slot['duration_hours']} hours)\n"
        f"  Where:    {job['address']}\n"
        f"  Details:  {job['description']}\n"
        f"  Job ref:  {job_id}\n\n"
        f"Our tech will call about 30 minutes before arriving. If anything changes, "
        f"call us at {phone}.\n\n"
        f"See you soon,\n{_business_name()}"
    )

    # Status stays "pending_confirmation" until the contractor marks the text as sent in the UI.
    return {
        "job_id": job_id,
        "is_emergency": job["urgency"] == "emergency",
        "sms": sms,
        "email_subject": email_subject,
        "email_body": email_body,
    }


def mark_confirmation_sent(job_id: str) -> None:
    """Called from the UI once the contractor has sent the confirmation to the customer."""
    store.get_store().set_job_status(job_id, "confirmed")


ALL_TOOLS = [
    collect_customer_info,
    check_availability,
    book_job,
    get_customer_history,
    generate_confirmation,
]
