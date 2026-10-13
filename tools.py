"""The five LangChain tools the front office agent can call."""

from __future__ import annotations

import os
from datetime import date, datetime, timedelta

import pandas as pd
from langchain_core.tools import tool

import data_store as ds

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


def _customer_record(row: pd.Series) -> dict:
    record = row.to_dict()
    record["is_vip"] = int(record.get("total_jobs") or 0) >= VIP_JOB_COUNT
    return record


def _find_customer(customers: pd.DataFrame, phone: str = "", name: str = "") -> int | None:
    """Index of a matching customer, by phone first, then exact (case-insensitive) name."""
    digits = ds.normalize_phone(phone)
    if digits:
        match = customers.index[customers["phone"].map(ds.normalize_phone) == digits]
        if len(match):
            return match[0]
    if name.strip():
        match = customers.index[customers["name"].str.strip().str.lower() == name.strip().lower()]
        if len(match):
            return match[0]
    return None


def _slot_datetime(slot: pd.Series) -> datetime:
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
    customers = ds.load_customers()
    idx = _find_customer(customers, phone=phone, name=name)

    if idx is not None:
        # Fill in any blanks with the new info, but never overwrite what's on file.
        for col, value in (("address", address), ("email", email), ("phone", ds.format_phone(phone))):
            if value and not customers.at[idx, col]:
                customers.at[idx, col] = value.strip()
        ds.save_customers(customers)
        return {"status": "existing_customer", "customer": _customer_record(customers.loc[idx])}

    customer_id = ds.next_id(customers, "customer_id", "C", 3)
    new = {
        "customer_id": customer_id,
        "name": name.strip().title() if name.islower() else name.strip(),
        "phone": ds.format_phone(phone),
        "email": email.strip(),
        "address": address.strip(),
        "total_jobs": "0",
        "last_job_date": "",
        "notes": "",
    }
    customers = pd.concat([customers, pd.DataFrame([new])], ignore_index=True)
    ds.save_customers(customers)
    return {"status": "new_customer", "customer": {**new, "is_vip": False}}


@tool
def check_availability(urgency: str, preferred_date: str = "") -> dict:
    """Return up to 3 open time slots from the contractor's schedule.
    urgency: 'emergency', 'this week' or 'flexible'. Emergencies get the very next open slots.
    preferred_date: optional YYYY-MM-DD; non-emergency searches start from this date."""
    level = _normalize_urgency(urgency)
    schedule = ds.load_schedule()
    open_slots = schedule[schedule["status"] == "available"].copy()
    if open_slots.empty:
        return {"urgency": level, "slots": [], "message": "No open slots on the schedule."}

    open_slots["start"] = open_slots.apply(_slot_datetime, axis=1)
    open_slots = open_slots[open_slots["start"] > ds.now()].sort_values("start")

    start_from = None
    if preferred_date and level != "emergency":
        try:
            start_from = date.fromisoformat(preferred_date.strip())
        except ValueError:
            start_from = None
    if start_from:
        later = open_slots[open_slots["start"].dt.date >= start_from]
        if not later.empty:
            open_slots = later

    picks = open_slots.head(3)
    slots = [
        {
            "slot_id": s["slot_id"],
            "date": s["date"],
            "time": s["time"],
            "duration_hours": s["duration_hours"],
            "display": f"{_friendly_day(s['start'].date())} at {_friendly_time(s['time'])}",
        }
        for _, s in picks.iterrows()
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
    customers = ds.load_customers()
    match = customers.index[customers["customer_id"] == customer_id]
    if not len(match):
        return {"error": f"Unknown customer_id {customer_id}. Call collect_customer_info first."}
    c_idx = match[0]
    customer = customers.loc[c_idx]

    schedule = ds.load_schedule()
    s_match = schedule.index[schedule["slot_id"] == slot_id]
    if not len(s_match):
        return {"error": f"Unknown slot_id {slot_id}. Call check_availability for valid slots."}
    s_idx = s_match[0]
    if schedule.at[s_idx, "status"] != "available":
        return {"error": f"Slot {slot_id} was just taken. Call check_availability again."}

    level = _normalize_urgency(urgency)
    notes = [n for n in [materials_or_crew_notes.strip()] if n]
    lowered = f"{job_type} {description}".lower()
    notes += [hint for key, hint in CREW_HINTS.items() if key in lowered and hint not in notes]
    if level == "emergency":
        notes.insert(0, "EMERGENCY")
    is_vip = int(customer["total_jobs"] or 0) + 1 >= VIP_JOB_COUNT
    if is_vip:
        notes.append("VIP customer")

    jobs = ds.load_jobs()
    job_id = ds.next_id(jobs, "job_id", "J", 4)
    today = ds.now().date().isoformat()
    job = {
        "job_id": job_id,
        "customer_name": customer["name"],
        "address": customer["address"],
        "phone": customer["phone"],
        "email": customer["email"],
        "job_type": job_type,
        "urgency": level,
        "description": description,
        "slot_id": slot_id,
        "status": "pending_confirmation",
        "created_date": today,
        "notes": "; ".join(notes),
    }
    ds.save_jobs(pd.concat([jobs, pd.DataFrame([job])], ignore_index=True))

    schedule.loc[s_idx, ["status", "job_id"]] = ["booked", job_id]
    ds.save_schedule(schedule)

    customers.at[c_idx, "total_jobs"] = str(int(customer["total_jobs"] or 0) + 1)
    customers.at[c_idx, "last_job_date"] = today
    ds.save_customers(customers)

    slot = schedule.loc[s_idx]
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
    customers = ds.load_customers()
    idx = _find_customer(customers, phone=phone_or_name, name=phone_or_name)
    jobs = ds.load_jobs()

    if idx is not None:
        customer = customers.loc[idx]
        digits = ds.normalize_phone(customer["phone"])
        mask = (jobs["phone"].map(ds.normalize_phone) == digits) | (
            jobs["customer_name"].str.lower() == customer["name"].lower()
        )
        past = jobs[mask]
        total = max(int(customer["total_jobs"] or 0), len(past))
        record = _customer_record(customer)
    else:
        past = jobs[jobs["customer_name"].str.lower() == phone_or_name.strip().lower()]
        if past.empty:
            return {"found": False, "message": "No history for this customer."}
        record, total = {"name": past.iloc[0]["customer_name"]}, len(past)

    is_vip = total >= VIP_JOB_COUNT
    return {
        "found": True,
        "customer": record,
        "total_jobs": total,
        "is_vip": is_vip,
        "vip_flag": "VIP - repeat client, prioritize and thank them for their loyalty" if is_vip else "",
        "past_jobs": past[["job_id", "job_type", "description", "status", "created_date"]].to_dict("records"),
    }


@tool
def generate_confirmation(job_id: str) -> dict:
    """Generate a ready-to-send SMS and email confirmation for a booked job. Call this right after
    book_job succeeds, then show the SMS text to the contractor."""
    jobs = ds.load_jobs()
    match = jobs.index[jobs["job_id"] == job_id]
    if not len(match):
        return {"error": f"Unknown job_id {job_id}."}
    job = jobs.loc[match[0]]

    schedule = ds.load_schedule()
    slot = schedule[schedule["slot_id"] == job["slot_id"]]
    if slot.empty:
        return {"error": f"Job {job_id} has no valid slot."}
    slot = slot.iloc[0]

    first_name = job["customer_name"].split()[0] if job["customer_name"] else "there"
    when = f"{_friendly_day(date.fromisoformat(slot['date']))} at {_friendly_time(slot['time'])}"
    trade = job["job_type"].lower()
    phone = _business_phone()

    sms = (
        f"Hi {first_name}, your {trade} appointment is confirmed for {when} at {job['address']}. "
        f"Our tech will call 30 mins before arrival. Questions? Call us at {phone}."
    )
    email_subject = f"Appointment confirmed: {job['job_type']} on {when}"
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
    jobs = ds.load_jobs()
    jobs.loc[jobs["job_id"] == job_id, "status"] = "confirmed"
    ds.save_jobs(jobs)


ALL_TOOLS = [
    collect_customer_info,
    check_availability,
    book_job,
    get_customer_history,
    generate_confirmation,
]
