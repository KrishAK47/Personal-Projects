"""Streamlit UI: customer intake chat (main) + contractor dashboard (sidebar)."""

from __future__ import annotations

import os
from datetime import timedelta

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

import data_store as ds
from seed_data import ensure_data
from tools import mark_confirmation_sent

load_dotenv(ds.DATA_DIR.parent / ".env")
ensure_data()

st.set_page_config(page_title="Front Office Agent", page_icon="🛠️", layout="wide")

GREETING = "Hi, thanks for reaching out! What can we help you with today?"
MAX_MESSAGE_CHARS = 500


def demo_message_limit() -> int:
    """Messages allowed per browser session when DEMO_MESSAGE_LIMIT is set; 0 means unlimited."""
    try:
        return max(int(os.getenv("DEMO_MESSAGE_LIMIT", "0")), 0)
    except ValueError:
        return 0


# ---------- Dashboard (sidebar) ----------

def _jobs_with_slots() -> pd.DataFrame:
    jobs = ds.load_jobs()
    schedule = ds.load_schedule()[["slot_id", "date", "time"]]
    merged = jobs.merge(schedule, on="slot_id", how="left")
    return merged.sort_values(["date", "time"])


def render_dashboard() -> None:
    st.sidebar.title("📋 Contractor Dashboard")
    today = ds.now().date()
    jobs = _jobs_with_slots()
    open_jobs = jobs[jobs["status"] != "completed"]

    emergencies = open_jobs[open_jobs["urgency"] == "emergency"]
    unconfirmed = open_jobs[open_jobs["status"] == "pending_confirmation"]
    todays = jobs[jobs["date"] == today.isoformat()]

    c1, c2, c3 = st.sidebar.columns(3)
    c1.metric("Today", len(todays))
    c2.metric("🚨 Urgent", len(emergencies))
    c3.metric("Unconfirmed", len(unconfirmed))

    if not emergencies.empty:
        st.sidebar.error("🚨 **Emergency jobs**")
        for _, j in emergencies.iterrows():
            st.sidebar.markdown(
                f"**{j['job_type']}** — {j['customer_name']}  \n"
                f"{j['date']} {j['time']} · {j['address']}  \n📞 {j['phone']}"
            )

    with st.sidebar.expander(f"Today's jobs ({len(todays)})", expanded=True):
        if todays.empty:
            st.caption("Nothing booked today.")
        for _, j in todays.iterrows():
            flag = "🚨 " if j["urgency"] == "emergency" else ""
            st.markdown(
                f"{flag}**{j['time']}** · {j['job_type']} — {j['customer_name']}  \n"
                f"{j['address']}" + (f"  \n_{j['notes']}_" if j["notes"] else "")
            )

    with st.sidebar.expander(f"Unconfirmed bookings ({len(unconfirmed)})"):
        if unconfirmed.empty:
            st.caption("All bookings confirmed.")
        for _, j in unconfirmed.iterrows():
            st.markdown(f"**{j['job_id']}** · {j['customer_name']} · {j['date']} {j['time']}")
            if st.button("Mark confirmation sent", key=f"confirm_{j['job_id']}"):
                mark_confirmation_sent(j["job_id"])
                st.rerun()

    with st.sidebar.expander("This week's schedule"):
        schedule = ds.load_schedule()
        week_end = (today + timedelta(days=6)).isoformat()
        week = schedule[(schedule["date"] >= today.isoformat()) & (schedule["date"] <= week_end)]
        week = week.merge(
            ds.load_jobs()[["job_id", "customer_name", "job_type"]], on="job_id", how="left"
        ).fillna("")
        st.dataframe(
            week[["date", "time", "status", "customer_name", "job_type"]],
            hide_index=True,
        )

    st.sidebar.divider()
    if st.sidebar.button("🔄 Reset sample data"):
        from seed_data import seed

        seed(reset=True)
        st.rerun()


# ---------- Intake chat (main) ----------

def reset_conversation() -> None:
    st.session_state.history = []          # LangChain messages sent to Claude
    st.session_state.display = [{"role": "assistant", "content": GREETING}]
    st.session_state.confirmations = []    # generate_confirmation results this session


def render_confirmation(conf: dict) -> None:
    title = "🚨 EMERGENCY job booked" if conf["is_emergency"] else "✅ Job booked"
    with st.container(border=True):
        st.markdown(f"**{title} — {conf['job_id']}**")
        st.caption("SMS text (use the copy icon in the top-right of the box):")
        st.code(conf["sms"], language=None, wrap_lines=True)
        with st.expander("Email version"):
            st.code(f"Subject: {conf['email_subject']}\n\n{conf['email_body']}", language=None)
        if st.button("Mark confirmation sent", key=f"sent_{conf['job_id']}"):
            mark_confirmation_sent(conf["job_id"])
            st.toast(f"{conf['job_id']} marked as confirmed")


def require_password() -> None:
    """If APP_PASSWORD is set (e.g. in Streamlit Cloud secrets), gate the app behind it."""
    password = os.getenv("APP_PASSWORD")
    if not password or st.session_state.get("authenticated"):
        return
    st.title("🛠️ Front Office Agent")
    entered = st.text_input("Password", type="password")
    if entered == password:
        st.session_state.authenticated = True
        st.rerun()
    elif entered:
        st.error("Incorrect password.")
    st.stop()


def main() -> None:
    require_password()
    render_dashboard()

    st.title("🛠️ Front Office Agent")
    st.caption("Virtual receptionist for your home services business — type as the customer.")

    limit = demo_message_limit()
    if limit:
        st.info(
            "**Demo mode** — please use made-up names, phones and addresses. "
            "Sample data resets regularly. Try a returning VIP: *Maria Lopez, 416-555-0198*."
        )

    if not os.getenv("ANTHROPIC_API_KEY"):
        st.error("Set ANTHROPIC_API_KEY in a `.env` file (see `.env.example`) and restart.")
        st.stop()

    if "history" not in st.session_state:
        reset_conversation()
    if st.button("New conversation"):
        reset_conversation()
        st.rerun()

    if "agent" not in st.session_state:
        from agent import FrontOfficeAgent

        st.session_state.agent = FrontOfficeAgent()

    for msg in st.session_state.display:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    for conf in st.session_state.confirmations:
        render_confirmation(conf)

    # Counted per session and not reset by "New conversation", so the cap can't be bypassed that way.
    used = st.session_state.setdefault("messages_used", 0)
    if limit and used >= limit:
        st.warning("Demo limit reached — thanks for trying it! Refresh the page to start a new demo.")
        return
    if limit:
        left = limit - used
        st.caption(f"{left} demo message{'' if left == 1 else 's'} left")

    if prompt := st.chat_input("Customer message...", max_chars=MAX_MESSAGE_CHARS):
        st.session_state.messages_used += 1
        st.session_state.display.append({"role": "user", "content": prompt})
        with st.spinner("Thinking..."):
            try:
                reply, events = st.session_state.agent.respond(st.session_state.history, prompt)
            except Exception as exc:
                reply, events = f"⚠️ Error talking to Claude: {exc}", []
        st.session_state.display.append({"role": "assistant", "content": reply})
        for event in events:
            if event["tool"] == "generate_confirmation" and "sms" in event["result"]:
                st.session_state.confirmations.append(event["result"])
        st.rerun()


main()
