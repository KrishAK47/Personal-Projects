"""Conversation evals: a simulated customer (Claude playing a persona) talks to the real agent,
then we check what the agent actually did: which tools it called, what it saved, what it said.

These call the Claude API (roughly 10-20 cents per full run), so they only run when
ANTHROPIC_API_KEY is set:
    pytest -m eval
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import pytest
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

import data_store as ds

pytestmark = [
    pytest.mark.eval,
    pytest.mark.usefixtures("data_dir"),
    pytest.mark.skipif(not os.getenv("ANTHROPIC_API_KEY"), reason="needs ANTHROPIC_API_KEY"),
]

CUSTOMER_MODEL = os.getenv("EVAL_CUSTOMER_MODEL", "claude-haiku-4-5")
MAX_TURNS = 16

CUSTOMER_PROMPT = """You are role-playing a customer contacting a home services company by text.
Stay in character. Reply with ONLY the customer's next message: short, casual, one or two lines.
Answer only what you were just asked, using these facts (never volunteer facts that weren't asked for):
{facts}
{behaviour}
If the assistant offers time slots, {slot_choice}
If the assistant reads back details and they match your facts, say yes, that's correct."""


@dataclass
class Conversation:
    transcript: list[tuple[str, str]] = field(default_factory=list)
    events: list[tuple[int, dict]] = field(default_factory=list)  # (turn index, tool event)

    def tools_called(self) -> list[str]:
        return [e["tool"] for _, e in self.events]

    def calls(self, name: str) -> list[dict]:
        return [e for _, e in self.events if e["tool"] == name]

    def first_turn(self, name: str) -> int | None:
        return next((t for t, e in self.events if e["tool"] == name), None)

    def agent_replies(self) -> list[str]:
        return [text for who, text in self.transcript if who == "agent"]

    def dump(self) -> str:
        return "\n".join(f"{who.upper()}: {text}" for who, text in self.transcript)


def run_conversation(opening: str, facts: str, behaviour: str = "",
                     slot_choice: str = "pick the first one offered.") -> Conversation:
    from agent import FrontOfficeAgent

    agent = FrontOfficeAgent()
    customer = ChatAnthropic(model=CUSTOMER_MODEL, temperature=0, max_tokens=150)
    system = SystemMessage(content=CUSTOMER_PROMPT.format(
        facts=facts, behaviour=behaviour, slot_choice=slot_choice))

    convo, history, message = Conversation(), [], opening
    for turn in range(MAX_TURNS):
        reply, events = agent.respond(history, message)
        convo.transcript += [("customer", message), ("agent", reply)]
        convo.events += [(turn, e) for e in events]
        if "generate_confirmation" in convo.tools_called():
            break
        message = customer.invoke([
            system,
            HumanMessage(content=f"Conversation so far:\n{convo.dump()}\n\nYour next message:"),
        ]).content.strip()
    return convo


def assert_one_question_at_a_time(convo: Conversation) -> None:
    for reply in convo.agent_replies():
        assert reply.count("?") <= 2, f"Too many questions at once:\n{reply}\n\n{convo.dump()}"


def assert_confirmed_before_booking(convo: Conversation) -> None:
    turn = convo.first_turn("book_job")
    assert turn is not None and turn > 0, convo.dump()
    read_back = convo.transcript[2 * turn - 1][1]  # agent message just before the customer said yes
    assert "?" in read_back, f"Agent booked without asking for confirmation:\n{convo.dump()}"


def test_new_customer_emergency_is_booked_end_to_end():
    convo = run_conversation(
        opening="Hi I have a leaking pipe under my kitchen sink",
        facts="Name: John Smith. Phone: 647-555-1234. Address: 123 Main Street, Toronto. "
              "Email: john.smith@example.com. The pipe is actively dripping water right now.",
    )
    log = convo.dump()
    assert "generate_confirmation" in convo.tools_called(), log

    [booking] = [e for e in convo.calls("book_job") if e["result"].get("status") == "booked"]
    job = booking["result"]["job"]
    assert job["urgency"] == "emergency", log
    assert job["phone"] == "647-555-1234", log
    assert "Main" in job["address"], log
    assert job["email"] == "john.smith@example.com", f"Agent never collected the email:\n{log}"

    # Offered slots must come from the tool, and the booked one must be among them.
    offered = {s["slot_id"] for e in convo.calls("check_availability") for s in e["result"]["slots"]}
    assert job["slot_id"] in offered, log
    assert ds.load_schedule().set_index("slot_id").loc[job["slot_id"], "status"] == "booked"

    sms = convo.calls("generate_confirmation")[-1]["result"]["sms"]
    assert sms.startswith("Hi John"), log
    assert "EMERGENCY" in log.upper(), "Emergency was never flagged to the customer"
    assert_confirmed_before_booking(convo)
    assert_one_question_at_a_time(convo)


def test_returning_vip_is_recognized_and_not_duplicated():
    customers_before = len(ds.load_customers())
    convo = run_conversation(
        opening="Hey, it's time for my furnace tune-up again",
        facts="Name: Maria Lopez. Phone: 416-555-0198. Address: 88 Queen St W, Toronto. "
              "Not urgent, sometime this week is fine.",
    )
    log = convo.dump()
    assert "get_customer_history" in convo.tools_called(), log
    assert len(ds.load_customers()) == customers_before, f"Duplicate customer created:\n{log}"

    [booking] = [e for e in convo.calls("book_job") if e["result"].get("status") == "booked"]
    assert booking["args"]["customer_id"] == "C001", log
    assert any("Maria" in r for r in convo.agent_replies()), "Never greeted by name"
    assert_confirmed_before_booking(convo)
    assert_one_question_at_a_time(convo)


def test_preferred_date_is_passed_to_availability():
    convo = run_conversation(
        opening="Hi, a breaker in my house keeps tripping. Not urgent.",
        facts="Name: Alex Rivera. Phone: 905-555-0111. Address: 4 Birch Lane, Oakville. "
              "Email: none, don't have one. Flexible timing, but you'd prefer Friday October 23, 2026 or later.",
    )
    log = convo.dump()
    calls = convo.calls("check_availability")
    assert calls, log
    assert any(c["args"].get("preferred_date", "") >= "2026-10-23" for c in calls), log

    [booking] = [e for e in convo.calls("book_job") if e["result"].get("status") == "booked"]
    slot_date = ds.load_schedule().set_index("slot_id").loc[booking["args"]["slot_id"], "date"]
    assert slot_date >= "2026-10-23", log


def test_no_booking_when_customer_declines():
    convo = run_conversation(
        opening="Hi, do you guys fix roofs? A few shingles came off.",
        facts="Name: Sam Lee. Phone: 647-555-0199. Address: 77 Pine St, Toronto. Flexible timing.",
        behaviour="You are only checking options today. When the assistant reads back details or "
                  "asks you to confirm a booking, say no thanks, you'll call back later.",
    )
    assert "book_job" not in convo.tools_called(), convo.dump()
    assert len(ds.load_jobs()) == 5
