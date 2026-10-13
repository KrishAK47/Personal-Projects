"""Front office agent: Claude (via LangChain) running a tool-calling loop over tools.py."""

from __future__ import annotations

import json
import os

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

import data_store as ds
from tools import ALL_TOOLS

DEFAULT_MODEL = "claude-sonnet-4-6"
MAX_TOOL_ROUNDS = 8

BASE_PROMPT = """You are a friendly and efficient front office assistant
for a home services contractor.
Your job is to collect customer information,
check availability, and book jobs.
Ask one question at a time — do not overwhelm the customer.
Be warm and professional.
Always confirm details back to the customer before booking.
Flag emergency jobs immediately.
If a returning customer calls, greet them by name."""

OPERATING_RULES = """
Context:
- Today is {today}. The current time is {time}.
- The business handles plumbing, roofing, electrical, HVAC and general repairs.

How to run the intake:
1. Find out what the problem is (job type + short description) from the first message if given.
2. Get the customer's name, then their phone number. As soon as you have both, call
   collect_customer_info. If they are an existing customer, greet them by name, call
   get_customer_history, and mention their previous work briefly. If they are VIP, thank them.
3. Collect any missing details one at a time: address, then email (email is optional -
   accept "no" and move on). Update the record by calling collect_customer_info again.
4. Ask about urgency (emergency / this week / flexible) unless it is already obvious.
   Active leaks, no heat in winter, sparking, burning smells, or water coming through the
   roof are emergencies: say "**EMERGENCY**" and that you are prioritizing them.
5. Call check_availability and offer the slots exactly as returned (use their "display" text).
   Never invent slots.
6. Once they pick a slot, read back a short summary (name, address, phone, job, date/time) and
   ask them to confirm. Only after they say yes, call book_job, then generate_confirmation.
7. Finish by showing the SMS confirmation text in a quote block, and mention any flags
   (emergency, VIP, materials or crew needed).

Never ask for more than one thing per message. Keep replies short - this is a phone/text style
conversation. If a tool returns an error, recover (e.g. re-check availability) without
exposing technical details.
"""


def system_prompt() -> str:
    now = ds.now()
    return BASE_PROMPT + "\n" + OPERATING_RULES.format(
        today=now.strftime("%A, %B %d, %Y"), time=now.strftime("%I:%M %p")
    )


def _text(message: AIMessage) -> str:
    """Pull plain text out of an AI message whose content may be a list of blocks."""
    if isinstance(message.content, str):
        return message.content
    return "".join(
        block.get("text", "") for block in message.content
        if isinstance(block, dict) and block.get("type") == "text"
    )


class FrontOfficeAgent:
    def __init__(self, model: str | None = None):
        llm = ChatAnthropic(
            model=model or os.getenv("CLAUDE_MODEL", DEFAULT_MODEL),
            temperature=0.3,
            max_tokens=1024,
        )
        self.llm = llm.bind_tools(ALL_TOOLS)
        self.tools = {t.name: t for t in ALL_TOOLS}

    def respond(self, history: list[BaseMessage], user_text: str) -> tuple[str, list[dict]]:
        """Append the user's message to history, run the agent until it replies, and return
        (reply_text, tool_events). history is mutated in place so the caller can keep it."""
        history.append(HumanMessage(content=user_text))
        events: list[dict] = []

        for _ in range(MAX_TOOL_ROUNDS):
            ai: AIMessage = self.llm.invoke([SystemMessage(content=system_prompt()), *history])
            history.append(ai)
            if not ai.tool_calls:
                return _text(ai), events

            for call in ai.tool_calls:
                tool = self.tools.get(call["name"])
                try:
                    result = tool.invoke(call["args"]) if tool else {"error": f"Unknown tool {call['name']}"}
                except Exception as exc:  # surface to the model so it can recover
                    result = {"error": str(exc)}
                events.append({"tool": call["name"], "args": call["args"], "result": result})
                history.append(
                    ToolMessage(content=json.dumps(result, default=str), tool_call_id=call["id"])
                )

        return "Sorry, I got a bit tangled up there. Could you repeat that?", events
