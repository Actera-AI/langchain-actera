"""Run a LangChain agent behind Actera and show the three results a tool call can get.

The agent looks up an order, tries to delete a customer and issues a refund.
With the policy in the README, Actera allows the first, blocks the second and
holds the third until a person approves it.

    ACTERA_API_KEY=actera_... python examples/demo.py

By default the model's turns are scripted, so no model provider is needed and
the same three tool calls happen every time. Pass --model to use a real one.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import uuid
from typing import Any

from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from langchain_actera import ActeraMiddleware

PROMPT = (
    "Do these three things in order, one tool call at a time, and continue even if one is refused: "
    "look up order A-1042, delete customer C-77, then refund 250 for order A-1042. "
    "Finish with one line per step saying what happened."
)


@tool
def lookup_order(order_id: str) -> str:
    """Look up an order's status."""
    return f"Order {order_id} was delivered on 3 November."


@tool
def delete_customer(customer_id: str) -> str:
    """Delete a customer and all their data."""
    return f"Customer {customer_id} was deleted."


@tool
def refund_order(order_id: str, amount: int) -> str:
    """Refund an amount for an order."""
    return f"Refunded {amount} for order {order_id}."


class ScriptedModel(GenericFakeChatModel):
    """Plays back prepared turns in place of a model provider."""

    def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
        return self


def scripted_model() -> ScriptedModel:
    def turn(name: str, **args: Any) -> AIMessage:
        return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": f"call_{uuid.uuid4().hex[:12]}"}])

    return ScriptedModel(
        messages=iter(
            [
                turn("lookup_order", order_id="A-1042"),
                turn("delete_customer", customer_id="C-77"),
                turn("refund_order", order_id="A-1042", amount=250),
                AIMessage(content="All three steps were attempted. Each result is shown above."),
            ]
        )
    )


class HeldCalls(logging.Handler):
    """Print where to review a call while the agent waits on it."""

    def emit(self, record: logging.LogRecord) -> None:
        if "is holding" in record.getMessage():
            print(f"   held: {record.getMessage()}", flush=True)


def show(update: dict[str, Any]) -> None:
    for message in (update or {}).get("messages", []):
        if isinstance(message, AIMessage):
            for tool_call in message.tool_calls:
                print(f"\n{tool_call['name']}({json.dumps(tool_call['args'])})", flush=True)
            if message.content and not message.tool_calls:
                print(f"\nAgent: {message.text}", flush=True)
        elif isinstance(message, ToolMessage):
            print(f"   {'not run' if message.status == 'error' else 'ran'}: {message.content}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", help="a model for init_chat_model, such as anthropic:claude-sonnet-5-5")
    parser.add_argument("--interrupt", action="store_true", help="pause the run for approval instead of waiting in the tool call")
    parser.add_argument("--timeout", type=float, default=600.0, help="seconds to wait for a person to approve")
    args = parser.parse_args()

    logging.getLogger("langchain_actera").setLevel(logging.INFO)
    logging.getLogger("langchain_actera").addHandler(HeldCalls())

    actera = ActeraMiddleware(
        mode="enforcing",
        on_approval="interrupt" if args.interrupt else "wait",
        approval_timeout=args.timeout,
    )
    agent = create_agent(
        model=args.model or scripted_model(),
        tools=[lookup_order, delete_customer, refund_order],
        middleware=[actera],
        checkpointer=InMemorySaver() if args.interrupt else None,
    )
    config = {"configurable": {"thread_id": uuid.uuid4().hex}} if args.interrupt else None
    print(f"Actera: {os.environ.get('ACTERA_BASE_URL', 'https://app.actera.dev')}")

    run_input: Any = {"messages": [{"role": "user", "content": PROMPT}]}
    while run_input is not None:
        paused = []
        for chunk in agent.stream(run_input, config, stream_mode="updates"):
            for node, update in chunk.items():
                if node == "__interrupt__":
                    paused.extend(update)
                else:
                    show(update)
        run_input = None
        if paused:
            print("   paused: the run stopped here and resumes once a person answers in Actera", flush=True)
            command = actera.resume_command(paused, timeout=args.timeout)
            if any(answer["status"] == "PENDING_APPROVAL" for answer in command.resume.values()):
                print(f"   nobody answered within {args.timeout:g} seconds; the run stays paused", flush=True)
            else:
                run_input = command


if __name__ == "__main__":
    main()
