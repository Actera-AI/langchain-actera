"""A stand-in for Actera's API and a scripted agent to run the middleware in."""

from __future__ import annotations

import json
import uuid
from typing import Any

import httpx
from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.tools import tool

from langchain_actera import ActeraMiddleware
from langchain_actera._client import ActeraClient

API_KEY = "actera_" + "k" * 40
AGENT_ID = "support-agent"
BASE_URL = "https://actera.test"

EFFECT_STATUS = {"ALLOW": "ALLOWED", "BLOCK": "BLOCKED", "REQUIRE_APPROVAL": "PENDING_APPROVAL"}


class FakeActera:
    """Implements the decision, status and report endpoints the way the contract describes them."""

    def __init__(self, effects: dict[str, str] | None = None) -> None:
        self.effects = effects or {}
        self.decisions: dict[str, dict[str, Any]] = {}
        self.decision_requests: list[dict[str, Any]] = []
        self.reports: list[dict[str, Any]] = []
        self.status_reads = 0
        self.headers: list[httpx.Headers] = []
        self.down = False
        self.decision_error: int | None = None
        self.failing_status_reads = 0
        self._answer: tuple[int, str, str] | None = None

    def answer_after(self, reads: int, status: str, reason: str = "") -> None:
        """Have a reviewer answer once the held call's status has been read this many times."""
        self._answer = (reads, status, reason)

    def review(self, status: str, reason: str = "") -> None:
        for decision in self.decisions.values():
            if decision["status"] == "PENDING_APPROVAL":
                decision["status"] = status
                decision["reason"] = reason

    def handler(self, request: httpx.Request) -> httpx.Response:
        if self.down:
            raise httpx.ConnectError("connection refused", request=request)
        self.headers.append(request.headers)
        path = request.url.path
        if request.method == "POST" and path == "/v1/decisions":
            return self._decide(json.loads(request.content))
        if request.method == "GET" and path.endswith("/status"):
            return self._status(path.split("/")[3])
        if request.method == "POST" and path.endswith("/reports"):
            self.reports.append(json.loads(request.content))
            return httpx.Response(201, json={})
        return httpx.Response(404, json={"message": "not found"})

    def _decide(self, body: dict[str, Any]) -> httpx.Response:
        self.decision_requests.append(body)
        if self.decision_error:
            return httpx.Response(self.decision_error, json={"message": "refused", "code": "x", "field_errors": []})
        existing = self.decisions.get(body["idempotency_key"])
        if existing:
            return httpx.Response(200, json=existing)
        tool_name = body["action"]["attributes"]["tool_name"]
        effect = self.effects.get(tool_name, "ALLOW")
        monitoring = body.get("mode") == "MONITORING"
        decision = {
            "id": str(uuid.uuid4()),
            "workspace_id": "11111111-1111-4111-8111-111111111111",
            "agent_id": AGENT_ID,
            "action_type": body["action"]["type"],
            "mode": body.get("mode", "ENFORCING"),
            "effect": effect,
            "status": "MONITORED" if monitoring else EFFECT_STATUS[effect],
            "explanation": {"BLOCK": "Customer records cannot be deleted by agents.", "REQUIRE_APPROVAL": "Refunds need approval."}.get(effect, "Allowed."),
            "idempotency_key": body["idempotency_key"],
        }
        self.decisions[body["idempotency_key"]] = decision
        return httpx.Response(200, json=decision)

    def _status(self, decision_id: str) -> httpx.Response:
        self.status_reads += 1
        if self.failing_status_reads:
            self.failing_status_reads -= 1
            return httpx.Response(503, json={"message": "unavailable", "code": "x", "field_errors": []})
        if self._answer and self.status_reads > self._answer[0]:
            self.review(self._answer[1], self._answer[2])
        for decision in self.decisions.values():
            if decision["id"] == decision_id:
                return httpx.Response(
                    200,
                    json={
                        "id": decision["id"],
                        "agent_id": decision["agent_id"],
                        "action_type": decision["action_type"],
                        "idempotency_key": decision["idempotency_key"],
                        "status": decision["status"],
                        "reason": decision.get("reason", ""),
                        "review_url": f"{BASE_URL}/acme/decisions/{decision['id']}",
                    },
                )
        return httpx.Response(404, json={"message": "not found", "code": "x", "field_errors": []})


class ScriptedModel(GenericFakeChatModel):
    """Plays back prepared model turns, so tests need no model provider."""

    def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
        return self


class Agent:
    """A LangChain agent with three tools, a scripted model and the middleware under test."""

    def __init__(self, fake: FakeActera, calls: list[dict[str, Any]], **options: Any) -> None:
        self.ran: list[str] = []
        ran = self.ran

        @tool
        def lookup_order(order_id: str) -> str:
            """Look up an order."""
            ran.append(f"lookup_order:{order_id}")
            return f"Order {order_id} was delivered."

        @tool
        def refund(order_id: str, amount: int) -> str:
            """Refund an order."""
            ran.append(f"refund:{order_id}:{amount}")
            return f"Refunded {amount} for {order_id}."

        @tool
        def delete_customer(customer_id: str) -> str:
            """Delete a customer."""
            ran.append(f"delete_customer:{customer_id}")
            return "Deleted."

        @tool
        def charge_card(card_token: str, amount: int) -> str:
            """Charge a card, which fails."""
            ran.append(f"charge_card:{amount}")
            raise RuntimeError("card declined")

        checkpointer = options.pop("checkpointer", None)
        options.setdefault("poll_interval", 0.01)
        self.middleware = ActeraMiddleware(api_key=API_KEY, base_url=BASE_URL, **options)
        self.middleware._client = ActeraClient(
            api_key=API_KEY,
            agent_id=options.get("agent_id"),
            base_url=BASE_URL,
            transport=httpx.MockTransport(fake.handler),
            async_transport=httpx.MockTransport(fake.handler),
        )
        turns = iter([AIMessage(content="", tool_calls=calls), AIMessage(content="Done.")])
        self.graph = create_agent(
            model=ScriptedModel(messages=turns),
            tools=[lookup_order, refund, delete_customer, charge_card],
            middleware=[self.middleware],
            checkpointer=checkpointer,
        )

    def invoke(self, config: dict[str, Any] | None = None) -> dict[str, Any]:
        return self.graph.invoke({"messages": [{"role": "user", "content": "Go."}]}, config)

    async def ainvoke(self, config: dict[str, Any] | None = None) -> dict[str, Any]:
        return await self.graph.ainvoke({"messages": [{"role": "user", "content": "Go."}]}, config)


def call(name: str, call_id: str = "call_1", **args: Any) -> dict[str, Any]:
    return {"name": name, "args": args, "id": call_id}


def tool_message(result: dict[str, Any], call_id: str = "call_1") -> Any:
    return next(m for m in result["messages"] if getattr(m, "tool_call_id", None) == call_id)
