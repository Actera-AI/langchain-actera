"""Actera middleware for LangChain agents."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import time
import uuid
from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal

import httpx
from langchain.agents.middleware import AgentMiddleware, ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.errors import GraphBubbleUp
from langgraph.types import Command, interrupt

from ._capture import Capture, capture_input
from ._client import DEFAULT_BASE_URL, ActeraClient, ActeraError

logger = logging.getLogger("langchain_actera")

Mode = Literal["enforcing", "monitoring"]
OnError = Literal["block", "allow"]
OnApproval = Literal["wait", "interrupt"]

_RELEASED = frozenset({"ALLOWED", "APPROVED"})
_WAITING = frozenset({"PENDING_APPROVAL", "RECEIVED"})
_INTERRUPT_TYPE = "actera_approval"
_SUMMARIES = {
    "SUCCESS": "LangChain reported that the tool call completed successfully.",
    "FAILURE": "LangChain reported that the tool call failed.",
}


@dataclass
class _Call:
    """One tool call as Actera sees it."""

    name: str
    tool_call_id: str
    key: str
    body: dict[str, Any]
    can_pause: bool


def _choice(value: str | None, choices: tuple[str, ...], label: str) -> str:
    """Validate a setting. The first choice is the default."""
    chosen = (value or choices[0]).strip().lower()
    if chosen not in choices:
        raise ValueError(f"Invalid {label}: {chosen!r}. Use {' or '.join(choices)}.")
    return chosen


def _sentence(text: str) -> str:
    return text.strip().rstrip(".")


def _outcome(result: Any) -> str:
    return "FAILURE" if isinstance(result, ToolMessage) and result.status == "error" else "SUCCESS"


class ActeraMiddleware(AgentMiddleware):
    """Ask Actera before every tool call an agent makes.

    Actera answers with one of three results. An allowed call runs. A blocked
    call does not run, and the model gets a message saying why. A call that
    needs approval is held until a person answers in Actera, Slack or email,
    then runs or is refused. Every call is recorded in Actera as a decision,
    together with what happened when the tool ran.

    The middleware starts in monitoring: each call is recorded with the answer
    its policies give, and every call runs. Pass `mode="enforcing"` to apply
    the answers.

    Example:
        ```python
        from langchain.agents import create_agent
        from langchain_actera import ActeraMiddleware

        agent = create_agent(model, tools=tools, middleware=[ActeraMiddleware()])
        ```

    Args:
        api_key: The agent's Actera API key. Defaults to `ACTERA_API_KEY`.
        agent_id: The agent's ID in Actera. Optional, because the key already
            identifies the agent. Defaults to `ACTERA_AGENT_ID`.
        base_url: The Actera API address. Defaults to `ACTERA_BASE_URL`, then
            `https://app.actera.dev`.
        mode: `"monitoring"` records Actera's answer and lets every call run.
            `"enforcing"` applies the answer. Defaults to `ACTERA_MODE`, then
            `"monitoring"`.
        on_error: What to do in enforcing mode when Actera cannot evaluate a
            call: `"block"` refuses it (fail closed), `"allow"` runs it (fail
            open). Defaults to `ACTERA_EVALUATION_FAILURE`, then `"block"`.
        on_approval: How to hold a call that needs approval. `"wait"` keeps the
            tool call waiting until a person answers. `"interrupt"` pauses the
            run with a LangGraph interrupt, which needs a checkpointer; resume
            it with `resume_command`.
        approval_timeout: Seconds to wait for a person. When they run out, the
            call is refused.
        poll_interval: Seconds between checks while waiting.
        capture: `"tool-inputs"` sends the call's arguments, with values under
            sensitive field names masked. Values are not scanned for secrets.
            `"metadata"` sends argument names only.
        timeout: Seconds allowed for each request to Actera.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        agent_id: str | None = None,
        base_url: str | None = None,
        mode: Mode | None = None,
        on_error: OnError | None = None,
        on_approval: OnApproval = "wait",
        approval_timeout: float = 600.0,
        poll_interval: float = 2.0,
        capture: Capture = "tool-inputs",
        timeout: float = 8.0,
    ) -> None:
        super().__init__()
        self.tools = []
        env = os.environ
        self._enforcing = _choice(mode or env.get("ACTERA_MODE"), ("monitoring", "enforcing"), "mode") == "enforcing"
        self._fail_open = _choice(on_error or env.get("ACTERA_EVALUATION_FAILURE"), ("block", "allow"), "on_error") == "allow"
        self._interrupts = _choice(on_approval, ("wait", "interrupt"), "on_approval") == "interrupt"
        self._capture: Capture = _choice(capture, ("tool-inputs", "metadata"), "capture")  # type: ignore[assignment]
        if approval_timeout < 0 or poll_interval <= 0 or timeout <= 0:
            raise ValueError("approval_timeout cannot be negative, and poll_interval and timeout must be positive.")
        self._approval_timeout = approval_timeout
        self._poll_interval = poll_interval
        self._client = ActeraClient(
            api_key=api_key or env.get("ACTERA_API_KEY", ""),
            agent_id=agent_id or env.get("ACTERA_AGENT_ID"),
            base_url=base_url or env.get("ACTERA_BASE_URL") or DEFAULT_BASE_URL,
            timeout=timeout,
        )

    # LangChain hooks

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        """Ask Actera about the call, then run it, refuse it or hold it."""
        decision = None
        try:
            call = self._call(request)
            decision = self._client.decide(call.body)
        except Exception as error:
            refusal = self._unevaluated(request, error)
            if refusal is not None:
                return refusal
        if decision is None:
            return handler(request)  # Runs with no decision to report against.
        if self._enforcing and decision["status"] in _WAITING:
            decision = self._hold(call, decision)
        if self._enforcing and decision["status"] not in _RELEASED:
            return self._refusal(call, decision)
        try:
            result = handler(request)
        except GraphBubbleUp:
            raise  # The tool paused itself and has not finished.
        except Exception:
            self._report(call, decision, "FAILURE")
            raise
        self._report(call, decision, _outcome(result))
        return result

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        """Async version of `wrap_tool_call`."""
        async with self._client.async_session() as session:
            decision = None
            try:
                call = self._call(request)
                decision = await self._client.adecide(session, call.body)
            except Exception as error:
                refusal = self._unevaluated(request, error)
                if refusal is not None:
                    return refusal
            if decision is None:
                return await handler(request)
            if self._enforcing and decision["status"] in _WAITING:
                decision = await self._ahold(session, call, decision)
            if self._enforcing and decision["status"] not in _RELEASED:
                return self._refusal(call, decision)
            try:
                result = await handler(request)
            except GraphBubbleUp:
                raise
            except Exception:
                await self._areport(session, call, decision, "FAILURE")
                raise
            await self._areport(session, call, decision, _outcome(result))
            return result

    # Resuming a run paused with on_approval="interrupt"

    def resume_command(self, result: Mapping[str, Any] | Iterable[Any], *, timeout: float | None = None) -> Command[Any]:
        """Wait for reviewers, then return the command that resumes the held calls.

        Pass the value `agent.invoke` returned when the run paused. This waits
        up to `timeout` seconds (`approval_timeout` by default) for every call
        Actera is holding in it to be answered. A call that is still waiting
        when the run resumes pauses the run again.

        Example:
            ```python
            result = agent.invoke(inputs, config)
            while "__interrupt__" in result:
                result = agent.invoke(actera.resume_command(result), config)
            ```
        """
        held = _held(result)
        deadline = time.monotonic() + (self._approval_timeout if timeout is None else timeout)
        statuses = {key: self._status_or_waiting(decision) for key, decision in held.items()}
        while any(status in _WAITING for status in statuses.values()) and time.monotonic() < deadline:
            time.sleep(max(0.0, min(self._poll_interval, deadline - time.monotonic())))
            for key, decision in held.items():
                if statuses[key] in _WAITING:
                    statuses[key] = self._status_or_waiting(decision)
        return _resume(held, statuses)

    async def aresume_command(
        self, result: Mapping[str, Any] | Iterable[Any], *, timeout: float | None = None
    ) -> Command[Any]:
        """Async version of `resume_command`."""
        held = _held(result)
        deadline = time.monotonic() + (self._approval_timeout if timeout is None else timeout)
        async with self._client.async_session() as session:
            statuses = {key: await self._astatus_or_waiting(session, decision) for key, decision in held.items()}
            while any(status in _WAITING for status in statuses.values()) and time.monotonic() < deadline:
                await asyncio.sleep(max(0.0, min(self._poll_interval, deadline - time.monotonic())))
                for key, decision in held.items():
                    if statuses[key] in _WAITING:
                        statuses[key] = await self._astatus_or_waiting(session, decision)
        return _resume(held, statuses)

    # The decision request

    def _call(self, request: ToolCallRequest) -> _Call:
        tool_call = request.tool_call
        name = str(tool_call["name"])
        tool_call_id = tool_call.get("id") or ""
        arguments = tool_call.get("args") or {}
        parameters, captured = capture_input(arguments, self._capture)
        if self._enforcing and parameters is None and captured.get("reason") != "metadata_only":
            raise ActeraError(f"the tool inputs could not be sent ({captured.get('reason')})")

        config = getattr(getattr(request, "runtime", None), "config", None) or {}
        configurable = config.get("configurable") or {}
        thread_id = configurable.get("thread_id")
        # A paused run starts the held call again when it resumes. The thread and
        # the call's place in it give that second attempt the same key, so Actera
        # returns the decision it already made. Without a thread, a call runs once.
        identity = (
            [self._client.credential_id, str(thread_id), str(configurable.get("checkpoint_ns") or ""), tool_call_id]
            if thread_id is not None
            else [self._client.credential_id, uuid.uuid4().hex, tool_call_id]
        )
        key = hashlib.sha256(json.dumps(identity).encode("utf-8")).hexdigest()

        attributes: dict[str, Any] = {"tool_name": name}
        if parameters is not None:
            attributes["parameters"] = parameters
        if isinstance(arguments, dict):
            attributes["input_fields"] = [str(field) for field in arguments]
        langchain = {"tool_call_id": tool_call_id}
        if thread_id is not None:
            langchain["thread_id"] = str(thread_id)[:1024]
        body: dict[str, Any] = {
            "idempotency_key": f"langchain-tool:{key}",
            "mode": "ENFORCING" if self._enforcing else "MONITORING",
            "action": {"type": f"langchain.tool.{name}", "attributes": attributes},
            "context": {
                "langchain": langchain,
                "source": "langchain_middleware",
                "capture": self._capture,
                "input_capture": captured,
            },
        }
        if self._client.agent_id:
            body["agent"] = {"id": self._client.agent_id}
        can_pause = thread_id is not None and configurable.get("__pregel_checkpointer") is not None
        return _Call(name=name, tool_call_id=tool_call_id, key=key, body=body, can_pause=can_pause)

    # Results the model sees

    def _message(self, name: str, tool_call_id: str, text: str) -> ToolMessage:
        return ToolMessage(content=text, name=name, tool_call_id=tool_call_id, status="error")

    def _unevaluated(self, request: ToolCallRequest, error: Exception) -> ToolMessage | None:
        """Actera gave no answer. Refuse the call, or return None to let it run."""
        name = str(request.tool_call["name"])
        if not self._enforcing or self._fail_open:
            logger.warning("Actera could not evaluate tool call %s, which ran anyway: %s", name, error)
            return None
        logger.warning("Actera could not evaluate tool call %s, which was not run: %s", name, error)
        return self._message(
            name,
            request.tool_call.get("id") or "",
            f"Actera could not evaluate this tool call, so it was not run: {_sentence(str(error))}.",
        )

    def _refusal(self, call: _Call, decision: dict[str, Any]) -> ToolMessage:
        status = decision["status"]
        reference = f"decision {decision['id']}"
        if status == "BLOCKED":
            why = _sentence(decision.get("explanation") or "") or "a policy does not allow it"
            text = f"Actera blocked this tool call ({reference}): {why}. It was not run."
        elif status == "REJECTED":
            why = _sentence(decision.get("reason") or "")
            text = (
                f"A reviewer rejected this tool call in Actera ({reference}){': ' + why if why else ''}. "
                "It was not run. Do not retry it unless the person asks."
            )
        elif status in _WAITING:
            review = f" It can be reviewed at {decision['review_url']}." if decision.get("review_url") else ""
            text = (
                f"Actera is holding this tool call for a person's approval ({reference}) and nobody answered "
                f"within {self._approval_timeout:g} seconds, so it was not run.{review} "
                "Approving it later does not run it. Ask again to request a new approval."
            )
        else:
            text = f"Actera did not allow this tool call ({reference}, status {status}). It was not run."
        return self._message(call.name, call.tool_call_id, text)

    # Holding a call that needs approval

    def _hold(self, call: _Call, decision: dict[str, Any]) -> dict[str, Any]:
        current = self._latest(decision, decision)
        if current["status"] not in _WAITING:
            return current
        if self._pauses(call, current):
            while current["status"] in _WAITING:
                interrupt(_interrupt_value(call, current))  # Pauses the run. Returns once it is resumed.
                current = self._latest(decision, current)
            return current
        deadline = time.monotonic() + self._approval_timeout
        while current["status"] in _WAITING and time.monotonic() < deadline:
            time.sleep(max(0.0, min(self._poll_interval, deadline - time.monotonic())))
            current = self._latest(decision, current)
        return current

    async def _ahold(self, session: httpx.AsyncClient, call: _Call, decision: dict[str, Any]) -> dict[str, Any]:
        current = await self._alatest(session, decision, decision)
        if current["status"] not in _WAITING:
            return current
        if self._pauses(call, current):
            while current["status"] in _WAITING:
                interrupt(_interrupt_value(call, current))
                current = await self._alatest(session, decision, current)
            return current
        deadline = time.monotonic() + self._approval_timeout
        while current["status"] in _WAITING and time.monotonic() < deadline:
            await asyncio.sleep(max(0.0, min(self._poll_interval, deadline - time.monotonic())))
            current = await self._alatest(session, decision, current)
        return current

    def _pauses(self, call: _Call, decision: dict[str, Any]) -> bool:
        logger.info(
            "Actera is holding tool call %s for approval (decision %s). Review it at %s",
            call.name,
            decision["id"],
            decision.get("review_url") or "the Needs approval page in Actera",
        )
        if self._interrupts and not call.can_pause:
            logger.warning(
                "on_approval='interrupt' needs an agent with a checkpointer and a thread_id. "
                "Waiting for the reviewer inside the tool call instead."
            )
        return self._interrupts and call.can_pause

    # A failed status check keeps the call held. Only Actera's answer releases it.

    def _latest(self, decision: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
        try:
            return {**decision, **self._client.status(decision)}
        except ActeraError as error:
            logger.warning("Could not read Actera decision %s: %s", decision["id"], error)
            return current

    async def _alatest(
        self, session: httpx.AsyncClient, decision: dict[str, Any], current: dict[str, Any]
    ) -> dict[str, Any]:
        try:
            return {**decision, **await self._client.astatus(session, decision)}
        except ActeraError as error:
            logger.warning("Could not read Actera decision %s: %s", decision["id"], error)
            return current

    def _status_or_waiting(self, decision: dict[str, Any]) -> str:
        return self._latest(decision, {"status": "PENDING_APPROVAL"})["status"]

    async def _astatus_or_waiting(self, session: httpx.AsyncClient, decision: dict[str, Any]) -> str:
        return (await self._alatest(session, decision, {"status": "PENDING_APPROVAL"}))["status"]

    # Reporting what happened. A report that fails is logged; the tool's result stands.

    def _report_body(self, call: _Call, decision: dict[str, Any], outcome: str) -> dict[str, Any]:
        return {
            "idempotency_key": f"langchain-tool-outcome:{self._client.credential_id}:{call.key}:{outcome}",
            "decision_id": decision["id"],
            "outcome": outcome,
            "occurred_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "agent": {"id": decision.get("agent_id") or self._client.agent_id},
            "action": call.body["action"],
            "context": call.body["context"],
            "details": {
                "schema_version": 1,
                "source": "langchain_middleware",
                "tool_name": call.name,
                "tool_call_id": call.tool_call_id,
                "outcome_scope": "tool_call",
                "summary": _SUMMARIES[outcome],
            },
        }

    def _report(self, call: _Call, decision: dict[str, Any], outcome: str) -> None:
        try:
            self._client.report(decision["id"], self._report_body(call, decision, outcome))
        except Exception as error:
            logger.warning("Actera did not record the outcome of tool call %s: %s", call.name, error)

    async def _areport(self, session: httpx.AsyncClient, call: _Call, decision: dict[str, Any], outcome: str) -> None:
        try:
            await self._client.areport(session, decision["id"], self._report_body(call, decision, outcome))
        except Exception as error:
            logger.warning("Actera did not record the outcome of tool call %s: %s", call.name, error)


def _interrupt_value(call: _Call, decision: dict[str, Any]) -> dict[str, Any]:
    value = {
        "type": _INTERRUPT_TYPE,
        "decision_id": decision["id"],
        "agent_id": decision.get("agent_id"),
        "tool": call.name,
        "tool_call_id": call.tool_call_id,
    }
    if decision.get("review_url"):
        value["review_url"] = decision["review_url"]
    return value


def _held(result: Mapping[str, Any] | Iterable[Any]) -> dict[str, dict[str, Any]]:
    """The decisions Actera is holding in a paused run, by interrupt ID."""
    interrupts = result.get("__interrupt__", ()) if isinstance(result, Mapping) else result
    held = {
        item.id: {"id": item.value["decision_id"], "agent_id": item.value.get("agent_id")}
        for item in interrupts or ()
        if isinstance(getattr(item, "value", None), dict) and item.value.get("type") == _INTERRUPT_TYPE
    }
    if not held:
        raise ValueError("Actera is not holding any tool call in this result.")
    return held


def _resume(held: dict[str, dict[str, Any]], statuses: dict[str, str]) -> Command[Any]:
    # The values are for the application's own records. On resume the middleware
    # reads each decision from Actera again and acts on that.
    return Command(resume={key: {"decision_id": held[key]["id"], "status": statuses[key]} for key in held})
