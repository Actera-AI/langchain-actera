"""What an agent does with each answer Actera can give, run through a real LangChain agent."""

from __future__ import annotations

import logging
from typing import Any

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from langchain_actera import ActeraMiddleware
from tests.unit_tests.fakes import AGENT_ID, API_KEY, Agent, FakeActera, call, tool_message

THREAD = {"configurable": {"thread_id": "thread-1"}}


async def run(agent: Agent, style: str, config: dict[str, Any] | None = None) -> dict[str, Any]:
    return agent.invoke(config) if style == "sync" else await agent.ainvoke(config)


styles = pytest.mark.parametrize("style", ["sync", "async"])


@styles
async def test_an_allowed_call_runs_and_its_outcome_is_reported(fake: FakeActera, style: str) -> None:
    agent = Agent(fake, [call("lookup_order", order_id="A-1")])

    result = await run(agent, style)

    assert agent.ran == ["lookup_order:A-1"]
    assert tool_message(result).content == "Order A-1 was delivered."
    [request] = fake.decision_requests
    assert request["mode"] == "ENFORCING"
    assert request["idempotency_key"].startswith("langchain-tool:")
    assert request["action"] == {
        "type": "langchain.tool.lookup_order",
        "attributes": {"tool_name": "lookup_order", "parameters": {"order_id": "A-1"}, "input_fields": ["order_id"]},
    }
    assert request["context"]["source"] == "langchain_middleware"
    assert request["context"]["langchain"] == {"tool_call_id": "call_1"}
    assert "agent" not in request  # The key identifies the agent.
    assert fake.headers[0]["authorization"] == f"Bearer {API_KEY}"
    [report] = fake.reports
    [decision] = fake.decisions.values()
    assert report["outcome"] == "SUCCESS"
    assert report["decision_id"] == decision["id"]
    assert report["agent"] == {"id": AGENT_ID}
    assert report["action"] == request["action"]


@styles
async def test_a_blocked_call_does_not_run_and_the_model_is_told_why(fake: FakeActera, style: str) -> None:
    agent = Agent(fake, [call("delete_customer", customer_id="C-9")])

    result = await run(agent, style)

    assert agent.ran == []
    message = tool_message(result)
    [decision] = fake.decisions.values()
    assert message.status == "error"
    assert "Customer records cannot be deleted by agents" in message.content
    assert decision["id"] in message.content
    assert fake.reports == []
    assert result["messages"][-1].content == "Done."  # The agent carries on after a refusal.


@styles
async def test_a_held_call_runs_once_a_person_approves(fake: FakeActera, style: str) -> None:
    fake.answer_after(2, "APPROVED")
    agent = Agent(fake, [call("refund", order_id="A-1", amount=40)])

    result = await run(agent, style)

    assert agent.ran == ["refund:A-1:40"]
    assert tool_message(result).content == "Refunded 40 for A-1."
    assert fake.status_reads >= 3
    assert [report["outcome"] for report in fake.reports] == ["SUCCESS"]


@styles
async def test_a_held_call_a_reviewer_rejects_does_not_run(fake: FakeActera, style: str) -> None:
    fake.answer_after(1, "REJECTED", "Over this customer's refund limit.")
    agent = Agent(fake, [call("refund", order_id="A-1", amount=40)])

    result = await run(agent, style)

    assert agent.ran == []
    message = tool_message(result)
    assert message.status == "error"
    assert "rejected" in message.content
    assert "Over this customer's refund limit" in message.content


@styles
async def test_a_held_call_nobody_answers_is_refused_when_the_wait_runs_out(fake: FakeActera, style: str) -> None:
    agent = Agent(fake, [call("refund", order_id="A-1", amount=40)], approval_timeout=0.05)

    result = await run(agent, style)

    assert agent.ran == []
    message = tool_message(result)
    assert message.status == "error"
    assert "nobody answered" in message.content
    assert "/acme/decisions/" in message.content  # Where a person can review it.


@styles
async def test_failed_status_checks_keep_a_call_held(fake: FakeActera, style: str) -> None:
    fake.failing_status_reads = 1000
    agent = Agent(fake, [call("refund", order_id="A-1", amount=40)], approval_timeout=0.05, on_error="allow")

    result = await run(agent, style)

    # Failing open applies to calls Actera could not evaluate, never to one it decided to hold.
    assert agent.ran == []
    assert tool_message(result).status == "error"


@styles
async def test_a_call_is_refused_when_actera_is_unreachable(fake: FakeActera, style: str) -> None:
    fake.down = True
    agent = Agent(fake, [call("lookup_order", order_id="A-1")])

    result = await run(agent, style)

    assert agent.ran == []
    message = tool_message(result)
    assert message.status == "error"
    assert "could not evaluate" in message.content


@styles
async def test_a_call_runs_when_actera_is_unreachable_and_the_agent_fails_open(fake: FakeActera, style: str) -> None:
    fake.down = True
    agent = Agent(fake, [call("delete_customer", customer_id="C-9")], on_error="allow")

    result = await run(agent, style)

    assert agent.ran == ["delete_customer:C-9"]
    assert tool_message(result).content == "Deleted."


def test_a_rejected_api_key_is_named_in_the_refusal(fake: FakeActera) -> None:
    fake.decision_error = 401
    agent = Agent(fake, [call("lookup_order", order_id="A-1")])

    message = tool_message(agent.invoke())

    assert agent.ran == []
    assert "ACTERA_API_KEY" in message.content


@styles
async def test_monitoring_records_the_answer_and_lets_every_call_run(fake: FakeActera, style: str) -> None:
    agent = Agent(fake, [call("delete_customer", customer_id="C-9")], mode="monitoring")

    result = await run(agent, style)

    assert agent.ran == ["delete_customer:C-9"]
    assert tool_message(result).content == "Deleted."
    assert fake.decision_requests[0]["mode"] == "MONITORING"
    assert [report["outcome"] for report in fake.reports] == ["SUCCESS"]


def test_monitoring_lets_calls_run_when_actera_is_unreachable(fake: FakeActera) -> None:
    fake.down = True
    agent = Agent(fake, [call("delete_customer", customer_id="C-9")], mode="monitoring")

    agent.invoke()

    assert agent.ran == ["delete_customer:C-9"]


@styles
async def test_a_tool_that_fails_is_reported_as_a_failure(fake: FakeActera, style: str) -> None:
    agent = Agent(fake, [call("charge_card", card_token="tok_1", amount=5)])

    with pytest.raises(RuntimeError, match="card declined"):
        await run(agent, style)

    assert [report["outcome"] for report in fake.reports] == ["FAILURE"]


def test_calls_in_one_step_are_decided_separately(fake: FakeActera) -> None:
    agent = Agent(
        fake,
        [call("lookup_order", "call_1", order_id="A-1"), call("delete_customer", "call_2", customer_id="C-9")],
    )

    result = agent.invoke()

    assert agent.ran == ["lookup_order:A-1"]
    assert tool_message(result, "call_1").status == "success"
    assert tool_message(result, "call_2").status == "error"
    assert len({request["idempotency_key"] for request in fake.decision_requests}) == 2


def test_values_under_sensitive_field_names_are_not_sent(fake: FakeActera) -> None:
    agent = Agent(fake, [call("charge_card", card_token="tok_live_123", amount=5)])

    with pytest.raises(RuntimeError):
        agent.invoke()

    [request] = fake.decision_requests
    assert request["action"]["attributes"]["parameters"] == {"card_token": "[redacted]", "amount": 5}
    assert request["context"]["input_capture"]["status"] == "redacted"
    assert agent.ran == ["charge_card:5"]  # The tool still received the real arguments.


def test_metadata_capture_sends_argument_names_only(fake: FakeActera) -> None:
    agent = Agent(fake, [call("lookup_order", order_id="A-1")], capture="metadata")

    agent.invoke()

    [request] = fake.decision_requests
    assert request["action"]["attributes"] == {"tool_name": "lookup_order", "input_fields": ["order_id"]}
    assert request["context"]["input_capture"] == {"status": "omitted", "reason": "metadata_only"}
    assert agent.ran == ["lookup_order:A-1"]


def test_inputs_too_large_to_send_are_refused_rather_than_run_unchecked(fake: FakeActera) -> None:
    agent = Agent(fake, [call("lookup_order", order_id="x" * 600_000)])

    result = agent.invoke()

    assert agent.ran == []
    assert fake.decision_requests == []
    assert "size_limit" in tool_message(result).content


def test_a_decision_for_another_agent_is_not_trusted(fake: FakeActera) -> None:
    agent = Agent(fake, [call("lookup_order", order_id="A-1")], agent_id="some-other-agent")

    result = agent.invoke()

    assert agent.ran == []
    assert fake.decision_requests[0]["agent"] == {"id": "some-other-agent"}
    assert "different agent" in tool_message(result).content


# Pausing the run instead of waiting inside the tool call


@styles
async def test_interrupt_pauses_the_run_and_resumes_it_after_approval(fake: FakeActera, style: str) -> None:
    agent = Agent(
        fake,
        [call("lookup_order", "call_1", order_id="A-1"), call("refund", "call_2", order_id="A-1", amount=40)],
        on_approval="interrupt",
        checkpointer=InMemorySaver(),
    )

    paused = await run(agent, style, THREAD)

    [held] = paused["__interrupt__"]
    [pending] = [d for d in fake.decisions.values() if d["status"] == "PENDING_APPROVAL"]
    assert held.value["type"] == "actera_approval"
    assert held.value["decision_id"] == pending["id"]
    assert held.value["tool"] == "refund"
    assert held.value["review_url"].endswith(pending["id"])
    assert agent.ran == ["lookup_order:A-1"]

    fake.review("APPROVED")
    if style == "sync":
        result = agent.graph.invoke(agent.middleware.resume_command(paused), THREAD)
    else:
        result = await agent.graph.ainvoke(await agent.middleware.aresume_command(paused), THREAD)

    assert "__interrupt__" not in result
    assert agent.ran == ["lookup_order:A-1", "refund:A-1:40"]  # Each tool ran exactly once.
    assert tool_message(result, "call_2").content == "Refunded 40 for A-1."
    assert len(fake.decisions) == 2  # Resuming reused the held call's decision.
    assert sorted(report["outcome"] for report in fake.reports) == ["SUCCESS", "SUCCESS"]


def test_resuming_before_anyone_answers_pauses_the_run_again(fake: FakeActera) -> None:
    agent = Agent(
        fake, [call("refund", order_id="A-1", amount=40)], on_approval="interrupt", checkpointer=InMemorySaver()
    )
    paused = agent.invoke(THREAD)

    still_paused = agent.graph.invoke(agent.middleware.resume_command(paused, timeout=0), THREAD)

    assert agent.ran == []
    assert still_paused["__interrupt__"][0].value["decision_id"] == paused["__interrupt__"][0].value["decision_id"]

    fake.review("APPROVED")
    result = agent.graph.invoke(agent.middleware.resume_command(still_paused), THREAD)

    assert agent.ran == ["refund:A-1:40"]
    assert "__interrupt__" not in result
    assert len(fake.decisions) == 1


def test_a_paused_call_a_reviewer_rejects_is_refused_on_resume(fake: FakeActera) -> None:
    agent = Agent(
        fake, [call("refund", order_id="A-1", amount=40)], on_approval="interrupt", checkpointer=InMemorySaver()
    )
    paused = agent.invoke(THREAD)

    fake.review("REJECTED", "Not this month.")
    result = agent.graph.invoke(agent.middleware.resume_command(paused), THREAD)

    assert agent.ran == []
    assert "Not this month" in tool_message(result).content


def test_interrupt_without_a_checkpointer_waits_instead(fake: FakeActera, caplog: pytest.LogCaptureFixture) -> None:
    fake.answer_after(2, "APPROVED")
    agent = Agent(fake, [call("refund", order_id="A-1", amount=40)], on_approval="interrupt")

    with caplog.at_level(logging.WARNING, logger="langchain_actera"):
        result = agent.invoke()

    assert "__interrupt__" not in result
    assert agent.ran == ["refund:A-1:40"]
    assert "needs an agent with a checkpointer" in caplog.text


def test_resume_command_refuses_a_result_actera_is_not_holding(fake: FakeActera) -> None:
    agent = Agent(fake, [call("lookup_order", order_id="A-1")])

    with pytest.raises(ValueError, match="not holding"):
        agent.middleware.resume_command(agent.invoke())


# Configuration


def test_settings_come_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ACTERA_API_KEY", API_KEY)
    monkeypatch.setenv("ACTERA_AGENT_ID", "support-agent")
    monkeypatch.setenv("ACTERA_BASE_URL", "http://localhost:8080")
    monkeypatch.setenv("ACTERA_MODE", "enforcing")
    monkeypatch.setenv("ACTERA_EVALUATION_FAILURE", "allow")

    middleware = ActeraMiddleware()

    assert middleware._client.base_url == "http://localhost:8080"
    assert middleware._client.agent_id == "support-agent"
    assert middleware._enforcing is True
    assert middleware._fail_open is True


def test_the_defaults_monitor_and_fail_closed() -> None:
    middleware = ActeraMiddleware(api_key=API_KEY)

    assert middleware._client.base_url == "https://app.actera.dev"
    assert middleware._enforcing is False
    assert middleware._fail_open is False


def test_a_new_middleware_blocks_nothing_until_it_is_set_to_enforce(fake: FakeActera) -> None:
    agent = Agent(fake, [call("delete_customer", customer_id="C-9")], mode=None)

    result = agent.invoke()

    assert agent.ran == ["delete_customer:C-9"]
    assert tool_message(result).content == "Deleted."
    assert fake.decision_requests[0]["mode"] == "MONITORING"


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({}, "No Actera API key"),
        ({"api_key": "sk-not-actera"}, "not an Actera agent API key"),
        ({"api_key": API_KEY, "base_url": "http://actera.example.com"}, "HTTPS"),
        ({"api_key": API_KEY, "base_url": "https://app.actera.dev/v1"}, "origin"),
        ({"api_key": API_KEY, "mode": "strict"}, "Invalid mode"),
        ({"api_key": API_KEY, "on_error": "ignore"}, "Invalid on_error"),
        ({"api_key": API_KEY, "on_approval": "skip"}, "Invalid on_approval"),
    ],
)
def test_invalid_settings_are_refused_when_the_middleware_is_created(options: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        ActeraMiddleware(**options)
