"""A thin HTTP client for the three Actera endpoints the middleware uses.

- `POST /v1/decisions` evaluates one tool call.
- `GET /v1/decisions/{id}/status` reads the current status of a held call.
- `POST /v1/decisions/{id}/reports` records what happened when the tool ran.

The agent's API key authenticates every request and identifies the agent.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any
from urllib.parse import quote, urlsplit

import httpx

DEFAULT_BASE_URL = "https://app.actera.dev"

_API_KEY = re.compile(r"^actera_\S{25,}$")
_AGENT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_LOOPBACK = {"localhost", "127.0.0.1", "::1"}


class ActeraError(Exception):
    """Actera could not evaluate the call: it was unreachable, refused the request or answered badly."""


def validate_base_url(raw: str) -> str:
    """Return the API origin, refusing anything but HTTPS (or HTTP on this machine)."""
    url = urlsplit(raw.strip())
    if url.username or url.password or url.query or url.fragment or url.path not in ("", "/"):
        raise ValueError("The Actera base URL must be an origin without a path or credentials.")
    if url.scheme != "https" and not (url.scheme == "http" and url.hostname in _LOOPBACK):
        raise ValueError("Use HTTPS for Actera. HTTP is allowed only on localhost.")
    if not url.netloc:
        raise ValueError("The Actera base URL is not a valid address.")
    return f"{url.scheme}://{url.netloc}"


def validate_api_key(api_key: str | None) -> str:
    if not api_key:
        raise ValueError(
            "No Actera API key. In Actera, open Agents, choose Add agent, then API or SDK, "
            "and set the key as ACTERA_API_KEY or pass api_key."
        )
    if not _API_KEY.match(api_key):
        raise ValueError("That is not an Actera agent API key. Agent keys start with actera_.")
    return api_key


def validate_agent_id(agent_id: str | None) -> str | None:
    if agent_id is None or agent_id == "":
        return None
    if not _AGENT_ID.match(agent_id):
        raise ValueError("The Actera agent ID is 1 to 128 letters, digits, dots, underscores, colons or hyphens.")
    return agent_id


class ActeraClient:
    def __init__(
        self,
        *,
        api_key: str,
        agent_id: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 8.0,
        transport: httpx.BaseTransport | None = None,
        async_transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.api_key = validate_api_key(api_key)
        self.agent_id = validate_agent_id(agent_id)
        self.base_url = validate_base_url(base_url)
        self.timeout = timeout
        self.credential_id = hashlib.sha256(self.api_key.encode("utf-8")).hexdigest()[:16]
        self._async_transport = async_transport
        # Tool calls in one step run on separate threads and share this client.
        self._sync = httpx.Client(transport=transport, **self._options())

    def _options(self) -> dict[str, Any]:
        return {
            "base_url": self.base_url,
            "headers": {"authorization": f"Bearer {self.api_key}", "user-agent": "langchain-actera"},
            "timeout": self.timeout,
            "follow_redirects": False,
        }

    def async_session(self) -> httpx.AsyncClient:
        """A client for one tool call. Async clients belong to the event loop that made them."""
        return httpx.AsyncClient(transport=self._async_transport, **self._options())

    # Evaluate one tool call.

    def decide(self, body: dict[str, Any]) -> dict[str, Any]:
        return self._decision(_send(lambda: self._sync.post("/v1/decisions", json=body)))

    async def adecide(self, session: httpx.AsyncClient, body: dict[str, Any]) -> dict[str, Any]:
        return self._decision(await _asend(lambda: session.post("/v1/decisions", json=body)))

    def _decision(self, response: httpx.Response) -> dict[str, Any]:
        decision = _json(response, "decisions")
        if not isinstance(decision.get("id"), str) or not isinstance(decision.get("status"), str):
            raise ActeraError("Actera returned an invalid decision")
        if self.agent_id and decision.get("agent_id") != self.agent_id:
            raise ActeraError("Actera returned a decision for a different agent")
        return decision

    # Read the status of a held call.

    def status(self, decision: dict[str, Any]) -> dict[str, Any]:
        path = _status_path(decision)
        return _matching_status(decision, _json(_send(lambda: self._sync.get(path)), "decision status"))

    async def astatus(self, session: httpx.AsyncClient, decision: dict[str, Any]) -> dict[str, Any]:
        path = _status_path(decision)
        return _matching_status(decision, _json(await _asend(lambda: session.get(path)), "decision status"))

    # Record what happened when the tool ran.

    def report(self, decision_id: str, body: dict[str, Any]) -> None:
        _json(_send(lambda: self._sync.post(_report_path(decision_id), json=body)), "decision report")

    async def areport(self, session: httpx.AsyncClient, decision_id: str, body: dict[str, Any]) -> None:
        _json(await _asend(lambda: session.post(_report_path(decision_id), json=body)), "decision report")


def _status_path(decision: dict[str, Any]) -> str:
    return f"/v1/decisions/{quote(decision['id'], safe='')}/status"


def _report_path(decision_id: str) -> str:
    return f"/v1/decisions/{quote(decision_id, safe='')}/reports"


def _matching_status(decision: dict[str, Any], status: dict[str, Any]) -> dict[str, Any]:
    if (
        status.get("id") != decision["id"]
        or status.get("agent_id") != decision.get("agent_id")
        or not isinstance(status.get("status"), str)
    ):
        raise ActeraError("Actera returned a status for a different decision")
    return status


def _send(request: Any) -> httpx.Response:
    try:
        return request()
    except httpx.HTTPError as error:
        raise ActeraError(f"Actera could not be reached ({type(error).__name__})") from error


async def _asend(request: Any) -> httpx.Response:
    try:
        return await request()
    except httpx.HTTPError as error:
        raise ActeraError(f"Actera could not be reached ({type(error).__name__})") from error


def _json(response: httpx.Response, what: str) -> dict[str, Any]:
    if response.status_code == 429:
        raise _limit_error(response, what)
    if response.status_code == 401:
        raise ActeraError("Actera did not accept the agent API key (HTTP 401). Check ACTERA_API_KEY.")
    if not response.is_success:
        raise ActeraError(f"Actera {what} returned HTTP {response.status_code}")
    try:
        body = response.json()
    except ValueError as error:
        raise ActeraError(f"Actera {what} returned an unreadable response") from error
    if not isinstance(body, dict):
        raise ActeraError(f"Actera {what} returned an unreadable response")
    return body


def _limit_error(response: httpx.Response, what: str) -> ActeraError:
    try:
        body = response.json()
    except ValueError:
        body = None
    if isinstance(body, dict) and body.get("message") == "evaluation_monthly_limit_exceeded":
        return ActeraError("This Actera workspace has used its monthly decisions. See Plan & usage in Actera.")
    return ActeraError(f"Actera {what} returned HTTP 429")
