"""Prepare a tool call's arguments for sending to Actera.

Follows the capture rules of Actera's agent hooks: values under sensitive field
names are masked, oversized or too deeply nested inputs are left out, and
`metadata` mode sends no values at all. The original arguments are never changed.
"""

from __future__ import annotations

import json
import re
from typing import Any, Literal

Capture = Literal["tool-inputs", "metadata"]

MAX_INPUT_BYTES = 512 * 1024
MAX_DEPTH = 32
REDACTED = "[redacted]"

_SENSITIVE_KEY = re.compile(
    r"authorization|password|passwd|secret|token|api.?key|credential|cookie|private.?key",
    re.IGNORECASE,
)


class _NestingLimit(Exception):
    pass


def _size(value: Any) -> int:
    return len(json.dumps(value, separators=(",", ":"), default=str).encode("utf-8"))


def _pointer(key: object) -> str:
    return str(key).replace("~", "~0").replace("/", "~1")


def _mask(value: Any, location: str, depth: int, redactions: list[dict[str, str]]) -> Any:
    if depth > MAX_DEPTH:
        raise _NestingLimit
    if isinstance(value, dict):
        masked: dict[str, Any] = {}
        for key, item in value.items():
            field = f"{location}/{_pointer(key)}"
            if _SENSITIVE_KEY.search(str(key)):
                redactions.append({"path": field, "reason": "sensitive_field"})
                masked[str(key)] = REDACTED
            else:
                masked[str(key)] = _mask(item, field, depth + 1, redactions)
        return masked
    if isinstance(value, (list, tuple)):
        return [_mask(item, f"{location}/{index}", depth + 1, redactions) for index, item in enumerate(value)]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def capture_input(arguments: Any, mode: Capture = "tool-inputs") -> tuple[Any | None, dict[str, Any]]:
    """Return the arguments to send (or `None`) and the capture metadata that goes with them."""
    if mode == "metadata":
        return None, {"status": "omitted", "reason": "metadata_only"}
    if _size(arguments) > MAX_INPUT_BYTES:
        return None, {"status": "omitted", "reason": "size_limit"}
    redactions: list[dict[str, str]] = []
    try:
        parameters = _mask(arguments, "", 0, redactions)
    except (_NestingLimit, RecursionError):
        return None, {"status": "omitted", "reason": "nesting_limit"}
    metadata = {"status": "redacted" if redactions else "complete", "redactions": redactions}
    if _size({"parameters": parameters, "metadata": metadata}) > MAX_INPUT_BYTES:
        return None, {"status": "omitted", "reason": "size_limit"}
    return parameters, metadata
