"""Small local-LLM helpers used by optional anonymizer modes."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


# Project-wide Ollama defaults. Edit these values to change the judge used by
# `--name-judge`; CLI options can still override them for individual runs.
OLLAMA_HOST = "http://127.0.0.1:11434"
OLLAMA_MODEL = "ministral-3:3b"
OLLAMA_TIMEOUT = 60.0


@dataclass(frozen=True)
class LlmJsonResult:
    parsed: dict[str, Any] | None
    content: str
    latency_s: float
    schema_ok: bool
    error: str = ""
    retried: bool = False


def call_ollama_json(
    host: str,
    model: str,
    messages: list[dict[str, str]],
    schema: dict[str, Any],
    timeout: float = 60.0,
    options: dict[str, Any] | None = None,
    retry_schema_failures: int = 1,
) -> LlmJsonResult:
    """Call Ollama's chat endpoint and validate the returned JSON object.

    Transport errors are returned in the result instead of raised so callers can
    conservatively keep deterministic scanner output when the LLM is unavailable.
    """
    result = _call_ollama_json_once(host, model, messages, schema, timeout, options)
    retries_left = retry_schema_failures
    while not result.error and not result.schema_ok and retries_left > 0:
        retry = _call_ollama_json_once(host, model, messages, schema, timeout, options)
        result = LlmJsonResult(
            parsed=retry.parsed,
            content=retry.content,
            latency_s=result.latency_s + retry.latency_s,
            schema_ok=retry.schema_ok,
            error=retry.error,
            retried=True,
        )
        retries_left -= 1
    return result


def _call_ollama_json_once(
    host: str,
    model: str,
    messages: list[dict[str, str]],
    schema: dict[str, Any],
    timeout: float,
    options: dict[str, Any] | None,
) -> LlmJsonResult:
    payload = {
        "model": model,
        "messages": messages,
        "format": schema,
        "stream": False,
        "options": {"temperature": 0, **(options or {})},
    }
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{host.rstrip('/')}/api/chat",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    start = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = json.loads(response.read().decode("utf-8"))
    except (json.JSONDecodeError, TimeoutError, urllib.error.URLError) as exc:
        return LlmJsonResult(
            parsed=None,
            content="",
            latency_s=time.monotonic() - start,
            schema_ok=False,
            error=str(exc),
        )

    if not isinstance(raw, dict):
        return LlmJsonResult(
            parsed=None,
            content="",
            latency_s=time.monotonic() - start,
            schema_ok=False,
            error="unexpected response shape",
        )

    message = raw.get("message")
    content = message.get("content", "") if isinstance(message, dict) else ""
    parsed, schema_ok = validate_json_content(content, schema)
    return LlmJsonResult(
        parsed=parsed,
        content=content,
        latency_s=time.monotonic() - start,
        schema_ok=schema_ok,
    )


def validate_json_content(content: str, schema: dict[str, Any]) -> tuple[dict[str, Any] | None, bool]:
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        return None, False
    if not isinstance(parsed, dict):
        return None, False
    return parsed, validate_schema_value(parsed, schema)


def validate_schema_value(value: Any, schema: dict[str, Any]) -> bool:
    expected_type = schema.get("type")
    if expected_type == "object":
        if not isinstance(value, dict):
            return False
        required = schema.get("required", [])
        if not all(key in value for key in required):
            return False
        properties = schema.get("properties", {})
        if schema.get("additionalProperties") is False:
            if any(key not in properties for key in value):
                return False
        return all(
            key not in properties or validate_schema_value(item, properties[key])
            for key, item in value.items()
        )
    if expected_type == "array":
        if not isinstance(value, list):
            return False
        item_schema = schema.get("items")
        return item_schema is None or all(validate_schema_value(item, item_schema) for item in value)
    if expected_type == "string":
        if not isinstance(value, str):
            return False
        allowed = schema.get("enum")
        return allowed is None or value in allowed
    if expected_type == "number":
        return not isinstance(value, bool) and isinstance(value, (int, float))
    if expected_type == "integer":
        return not isinstance(value, bool) and isinstance(value, int)
    if expected_type == "boolean":
        return isinstance(value, bool)
    return True
