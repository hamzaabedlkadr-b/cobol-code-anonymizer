"""Small, configuration-scoped local caches for validated LLM JSON replies.

The cache lives below the local report directory because replies can contain
source text.  It is never written to the shareable anonymized-output folder.
Only validated replies are stored; model errors and malformed replies are
deliberately retried on a later occurrence instead of becoming cached state.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


def response_cache_key(
    *, stage: str, messages: list[dict[str, str]], schema: dict[str, Any],
    options: dict[str, Any], model_digest: str,
) -> str:
    """Hash the complete effective request and immutable model contents."""
    payload = json.dumps(
        {"stage": stage, "messages": messages, "schema": schema,
         "options": options, "model_digest": model_digest, "think": False, "stream": False},
        sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class PersistentResponseCache:
    """Configuration-scoped JSON cache with an in-memory fallback for tests."""

    def __init__(
        self,
        *,
        report_dir: Path | None,
        stage: str,
    ) -> None:
        self.stage = stage
        self.path = report_dir / "llm_cache" / f"{stage}.json" if report_dir is not None else None
        self.entries: dict[str, dict[str, Any]] = {}
        self._dirty = False
        self._load()

    def get(self, key: str) -> dict[str, Any] | None:
        value = self.entries.get(key)
        return dict(value) if value is not None else None

    def put(self, key: str, payload: dict[str, Any]) -> None:
        if self.entries.get(key) == payload:
            return
        self.entries[key] = dict(payload)
        self._dirty = True

    def _load(self) -> None:
        if self.path is None or not self.path.is_file():
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            if (
                not isinstance(payload, dict)
                or payload.get("stage") != self.stage
                or not isinstance(payload.get("entries"), dict)
            ):
                return
            self.entries = {
                key: value
                for key, value in payload["entries"].items()
                if isinstance(key, str) and isinstance(value, dict)
            }
        except (OSError, json.JSONDecodeError):
            # A local cache is an optimization. A corrupt old cache must never
            # block the privacy pipeline or make an answer look validated.
            self.entries = {}

    def flush(self) -> None:
        """Persist validated answers at the end of a source file."""
        if not self._dirty:
            return
        if self.path is None:
            self._dirty = False
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "stage": self.stage,
            "entries": self.entries,
        }
        file_descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{self.stage}-",
            suffix=".tmp",
            dir=self.path.parent,
        )
        try:
            with os.fdopen(file_descriptor, "w", encoding="utf-8") as temporary:
                json.dump(payload, temporary, ensure_ascii=False, sort_keys=True)
            os.replace(temporary_name, self.path)
            self._dirty = False
        except OSError:
            try:
                os.unlink(temporary_name)
            except OSError:
                pass
            raise
