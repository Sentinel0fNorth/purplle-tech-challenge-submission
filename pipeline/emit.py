from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


class EventEmitter:
    def __init__(
        self,
        api_url: str | None = None,
        jsonl_path: str | None = None,
        batch_size: int = 100,
        truncate: bool = True,
    ) -> None:
        self.api_url = api_url.rstrip("/") if api_url else None
        self.jsonl_path = Path(jsonl_path) if jsonl_path else None
        self.batch_size = min(batch_size, 500)
        self._batch: list[dict[str, Any]] = []
        if self.jsonl_path:
            self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            if truncate:
                self.jsonl_path.write_text("", encoding="utf-8")

    def emit(self, event: dict[str, Any]) -> None:
        if self.jsonl_path:
            with self.jsonl_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, separators=(",", ":")) + "\n")
        if self.api_url:
            self._batch.append(event)
            if len(self._batch) >= self.batch_size:
                self.flush()

    def emit_many(self, events: list[dict[str, Any]]) -> None:
        for event in events:
            self.emit(event)

    def flush(self) -> None:
        if not self.api_url or not self._batch:
            return
        payload = json.dumps(self._batch).encode("utf-8")
        req = urllib.request.Request(
            f"{self.api_url}/ingest",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        last_error: Exception | None = None
        for attempt in range(4):
            try:
                with urllib.request.urlopen(req, timeout=15) as resp:  # nosec - user-provided trusted API endpoint
                    if resp.status < 500:
                        self._batch.clear()
                        return
            except (urllib.error.URLError, TimeoutError) as exc:
                last_error = exc
            time.sleep(0.5 * (2**attempt))
        raise RuntimeError(f"Failed to deliver event batch after retries: {last_error}")
