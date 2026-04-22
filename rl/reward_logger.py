from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd


class RewardLogger:
    def __init__(self, output_path: str | Path, *, flush_every: int = 1) -> None:
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.flush_every = max(int(flush_every), 1)
        self._buffer: list[dict[str, Any]] = []

    def log_step(self, payload: Mapping[str, Any]) -> None:
        record = {"record_type": "step", **dict(payload)}
        self._buffer.append(record)
        self._maybe_flush()

    def log_episode(self, payload: Mapping[str, Any]) -> None:
        record = {"record_type": "episode", **dict(payload)}
        self._buffer.append(record)
        self._maybe_flush(force=True)

    def flush(self) -> None:
        if not self._buffer:
            return

        if self.output_path.suffix.lower() == ".parquet":
            existing: list[dict[str, Any]] = []
            if self.output_path.exists():
                existing = pd.read_parquet(self.output_path).to_dict(orient="records")
            combined = existing + self._buffer
            pd.DataFrame(combined).to_parquet(self.output_path, index=False)
        else:
            with self.output_path.open("a", encoding="utf-8") as handle:
                for record in self._buffer:
                    handle.write(json.dumps(record, ensure_ascii=True) + "\n")
        self._buffer.clear()

    def close(self) -> None:
        self.flush()

    def _maybe_flush(self, *, force: bool = False) -> None:
        if force or len(self._buffer) >= self.flush_every:
            self.flush()
