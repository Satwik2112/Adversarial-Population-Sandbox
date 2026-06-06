"""Immutable LOG_STORE — append-only simulation logging."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Optional

from pydantic import BaseModel

from aps.schemas.message import Message, SimulationResult


class LogEntry(BaseModel):
    """A single log entry with timestamp and event data."""
    timestamp: datetime
    event_type: str
    simulation_id: str
    data: dict


class LogStore:
    """Append-only log store for simulation events.

    Writes to JSON Lines file + keeps in-memory buffer.
    Immutable: append only, no deletes or updates.
    """

    def __init__(self, log_dir: Path | str = "logs"):
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._buffer: list[LogEntry] = []
        self._lock = Lock()
        self._file_path: Optional[Path] = None

    def _get_log_file(self, simulation_id: str) -> Path:
        """Get or create the log file for a simulation."""
        return self._log_dir / f"{simulation_id}.jsonl"

    def append(self, simulation_id: str, event_type: str, data: dict) -> LogEntry:
        """Append a new entry to the log. Thread-safe."""
        entry = LogEntry(
            timestamp=datetime.now(timezone.utc),
            event_type=event_type,
            simulation_id=simulation_id,
            data=data,
        )

        with self._lock:
            self._buffer.append(entry)
            log_file = self._get_log_file(simulation_id)
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(entry.model_dump_json() + "\n")

        return entry

    def log_message(self, simulation_id: str, message: Message) -> LogEntry:
        """Convenience: log a simulation message."""
        return self.append(
            simulation_id=simulation_id,
            event_type="message",
            data=message.model_dump(mode="json"),
        )

    def log_event(self, simulation_id: str, event_type: str, **kwargs) -> LogEntry:
        """Convenience: log an arbitrary event."""
        return self.append(
            simulation_id=simulation_id,
            event_type=event_type,
            data=kwargs,
        )

    def get_entries(
        self,
        simulation_id: Optional[str] = None,
        event_type: Optional[str] = None,
    ) -> list[LogEntry]:
        """Read entries from the in-memory buffer (filtered)."""
        with self._lock:
            entries = list(self._buffer)

        if simulation_id:
            entries = [e for e in entries if e.simulation_id == simulation_id]
        if event_type:
            entries = [e for e in entries if e.event_type == event_type]

        return entries

    def load_from_file(self, simulation_id: str) -> list[LogEntry]:
        """Load entries from a persisted log file."""
        log_file = self._get_log_file(simulation_id)
        if not log_file.exists():
            return []

        entries = []
        with open(log_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(LogEntry.model_validate_json(line))
        return entries

    @property
    def entry_count(self) -> int:
        """Number of entries in the in-memory buffer."""
        with self._lock:
            return len(self._buffer)

    def clear_buffer(self) -> None:
        """Clear the in-memory buffer (files are preserved — immutable)."""
        with self._lock:
            self._buffer.clear()
