"""journal — an append-only, on-disk record of every applied change.

The journal makes a session resumable and auditable: it powers ``strided undo``,
post-hoc inspection, and crash recovery. ``apply`` writes a *pending* record
**before** it mutates anything, so a crash mid-apply leaves a recoverable trail
rather than an unknown state; it marks the record *applied* once the write lands,
and ``rollback`` marks it *rolled_back*.

Storage is JSON Lines — one self-contained JSON object per line, appended, never
rewritten in place (a rollback appends a status update rather than editing history).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from gait.targets.base import ABSENT

STATUS_PENDING = "pending"
STATUS_APPLIED = "applied"
STATUS_ROLLED_BACK = "rolled_back"

# JSON has no way to represent the ABSENT sentinel, so it round-trips as this marker.
# Non-null on purpose, so a record carrying an ABSENT prior_value still reads as a
# full payload (not a status-only update) in get()/last_applied().
_ABSENT_KEY = "__gait_absent__"


def _encode(value: Any) -> Any:
    return {_ABSENT_KEY: True} if value is ABSENT else value


def _decode(value: Any) -> Any:
    if isinstance(value, dict) and value.get(_ABSENT_KEY) is True:
        return ABSENT
    return value


def default_journal_path() -> Path:
    """Where the journal lives unless overridden (env or explicit path)."""
    env = os.environ.get("STRIDED_GAIT_JOURNAL")
    if env:
        return Path(env)
    return Path.home() / ".strided" / "gait_journal.jsonl"


@dataclass(frozen=True)
class JournalEntry:
    change_id: str
    ts: str
    rule_id: str
    param: str
    prior_value: Any
    new_value: Any
    target_ref: str
    status: str

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "JournalEntry":
        return cls(
            change_id=d["change_id"],
            ts=d["ts"],
            rule_id=d.get("rule_id", ""),
            param=d["param"],
            prior_value=_decode(d["prior_value"]),
            new_value=_decode(d["new_value"]),
            target_ref=d["target_ref"],
            status=d["status"],
        )


class Journal:
    """Append-only change log backed by a JSONL file."""

    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = Path(path) if path is not None else default_journal_path()

    @classmethod
    def default(cls) -> "Journal":
        return cls()

    # -- writes -------------------------------------------------------------- #

    def record_pending(
        self,
        change_id: str,
        *,
        rule_id: str,
        param: str,
        prior_value: Any,
        new_value: Any,
        target_ref: str,
    ) -> None:
        self._append(
            {
                "change_id": change_id,
                "ts": _now(),
                "rule_id": rule_id,
                "param": param,
                "prior_value": _encode(prior_value),
                "new_value": _encode(new_value),
                "target_ref": target_ref,
                "status": STATUS_PENDING,
            }
        )

    def mark_applied(self, change_id: str) -> None:
        self._append_status(change_id, STATUS_APPLIED)

    def mark_rolled_back(self, change_id: str) -> None:
        self._append_status(change_id, STATUS_ROLLED_BACK)

    # -- reads --------------------------------------------------------------- #

    def entries(self) -> list[JournalEntry]:
        """Every raw record in append order (including status updates)."""
        if not self.path.exists():
            return []
        out: list[JournalEntry] = []
        for line in self.path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            out.append(JournalEntry.from_dict(json.loads(line)))
        return out

    def current_status(self, change_id: str) -> Optional[str]:
        """The latest status for a change id, or None if unknown."""
        status: Optional[str] = None
        for e in self.entries():
            if e.change_id == change_id:
                status = e.status
        return status

    def get(self, change_id: str) -> Optional[JournalEntry]:
        """The most recent full record for ``change_id`` (with latest status)."""
        latest: Optional[JournalEntry] = None
        status: Optional[str] = None
        for e in self.entries():
            if e.change_id != change_id:
                continue
            status = e.status
            # Records that carry the full payload (pending) seed the detail;
            # status-only updates just advance the status.
            if e.prior_value is not None or e.new_value is not None or latest is None:
                latest = e
        if latest is None:
            return None
        if status is not None and status != latest.status:
            latest = JournalEntry.from_dict({**latest.__dict__, "status": status})
        return latest

    def last_applied(self) -> Optional[JournalEntry]:
        """The most recently applied change that is not currently rolled back."""
        statuses: dict[str, str] = {}
        order: list[str] = []
        details: dict[str, JournalEntry] = {}
        for e in self.entries():
            statuses[e.change_id] = e.status
            if e.change_id not in order:
                order.append(e.change_id)
            if e.status == STATUS_PENDING:
                details[e.change_id] = e
        for change_id in reversed(order):
            if statuses.get(change_id) == STATUS_APPLIED and change_id in details:
                return details[change_id]
        return None

    # -- internals ----------------------------------------------------------- #

    def _append(self, record: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as fh:
            fh.write(json.dumps(record) + "\n")
            # Flush to disk so a pending record survives a hard process/host kill
            # between the write and a buffered close, not just a clean exception exit.
            fh.flush()
            os.fsync(fh.fileno())

    def _append_status(self, change_id: str, status: str) -> None:
        self._append(
            {
                "change_id": change_id,
                "ts": _now(),
                "param": "",
                "prior_value": None,
                "new_value": None,
                "target_ref": "",
                "status": status,
            }
        )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


__all__ = [
    "Journal",
    "JournalEntry",
    "default_journal_path",
    "STATUS_PENDING",
    "STATUS_APPLIED",
    "STATUS_ROLLED_BACK",
]
