"""The first ``ConfigTarget``: the vLLM launch / ``EngineArgs`` surface.

v1 models the surface as a vLLM launch command — the line a user runs to start the
server, e.g.::

    python -m vllm.entrypoints.openai.api_server --model X --block-size 16 --max-num-seqs 256

The target reads and writes ``--flag value`` pairs within that command, optionally
backed by a file so an edit is durable and inspectable (and reversible by writing
the prior value back). It knows a small table of vLLM defaults so a param that is
unset-but-defaulted resolves to its implicit value rather than reading as missing;
a param it does not recognize at all resolves to NOT_FOUND, and a flag given twice
resolves to AMBIGUOUS. Those three outcomes are exactly what :mod:`gait.resolve`
needs to decide between proceeding and abstaining.
"""

from __future__ import annotations

import shlex
from pathlib import Path
from typing import Any, Optional

from gait.targets.base import ABSENT, ConfigTarget, Location, ResolveOutcome

# Known vLLM engine args and their documented defaults. Only params with a concrete
# default live here; an unset param in this table resolves to its default (an unset
# param *not* in this table is NOT_FOUND — gait will not invent a knob).
_VLLM_DEFAULTS: dict[str, Any] = {
    "block-size": 16,
    "max-num-seqs": 256,
    "gpu-memory-utilization": 0.90,
    "swap-space": 4,
    "tensor-parallel-size": 1,
    "max-num-batched-tokens": 8192,
}


def _coerce(param: str, raw: str) -> Any:
    """Coerce a raw flag string to a typed value, guided by the known default."""
    default = _VLLM_DEFAULTS.get(param)
    if isinstance(default, bool):
        return raw.lower() in ("1", "true", "yes", "on")
    if isinstance(default, int) and not isinstance(default, bool):
        try:
            return int(raw)
        except ValueError:
            return raw
    if isinstance(default, float):
        try:
            return float(raw)
        except ValueError:
            return raw
    # Unknown type: best-effort numeric, else string.
    for caster in (int, float):
        try:
            return caster(raw)
        except ValueError:
            continue
    return raw


class VllmArgsTarget:
    """A vLLM launch command as an editable, reversible config surface."""

    def __init__(self, command: str, *, path: Optional[Path] = None) -> None:
        self._tokens: list[str] = shlex.split(command)
        self._path = Path(path) if path is not None else None

    # -- constructors -------------------------------------------------------- #

    @classmethod
    def from_file(cls, path: Path | str) -> "VllmArgsTarget":
        """Load the launch command from a file (its contents are the command)."""
        p = Path(path)
        return cls(p.read_text().strip(), path=p)

    @classmethod
    def from_command(cls, command: str) -> "VllmArgsTarget":
        """Build from an in-memory command string (no file backing)."""
        return cls(command)

    # -- ConfigTarget protocol ---------------------------------------------- #

    @property
    def ref(self) -> str:
        return str(self._path) if self._path is not None else "vllm-args(in-memory)"

    def locate(self, param: str) -> Location:
        flag = f"--{param}"
        indices = [i for i, tok in enumerate(self._tokens) if tok == flag]

        if len(indices) > 1:
            candidates = tuple(self._value_after(i) for i in indices)
            return Location(
                ResolveOutcome.AMBIGUOUS,
                candidates=tuple(c for c in candidates if c is not None),
                note=f"{flag} specified {len(indices)} times",
            )

        if len(indices) == 1:
            raw = self._value_after(indices[0])
            if raw is None:
                # A bare flag with no value (store-true style); gait does not size
                # those in v1, so treat as not a value-bearing knob.
                return Location(ResolveOutcome.NOT_FOUND, note=f"{flag} has no value")
            return Location(ResolveOutcome.RESOLVED, value=_coerce(param, raw))

        # Not present in the command.
        if param in _VLLM_DEFAULTS:
            return Location(
                ResolveOutcome.RESOLVED,
                value=_VLLM_DEFAULTS[param],
                note="implicit default (flag not present in launch command)",
                implicit=True,
            )
        return Location(ResolveOutcome.NOT_FOUND, note=f"{flag} not in command and no known default")

    def write(self, param: str, value: Any) -> None:
        flag = f"--{param}"
        indices = [i for i, tok in enumerate(self._tokens) if tok == flag]
        if len(indices) > 1:
            raise ValueError(
                f"refusing to write ambiguous param {param!r}: {flag} appears "
                f"{len(indices)} times"
            )

        # ABSENT means "restore this param to not-present": remove the flag and its
        # value token, so undoing a change to an implicit-default param leaves the
        # command byte-identical to the original rather than an explicit flag.
        if value is ABSENT:
            if indices:
                i = indices[0]
                if i + 1 < len(self._tokens) and not self._tokens[i + 1].startswith("--"):
                    del self._tokens[i:i + 2]
                else:
                    del self._tokens[i]
            self._flush()
            return

        if indices:
            i = indices[0]
            if i + 1 < len(self._tokens) and not self._tokens[i + 1].startswith("--"):
                self._tokens[i + 1] = str(value)
            else:
                self._tokens.insert(i + 1, str(value))
        else:
            self._tokens.extend([flag, str(value)])
        self._flush()

    # -- helpers ------------------------------------------------------------- #

    def command(self) -> str:
        """The current launch command, reconstructed from tokens."""
        return shlex.join(self._tokens)

    def _value_after(self, flag_index: int) -> Optional[str]:
        nxt = flag_index + 1
        if nxt < len(self._tokens) and not self._tokens[nxt].startswith("--"):
            return self._tokens[nxt]
        return None

    def _flush(self) -> None:
        if self._path is not None:
            self._path.write_text(self.command() + "\n")


# Static structural check: VllmArgsTarget satisfies the ConfigTarget protocol.
_: ConfigTarget = VllmArgsTarget("python -m vllm --model x")


__all__ = ["VllmArgsTarget"]
