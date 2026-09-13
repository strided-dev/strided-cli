"""Live metric sources for ``watch``.

Each ``Source`` polls one place (an HTTP endpoint, or a sequence of replay files)
and returns a parsed ``DiagnosisInput`` for this tick — or ``None`` when it has
nothing (a transient scrape failure, or a replay stream that has run dry). A
source never raises out of ``poll``: a failure is recorded on ``last_error`` and
the loop carries on. The ``Source`` shape is the seam a future Rust collector
slots into.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Protocol, runtime_checkable

from collect.scrape import ScrapeError, http_get
from parsers.dcgm import parse_dcgm_prometheus
from parsers.vllm import parse_vllm_metrics
from schema import DiagnosisInput


@runtime_checkable
class Source(Protocol):
    name: str
    last_error: Optional[str]
    exhausted: bool

    def poll(self) -> Optional[DiagnosisInput]:
        ...


class _HttpSource:
    """Shared plumbing for live endpoint sources: scrape text, then parse it."""

    exhausted = False  # live endpoints never run dry

    def __init__(self, name: str, url: str, model: str, gpu: str, timeout: float) -> None:
        self.name = name
        self.url = url
        self._model = model
        self._gpu = gpu
        self._timeout = timeout
        self.last_error: Optional[str] = None

    def _parse(self, text: str) -> DiagnosisInput:  # pragma: no cover - overridden
        raise NotImplementedError

    def poll(self) -> Optional[DiagnosisInput]:
        self.last_error = None
        try:
            text = http_get(self.url, self._timeout)
        except ScrapeError as exc:
            self.last_error = str(exc)
            return None
        try:
            return self._parse(text)
        except Exception as exc:  # noqa: BLE001 — boundary; a bad scrape must not crash the loop
            self.last_error = f"parse failed for {self.url}: {type(exc).__name__}: {exc}"
            return None


class VllmSource(_HttpSource):
    def __init__(self, url: str, model: str = "unknown", gpu: str = "unknown",
                 *, timeout: float = 10.0) -> None:
        super().__init__("vllm", url, model, gpu, timeout)

    def _parse(self, text: str) -> DiagnosisInput:
        return parse_vllm_metrics(text, model_name=self._model, gpu_type=self._gpu,
                                  source_file=self.url)


class DcgmSource(_HttpSource):
    def __init__(self, url: str, model: str = "unknown", gpu: str = "unknown",
                 *, timeout: float = 10.0) -> None:
        super().__init__("dcgm", url, model, gpu, timeout)

    def _parse(self, text: str) -> DiagnosisInput:
        return parse_dcgm_prometheus(text, model_name=self._model, gpu_type=self._gpu,
                                     source_file=self.url)


class ReplaySource:
    """Replay a fixed sequence of captured files, one per ``poll`` (offline demo).

    ``parse_fn`` maps a file path (str) to a ``DiagnosisInput`` — e.g.
    ``parse_vllm_metrics_file`` or ``parse_dcgm_prometheus_file``. When the
    sequence is exhausted, ``poll`` returns ``None`` and ``exhausted`` flips True
    so the watch loop can stop cleanly.
    """

    def __init__(self, files: list[Path], parse_fn: Callable[[str], DiagnosisInput],
                 *, name: str = "replay") -> None:
        self.name = name
        self._files = list(files)
        self._parse_fn = parse_fn
        self._i = 0
        self.last_error: Optional[str] = None
        self.exhausted = len(self._files) == 0

    def poll(self) -> Optional[DiagnosisInput]:
        self.last_error = None
        if self._i >= len(self._files):
            self.exhausted = True
            return None
        path = self._files[self._i]
        self._i += 1
        if self._i >= len(self._files):
            self.exhausted = True
        try:
            return self._parse_fn(str(path))
        except Exception as exc:  # noqa: BLE001 — a bad fixture must not crash the loop
            self.last_error = f"replay parse failed for {path.name}: {type(exc).__name__}: {exc}"
            return None


__all__ = ["Source", "VllmSource", "DcgmSource", "ReplaySource"]
