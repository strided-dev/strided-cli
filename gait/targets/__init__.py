"""Config targets: the declared surfaces ``gait`` reads and writes.

One target, done well, beats five resolved flakily. v1 ships the vLLM args surface;
new surfaces implement :class:`ConfigTarget` and slot in behind the same protocol.
"""

from gait.targets.base import ABSENT, ConfigTarget, Location, ResolveOutcome
from gait.targets.vllm_args import VllmArgsTarget

__all__ = ["ABSENT", "ConfigTarget", "Location", "ResolveOutcome", "VllmArgsTarget"]
