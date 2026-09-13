"""Shared fixtures for the gait test suite (builders live in gait_builders.py)."""

from __future__ import annotations

import pytest

from gait import Journal, VllmArgsTarget
from gait_builders import make_diagnosis, make_snapshot


@pytest.fixture
def diagnosis():
    return make_diagnosis()


@pytest.fixture
def snapshot():
    return make_snapshot()


@pytest.fixture
def target() -> VllmArgsTarget:
    return VllmArgsTarget.from_command(
        "python -m vllm.entrypoints.openai.api_server "
        "--model meta-llama/Llama-3-8B --block-size 16 --max-num-seqs 256"
    )


@pytest.fixture
def journal(tmp_path) -> Journal:
    return Journal(tmp_path / "journal.jsonl")
