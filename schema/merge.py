"""Merge multiple ``DiagnosisInput`` records into one canonical input.

Lifted out of ``cli/main.py`` so both the one-shot CLI (``diagnose``) and the
live collector (``collect/``) can reuse it without ``collect`` depending on the
CLI layer. This is pure canonical-schema logic — no I/O, no ``click`` — so it
sits at the schema level that every other layer already depends on.

Field-by-field, first-non-None wins, in the input order. Nested pydantic models
(``PhaseMetrics``) recurse so vLLM's latency-derived decode phase is enriched
with Nsight's / DCGM's ``sm_occupancy`` / ``hbm_bandwidth_util`` rather than
clobbered by it. Because earlier sources win per field, parsers must not populate
the same field with differently-scoped values: e.g. vLLM stores TPOT in
``decode.latency_ms`` (a per-token figure), NOT ``decode.duration_ms``, so
Nsight's kernel-measured phase duration is the one that survives.

Lists for provenance (``source_files``, ``parse_warnings``) are concatenated;
other lists fall through to first-wins.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel

from schema.diagnosis_input import DiagnosisInput

_PROVENANCE_LIST_FIELDS = {"source_files", "parse_warnings"}


def merge_inputs(
    inputs: list[DiagnosisInput],
    *,
    model_name: Optional[str],
    gpu_type: Optional[str],
) -> DiagnosisInput:
    """Combine source-specific inputs into one canonical ``DiagnosisInput``.

    ``model_name`` / ``gpu_type`` overrides win regardless of merge order. Raises
    ``ValueError`` on an empty input list — callers guarantee at least one source.
    """
    if not inputs:
        raise ValueError("No inputs to merge.")
    if len(inputs) == 1:
        merged = inputs[0]
    else:
        # dict(model) iterates pydantic's __iter__ → a shallow {field: value}
        # map of top-level fields (nested BaseModels are shared, not copied).
        # That is safe here: every field is replaced via _merge_field below and
        # the result is rebuilt into a fresh DiagnosisInput, so no input is mutated.
        merged_dict = dict(inputs[0])
        for nxt in inputs[1:]:
            for field_name in DiagnosisInput.model_fields:
                merged_dict[field_name] = _merge_field(
                    field_name, merged_dict.get(field_name), getattr(nxt, field_name)
                )
        merged = DiagnosisInput(**merged_dict)

    overrides: dict[str, str] = {}
    if model_name is not None:
        overrides["model_name"] = model_name
    if gpu_type is not None:
        overrides["gpu_type"] = gpu_type
    if overrides:
        merged = merged.model_copy(update=overrides)

    return merged


def _merge_field(name: str, a, b):
    if name in _PROVENANCE_LIST_FIELDS:
        return (a or []) + (b or [])
    if a is None:
        return b
    if b is None:
        return a
    if isinstance(a, BaseModel) and isinstance(b, BaseModel) and type(a) is type(b):
        return _merge_basemodel(a, b)
    return a  # first-wins for scalars/lists when both populated


def _merge_basemodel(a: BaseModel, b: BaseModel) -> BaseModel:
    merged = dict(a)
    for field_name in type(a).model_fields:
        merged[field_name] = _merge_field(field_name, getattr(a, field_name), getattr(b, field_name))
    return type(a)(**merged)


__all__ = ["merge_inputs"]
