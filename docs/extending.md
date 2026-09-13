# Extending strided

strided is built to be extended along three seams: **rules** (new diagnoses),
**parsers** (new input sources), and **fixes** (new machine-actionable changes for
the `gait` agent). Each seam has a small, explicit contract. This guide shows how to
add to each, using the existing code as templates.

The one rule that governs everything: **the canonical schema
[`DiagnosisInput`](../schema/diagnosis_input.py) is sacred.** Parsers write to it,
rules read from it, and it's the only thing that crosses between the two halves.
Adding optional fields to it is fine; changing or repurposing existing fields is a
deliberate, signed-off decision — see [data-sources.md](data-sources.md).

---

## Add a rule

A rule translates schema fields into a diagnosis. The contract is in
[`rules/base.py`](../rules/base.py); [`rules/r01_decode_memory_bound.py`](../rules/r01_decode_memory_bound.py)
is the simplest complete example to copy.

**1. Create `rules/rNN_short_name.py`** and subclass `Rule`:

```python
from rules.base import (
    Abstention, ConfidenceBreakdown, Diagnosis, InsufficientData, Rule, RuleResult,
)
from schema import DiagnosisInput

class MyNewRule(Rule):
    rule_id = "r04"                      # must match the filename prefix, format rNN
    title = "Short human-readable name"
    references = (                       # at least one citation — rules can't be folk wisdom
        "Author et al., 'Paper Title,' Venue Year.",
    )

    def evaluate(self, dx: DiagnosisInput) -> RuleResult:
        # 1. Check required fields; name the gap if missing.
        if dx.some_field is None:
            return InsufficientData(missing=("some_field",))
        # 2. Compute the signal; abstain silently if it doesn't fire.
        if not (dx.some_field > THRESHOLD):
            return Abstention.BELOW_THRESHOLD
        # 3. Otherwise compute confidence (must be in [0.5, 1.0]) and return a Diagnosis.
        return Diagnosis(
            rule_id=self.rule_id,
            cause="What went wrong, past tense, kernel signals → model terms.",
            fix="What to do, present tense, with concrete flags where possible.",
            confidence=confidence,
            confidence_breakdown=ConfidenceBreakdown(
                signal_strength=..., data_completeness=..., notes="...",
            ),
            evidence={"some_field": dx.some_field},   # the numbers that fired it
        )
```

Contract rules enforced for you (`__init_subclass__` checks at import; `Diagnosis`
checks at construction):

- `rule_id` must match `rNN` and the filename prefix; `title` and `references`
  (≥1) are required.
- A firing `Diagnosis` must have confidence in **[0.5, 1.0]** — below 0.5, abstain
  via `Abstention.BELOW_THRESHOLD`. `cause` and `fix` may not be empty.
- Abstain two ways: **`InsufficientData(missing=…, reason=…)`** when fields are
  absent (surfaced to the user as "could not evaluate"), or
  **`Abstention.BELOW_THRESHOLD`** when data is present but the signal doesn't fire
  (silent). Prefer `InsufficientData` so the user learns what to provide.
- Rules **must not** mutate the input or do any I/O. They're evaluated in isolation;
  no rule may know about another (cross-rule logic belongs to the engine).

**2. Register it** in [`engine/registry.py`](../engine/registry.py) — add the class
to the `ALL_RULES` tuple. Registration is explicit (no filesystem auto-discovery, for
security and determinism), and `tests/engine/test_registry.py` asserts every
`rNN_*.py` file contributes exactly one registered class, so a forgotten
registration fails CI.

**3. Write a spec** in the rule module's docstring (copy the shape of
[r01's](../rules/r01_decode_memory_bound.py)): summary, signal, firing condition,
confidence model, abstention, cause/fix text, evidence fields, references, test
cases, open questions.

**4. Add tests** under `tests/rules/test_rNN_*.py`: a positive firing, a boundary
case, a below-threshold negative, and an insufficient-data case. (The target is 80%+
coverage on `rules/` and `engine/`.)

**5. If it should fire live**, add it to the tiering table in
[`collect/tiers.py`](../collect/tiers.py) (`_TIERS`) and map its required fields to
their source (`_FIELD_SOURCE`), so `watch` reports it honestly.

---

## Add a parser

A parser turns a raw dump into a `DiagnosisInput`. See
[`parsers/vllm.py`](../parsers/vllm.py) for the fullest example.

1. **Write `parse_<source>(text, model_name, gpu_type, source_file) -> DiagnosisInput`**
   plus a `parse_<source>_file(path, **kwargs)` wrapper. Populate the canonical
   fields you can extract; leave the rest `None`. Append notes to `parse_warnings`
   rather than raising on recoverable issues; record provenance in `source_files`.
2. **Never invent values.** If a field isn't in the dump, leave it unset — rules
   guard for `None`, and the renderer omits absent fields.
3. **Respect field scoping.** Because the merge is first-non-None-wins, don't write a
   field with a differently-scoped value than another parser uses (e.g. vLLM keeps
   per-token TPOT in `decode.latency_ms`, not `decode.duration_ms`, so it doesn't
   clobber Nsight's phase duration). See `schema/merge.py`.
4. **Wire it into the CLI** — add a flag and a branch in `_load_sources`
   ([`cli/main.py`](../cli/main.py)) for one-shot use, and/or a `Source` in
   [`collect/sources.py`](../collect/sources.py) for live `watch`.
5. **Add tests** under `tests/parsers/` with a representative fixture.

---

## Add a fix (make a rule actionable by `gait`)

A fix lets `strided fix` turn a diagnosis into a concrete config change. The registry
is in [`gait/fixes.py`](../gait/fixes.py); r03 is the worked example.

Register a `FixSpec` keyed by the rule id:

```python
_R04 = FixSpec(
    rule_id="r04",
    param="some-flag",                       # the config knob to change
    rationale="Why this change addresses the diagnosis.",
    applicable=_r04_applicable,              # (Diagnosis, Snapshot) -> Abstained | None
    propose_value=_r04_propose_value,        # (current) -> new value
    predict=_r04_predict,                    # (Diagnosis, Snapshot, current, proposed) -> Prediction
)
FIX_REGISTRY = { ..., _R04.rule_id: _R04 }
```

The pieces:

- **`applicable`** is a gate: return an `Abstained` if this snapshot isn't a shape
  the fix applies to (e.g. r03's fix only applies to paged vLLM/SGLang engines),
  else `None`. gait would rather stop than apply the wrong fix.
- **`propose_value`** sizes the new value from the current one (read from the config
  target), e.g. r03 halves the block size.
- **`predict`** returns a `Prediction` of checkable effects, recorded *before*
  acting, that `verify` will test the post-change snapshot against. This is what
  makes verification honest.

The change is located and written through a **`ConfigTarget`**. Today the only one is
[`VllmArgsTarget`](../gait/targets/vllm_args.py), which models a vLLM launch command
as editable `--flag value` pairs (and knows vLLM's defaults, so an unset-but-defaulted
flag resolves rather than reading as missing). To fix a knob that lives somewhere else
(a YAML file, an env var), implement the `ConfigTarget` protocol in
`gait/targets/base.py` for that surface.

Add tests under `tests/gait/` covering the applicable-gate abstention, the proposed
value, and the prediction.

---

## Run the suite

```bash
pip install pytest
python -m pytest -q          # full suite
python -m pytest tests/rules -q   # just your new rule
```

## Architecture reference

- [Engine architecture](engine/ARCHITECTURE.md) — how rules are fired, partitioned,
  ranked, and reconciled; the determinism and security guarantees you're extending.
- [`rules/base.py`](../rules/base.py) — the full rule contract with docstrings.
- [`gait/`](../gait/) — the fix state machine; `gait/__init__.py` documents its
  invariants (the type-enforced approval gate chief among them).
