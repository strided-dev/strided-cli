# Engine architecture

The engine is the **fan-in** half of the pipeline. Parsers fan out (three
sources → one schema); the engine takes one `DiagnosisInput` and the registered
rules and produces one ranked, annotated `DiagnosisReport`.

It is the only component that knows rules exist *as a set*. Every rule is, by
contract, blind to every other rule (`rules/base.py` forbids cross-rule I/O and
mutation, and rules are unit-tested in isolation). So anything **relational** —
conflict, corroboration, ranking — lives here, never in a rule.

The engine returns **data, never printed output.** Formatting is the CLI's job.

## Modules

| Module | Responsibility |
|---|---|
| `registry.py` | `ALL_RULES`: the explicit, ordered list of rules to run |
| `relations.py` | Declarative conflict / corroboration tables (data) + queries |
| `confidence.py` | Relational confidence policy (the only engine-produced number) |
| `runner.py` | Orchestration + the report value objects; `run_diagnosis()` |

Dependency direction is strictly one-way and acyclic:

```
runner → {registry, relations, confidence} → rules / schema
```

## Pipeline (`run_diagnosis`)

1. **Scan input** for non-finite floats (NaN/inf). Warn; raise under `strict`.
2. **Fire** every rule in registry order; partition results into diagnoses,
   insufficient-data notes, silently-dropped below-threshold, and errors.
3. **Resolve conflicts** (opt-in) — keep the highest-confidence member of each
   conflict set, record the rest as suppressed.
4. **Annotate + rank** survivors. Sort key `(-adjusted_confidence, rule_id)` is
   a total order, so output is independent of firing order.

Conflict resolution runs **before** corroboration: never boost a diagnosis
about to be suppressed, and never let a suppressed loser corroborate a survivor.

## Three design pillars

### Security (threat model: a local, single-shot CLI, not a service)

- **Untrusted input.** The schema is the trust boundary. Pydantic permits
  NaN/inf by default, which breaks deterministic ranking, so the engine scans
  for them, and since schema 1.5.0 the schema itself sets `allow_inf_nan=False`,
  so the scan is defense-in-depth.
- **No arbitrary code execution.** The registry imports rules by name. There is
  no filesystem auto-discovery, so a dropped file cannot be imported.
- **No data leakage.** No network, no disk, no telemetry, no logging of input.
  `RuleError` records the exception **class name only** — never the message or
  traceback, which can embed customer metric values. The report is never
  persisted by the engine; persistence is the CLI's explicit decision.
- **No new dependencies.** Pure stdlib + the existing `schema`.

### Performance

The engine fires ≤10 rules of scalar arithmetic over one parsed input; parsing
is the bottleneck, not this. Therefore: **sequential, not parallel** (pool
overhead and the GIL would make it slower and less deterministic), single-pass
partition, no shared derived-feature cache (premature). A perf budget belongs in
tests rather than in assumptions.

### Accuracy (the v1 stance: *rank and annotate, do not invent*)

The engine must never degrade rule accuracy against the Day-60 60% kill
criterion. So the two "smart" behaviours ship **dark behind flags, off by
default**:

- `enable_corroboration_boost=False` — corroboration is annotated, but the
  confidence scalar is unchanged. An invented boost could promote a wrong
  diagnosis above a right one.
- `enable_conflict_suppression=False` — conflicts are annotated and the loser is
  ranked below by confidence, but nothing is deleted. Hard suppression can hide
  a correct diagnosis when the (unvalidated) conflict table is wrong.

Both are fully built and tested; they turn on only once real customer dumps
validate them. When enabled, the boost is bounded (`+0.05` per corroborator,
capped at `0.97 < 1.0`), honouring r01's self-imposed 0.9 ceiling, which exists
to reserve exactly this headroom.

## Determinism

Same input → identical report. Guaranteed by explicit registry order, a
total-order sort key, the non-finite guard (so the sort is well-defined), and no
reliance on wall-clock, randomness, or dict iteration order.

## Known contract gap

`Abstention.INSUFFICIENT_DATA` carries no payload, so the engine can report
*that* a rule could not run but not *which* field was missing. Closing this
needs an extension to `RuleResult` in `rules/base.py` (sacred) — proposed, not
done unilaterally.
