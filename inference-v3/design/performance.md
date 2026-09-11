# Performance

**Measurement builds the same composition production runs, measures it under one
mechanism, validates before it assesses, and records what was selected.
Production depends on none of it.**

## A run

```text
composition (data) + workload ──build──► component ──prepare once──► invoke × n ──► complete
                                                                          │
                                        validate against an independent reference
                                                                          │
record: compiler build · composition digest · source digest · machine · runtime versions ·
        workload · metrics · realized selection · thermal trace
```

| Rule | Reason |
|---|---|
| The component is built from the composition, not from a benchmark's own construction | A benchmark that assembles its own object measures something else |
| One machine-wide lock | A timing never shares the device with a test or another timing |
| Validation precedes assessment | A fast wrong answer is not a result |
| The source must not change during a run | The record names one revision |
| Observation, model and comparison are separate fields | A latency carries its pass and boundary; a bound is a formula; neither is mistaken for the other |
| The plan's selection is in the record | A change in what was chosen is distinguishable from a change in a kernel |

## Whole-model measurement

A decode measurement includes no prefix construction and shares no live KV tail
with a checkpoint: the case retains an accepted history and aborts each measured
advance after completion, so every sample starts from the same state. The
reference is independent logits from the same artifact.

## External evidence

Session bench measures serving with simulated agent sessions over disposable
server processes. It is opaque to the engine and imported as external evidence;
native service diagnostics are never substituted for client-observed latency.
Its inputs are the [benchmark fixtures](benchmark-fixtures.md).

## Where numbers live

Raw runs under `runs/`; written-up baselines under `results/`. Design documents
carry no numbers. A claim about performance points at a record, and a record
that cannot say what was selected, on what, from which source, is not evidence.
