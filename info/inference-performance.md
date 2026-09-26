# Model performance workflow

Roofline measures production workloads and components, stores their evidence, and
provides a read-only CLI and Textual tree. Workers own preparation and execution;
separate CLI invocations reuse compatible resident resources automatically.

The contracts are [Roofline](../design/inference/roofline.md) and
[formula execution and measurement](../design/inference/formula-execution.md).
The [Roofline README](../inference-v3/roofline/README.md) documents installation,
local configuration, command options and worker directories.

## Measure and browse

Define exact model artifacts and machine connections in Roofline's gitignored
`models.json` and `targets.json`. Model IDs include container and quantization,
for example `qwen3.5-35b-a3b:gguf:q4_k_m`. Each model declares its checksum and
absolute locations by target. Those definitions are independent of product catalogs.

```sh
roofline measure --model qwen3.5-35b-a3b:gguf:q4_k_m \
  --context 16k --steps 128 --scope decode --targets m4-pro-01,m4-pro-02
roofline query --model qwen3.5-35b-a3b:gguf:q4_k_m
roofline
```

Measurements publish automatically, including failed checks and partial outcomes.
The required performance system derives one hardware-parameterized roofline from
the model's formula composition. Measurements across workloads and machines inform
the same model, with normalized performance and propagation through enclosing
formulas. Queries and the TUI require only a model selection.

The shared analytical package now drives model queries and the model-only tree.
Ops publications retain the numerical graph, explicit units, resource obligations
and measured execution mappings. Numerical qualification checks independent
formula arithmetic at production operation inputs, then checks compressed state
against an independent byte-exact encoding of those verified values. Diagnostic
replay is excluded from performance samples.
The [implementation spec](../specs/26-09-16/formula-performance-model.md) defines
the contracts, derivation, evidence propagation and completion gates.


## Optimize a component

Discover actual formula selectors with `roofline scopes --model MODEL`, or use a
selector from a recorded measurement. Select one using `measure --scope SELECTOR`;
`--step 0` restricts a decode component to the first declared workload position.
Cold workers prepare its real production inputs automatically. A component request
does not require a previous model measurement or a user-managed capture.

Edit ordinary authored operations and repeat the command. `query --measurement ID`
returns exact timing, correctness and diagnostic evidence. `query --artifact ID`
reads generated source or other recorded payloads. No experiment-specific driver
or publication code is needed.

```sh
roofline compare --baseline MEASUREMENT_A --candidate MEASUREMENT_B
roofline measure --model MODEL --scope SELECTOR --step 0 \
  --against-source SOURCE_ID
```

`compare` reads existing observations. `--against-source` requests fresh bounded
paired samples, alternating order on the same physical device. Pairing requires
compatible inputs and safely replaceable implementations; unsupported environment
changes remain unavailable. Independent historical samples are never called paired.

After a component improvement, measure the enclosing workload. Isolated latency,
in-parent contribution and whole-workload wall time have different boundaries.
A layer-zero saving cannot be extrapolated to every layer or position without
supporting evidence. Native interval unions avoid counting overlaps twice.

## Remote execution

```sh
roofline workers setup m4-pro-01
roofline workers setup m4-pro-02
roofline workers setup sparky
```

Workers receive immutable source and fixture bytes over SSH. They install and run
under their own user directories, defaulting to `~/.local/share/roofline`, without
using existing engine checkouts. Model files remain external, checksum-verified
inputs. The coordinator collects results even after the submitting CLI exits.
An SSH reconnection reconciles the original attempt rather than submitting again.

Pause prevents new acceptance and drains existing work. `roofline cancel --request
ID` cancels a submission; Ctrl-C only detaches the CLI. Hosts execute independently;
comparisons keep hardware and protocol differences visible.

## References, resource evidence and history

`--engine llama.cpp` selects the native reference integration. It preserves its
actual timer boundaries and states missing independent checking and formula
correspondence. Reference timings are observations, not hardware ceilings.
`roofline characterize TARGET` explicitly runs bounded resource probes. Ordinary
measurement and browsing never silently calibrate hardware. Formula-derived
resource analysis preserves its assumptions and missing evidence.

Session bench separately owns HTTP schedules, concurrency and response validation.
It does not publish into Roofline automatically. `roofline import-session DIRECTORY`
reads its completed records, matches exact artifact checksums, and retains native
and HTTP timers as separate measurements. Response validation does not establish
independent numerical correctness. Unmatched records remain available as artifacts;
missing source or token/state identity is never invented.

Use `roofline export --measurement ID --output BUNDLE` and `roofline import BUNDLE`
for selected evidence transfer. Import verifies content without executing recorded
code and preserves original chronology. Model weights and external toolchains are
explicit replay requirements. Losing a worker cache does not remove stored history.
