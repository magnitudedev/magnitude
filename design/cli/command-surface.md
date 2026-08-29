---
applies_to:
  - cli/src/commands/**
  - cli/src/index.tsx
  - cli/src/server/service.ts
  - cli/src/agent-docs/**
  - packages/acn-protocol/src/boundary/**
  - packages/acn/src/model-commands.ts
  - packages/acn/src/boundary/acn.ts
  - packages/client-common/src/local-models/service.ts
---

# Non-interactive CLI contract

The non-interactive CLI has one public command vocabulary:

```text
update
service install | uninstall | start | stop | status
catalog list | pull <model-id> | remove <model-id> | cancel <model-id>
models status | load <model-id> | stop
connections list | add <harness> [--set-current <model-id>] | sync [harness] | remove <harness>
docs [topic-id]
```

`catalog` owns discovery and on-disk acquisition. `models` owns runtime
residency. The zero-argument `models stop` operation stops the current active
local model; it is not model- or slot-addressed. Removed command names are not
aliases. Running `magnitude` without a subcommand remains the interactive
entrypoint.

`magnitude setup` is a second interactive entrypoint that opens onboarding
setup directly. It is intentionally outside the non-interactive result and
JSON contracts described below.

## Result contract

Every non-interactive leaf command produces one typed, Effect-Schema-backed
result before presentation. Human output and JSON output are renderings of
that same result. `--json` is inherited globally and may occur before or after
the subcommand. The established `update` and `service start` terminal workflows
remain outside this contract until their JSON behavior is designed separately.

A successful JSON command writes exactly one compact JSON document and one
trailing newline to stdout, writes nothing to stderr, and exits successfully.
A failed JSON command writes nothing to stdout, writes exactly one document of
the form `{ "error": { "code", "message", "retryable" } }` plus a trailing
newline to stderr, and exits unsuccessfully. Logs, ANSI presentation,
subprocess output, and transient progress never contaminate JSON streams.
Exact machine values such as byte counts and progress fractions remain numeric;
human renderers own unit and percentage formatting.

## Service operations

Installation state and runtime state are independent. `service install` writes
and enables the exact per-user service definition without starting or replacing
the runtime. It never overwrites a definition owned by a newer compatible
running release. `service start` installs or refreshes the definition, starts or
joins the service through the shared inline lifecycle, performs concurrent update
discovery, and succeeds only after public readiness. `service stop` stops the
exact Magnitude runtime, including a JIT-owned runtime, without disabling or
deleting the definition. `service uninstall` stops the runtime, disables and
deletes the definition, and preserves user data. `service status` observes and
reports installation, enablement, platform-manager ownership, runtime, and
revision facts separately.

## Model operations

Pull is the one acquisition verb. It converges a model to present-and-current
on disk: it installs from `NotInstalled` or `InstallFailed`, updates from
`UpdateAvailable` or `UpdateFailed`, and succeeds by reporting `AlreadyCurrent`
for an installed, current model — under pull's contract that is the goal state
already reached, not a masked error. Pull during active acquisition work is
rejected as busy, and pull during removal states is rejected as unpullable;
genuinely wrong-state requests still fail as typed domain rejections rather
than succeeding as no-ops. Cancel is valid only while that model has admitted
pull work. Remove is valid only for an installed family with no active
transfer, removal, or resident instance. Acquisition admission may outlive the
calling client and is observed through the catalog projection.

`catalog list` exposes the authoritative local discovery and assessment state,
including incomplete assessment counts, provenance, compatibility,
acquisition, and residency. `models status` is a filtered projection of those
same local product rows; it does not define a second state model.

## Conformance

- Command-tree tests prove the exact nouns, verbs, and argument arity.
- Every leaf has success and failure JSON tests that parse exactly one document.
- Acquisition tests prove pull-state validation and admitted work ownership.
- Lifecycle tests distinguish install, start, stop, uninstall, and status.
- Removed nouns and verbs fail command parsing and are never retained as aliases.
