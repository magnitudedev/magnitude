# Magnitude integration protocol

Public Effect Schemas for Magnitude's versioned CLI model commands and optional inference progress.
This package has no dependency on a harness, the daemon, or private workspace packages.

Consumers validate required fields, command identity and schema version, and ignore unknown fields.
Adding optional information is compatible; removing or changing a required field or its meaning
requires a new wire version. `fixtures/v1.json` contains the shared compatibility examples.

Progress and timings are observational. Timings are cumulative snapshots within a request, and
transport EOF is not proof of inference success. The harness's native parser owns the outcome.
