---
applies_to:
  - packages/agent/src/services/fs.ts
  - packages/agent/src/tools/fs.ts
---

# Filesystem search lifecycle

Filesystem search owns one bounded ripgrep subprocess for the lifetime of the search Effect.

- Stdout and stderr are consumed concurrently. Match output and retained stderr are bounded. At
  most 8 KiB of stderr is retained while draining the stream, and agent-visible process diagnostics
  contain only the first nonempty line capped at 300 characters with an explicit truncation marker.
- Exit code zero returns matches; exit code one returns an empty successful result. Any exit that
  emitted usable matches returns those partial results. Other exits are actionable typed failures.
- Reaching the match limit intentionally terminates and reaps ripgrep, returning the collected
  matches successfully.
- Deadline, interruption, read failure, and process failure cancel stream consumption and terminate
  and reap any live child. Natural completion is never followed by a kill signal.
- Ripgrep is invoked with an argument vector, never through shell interpolation.

## Acceptance criteria

1. Every non-interrupted search-owned outcome settles within the configured deadline plus bounded
   termination grace.
2. Every settled grep tool execution emits exactly one terminal tool event.
3. No search leaves a live or unreaped ripgrep child.
4. Timeout and process failures retain enough detail for the agent to choose a corrective action,
   while ripgrep process diagnostics exposed to the agent never exceed 300 characters.
5. An unreadable path cannot discard matches successfully collected from readable paths.
