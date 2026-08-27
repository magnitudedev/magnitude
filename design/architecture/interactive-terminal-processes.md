---
applies_to:
  - packages/utils/src/process/**
  - packages/launcher/src/cli-process-spawner.ts
  - cli/src/runtime/interactive.tsx
---

# Interactive terminal processes

An interactive process handoff transfers one existing terminal from Magnitude to another
terminal application. It is distinct from background command execution and uses one shared
process primitive across every launch boundary.

The child inherits standard input, output, and error directly and is invoked with an executable
and argv, never through a shell. It also inherits the caller's terminal association: on POSIX it
remains in the existing foreground process group, and on Windows it remains attached to the
existing console. An interactive child must not be detached into a new session or process group.
Consequently, terminal-generated events such as resize and job-control signals reach it through
the operating system rather than through application-level resize forwarding.

The caller must release any renderer, raw-mode ownership, and alternate-screen state before the
handoff. Magnitude and the child must never render concurrently. The process primitive owns the
child until it exits, reports normal and signal termination distinctly, and terminates and reaps
the child when its owning Effect scope is interrupted.

This contract governs both the npm launcher handing the terminal to the native Magnitude CLI and
the native CLI handing it to an external harness. General-purpose command executors remain valid
for non-interactive subprocesses but do not satisfy this contract.

## Required guarantees

- Terminal resizes reach an interactive child at both launch boundaries.
- Shrinking and growing the terminal repeatedly does not require polling or application-level
  resize relays.
- Arguments are passed literally and cannot be interpreted by a shell.
- The child's exit status is preserved, including termination by signal.
- No renderer remains active when an external harness begins rendering.
