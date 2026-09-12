---
applies_to:
  - packages/utils/src/process/**
  - packages/launcher/src/cli-process-spawner.ts
  - scripts/dev-pi.ts
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

This contract governs the npm launcher invoking the headless native Magnitude CLI and the
explicit development launcher invoking Pi. The native CLI does not launch an agent harness or
render onboarding. Its inherited terminal streams remain useful for normal command output and
signal propagation.

Pi's `/magnitude-setup` invokes the finite `magnitude app open` command with piped output while
Pi retains its terminal. The desktop owns onboarding. Successful command completion acknowledges
that the window opened; it does not prove model setup or connection completion. The user connects
Pi in the desktop and explicitly reloads Pi afterward. This navigation command is not an
interactive terminal handoff.

## Required guarantees

- Terminal resizes reach an interactive child at each interactive launch boundary.
- Shrinking and growing the terminal repeatedly does not require polling or application-level
  resize relays.
- Arguments are passed literally and cannot be interpreted by a shell.
- The child's exit status is preserved, including termination by signal.
- No renderer remains active when an external harness begins rendering.
