# Seismic / inference V4 implementation

Follow the master and accounting specifications linked in README.md. V3 is the
behavioral and numerical reference; do not redesign its policies while porting.

- Fix compiler, tooling and abstractions when they obstruct natural kernels. Do not
  contort kernel source or bypass Seismic to obtain numerical execution.
- Derive accounting from checked computation and candidate realizations. Selection
  must use resource/dependency models and applicable measurements, never heuristic
  scores, magic thresholds or model/device-name performance exceptions.
- Keep exact quantities, bounds, estimates, measured results and unknowns distinct.
  Missing information is not zero. Preserve predictions before measurement.
- Validate correctness and enclosing performance. The interpreter is a reference,
  not a production CPU backend. Existing smoke tests do not establish engine parity.
- Update applicable `design/` contracts together with intentional behavior changes.
- Maintain validation/continuation.md and qualification evidence as work proceeds.
- The user authorizes infrequent local commits at verified, coherent milestones,
  with concise informative messages. Review and stage only intended changes.
  **Never push.** Do not incorporate another agent's unfinished work.

Run targeted Rust tests while iterating, then workspace/remote qualification at the
appropriate gates. Preserve source and artifact identities for comparisons.
