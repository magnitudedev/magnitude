# Working on inference

## Workflow

**Fix the system, not the symptom.** Make engineering decisions: change what in the design causes
the problem, and redesign if needed. Patching a symptom leaves the cause in place, so the next
symptom appears and gets patched too, and the work gets stuck iterating in a loop.

How to run checks and measurements is in the root `AGENTS.md` ("Running checks and
measurements"); runs on hardware go through `mrig`.

## Autonomy

**Act with full autonomy.** Make the decisions the work requires and keep going. Do what has been
authorised without asking again, and bring something back only when it is not yours to decide.
Stopping to ask halts the work until someone answers.

**Clear space with `cargo clean` automatically.** When a local disk is full or filling, run
`cargo clean` on stale target directories at once, without hesitation and without asking. Build
output is regenerable. On the hosts, `mrig` evicts the least recently used checkouts itself.

## Tuning parameters and defaults

The tuning budget never grows. Keep declared spaces to configurations that could win.

- **Remove a configuration only when both of these hold:**
  1. *Reason:* reading the kernel explains why it cannot be the fastest, in terms of a quantity
     computed from the shape and parameters, so the reason covers shapes not yet measured.
  2. *Evidence:* tuning history shows it was never the fastest on any unit of that backend.

  A reason without evidence, or evidence without a reason, is not enough.
- **Write a removal as a `where` constraint or shorter value list**, with the reason and numbers in
  a comment.
- **The default is the first admitted configuration:** put first the value with the smallest
  worst-case gap to the measured best.
- **Judge each backend on its own devices**, and recheck for new GPU or model families.

## Measurement

- Model quality and speed are measured only through `mrig`. Its timed instruments hold the
  device alone and pass the thermal gate first.
- A difference between two builds is evidence only from `mrig ab`: both sides on one host, in
  alternating rounds, judged against what each side resolved.
- A run records the source it measured (`rig.json` in its results); check it is the tree you
  intend.
