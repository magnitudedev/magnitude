# Working on inference

## Workflow

**Fix the system, not the symptom.** Make engineering decisions: change what in the design causes
the problem, and redesign if needed. Patching a symptom leaves the cause in place, so the next
symptom appears and gets patched too, and the work gets stuck iterating in a loop.

**Build fully before validating.** Take a change to its fullest extent before checking it, unless
it is a proof of concept. Validation is expensive and slow, and validating partial work means
paying for it again every time the work changes.

**Use the smallest and fastest validation available.** Validation takes time, and time spent
waiting on a larger check than the question needs is wasted.

**Validation is run only out of necessity.** A validation is only ever run if that specific data
point is necessary to inform continued development. That is the exclusive condition: either it is
necessary to confirm that a particular piece works so that things can keep getting built, or to
confirm that, for example, a commit is ready. Longer validations and benchmarks fall under the same
condition. They are expensive and take a long time, and a run that is not necessary buys nothing
for that cost.

## Machines

**Use every machine available.** Builds, tests and device runs go on the remote machines, spread
across all of them, so that independent work runs at the same time. An idle machine is time lost.
Keep expensive work off the local machine.

## Autonomy

**Act with full autonomy.** Make the decisions the work requires and keep going. Do what has been
authorised without asking again, and bring something back only when it is not yours to decide.
Stopping to ask halts the work until someone answers.

**Clear space with `cargo clean` automatically.** When a disk is full or filling, run `cargo clean`
on stale target directories at once, without hesitation and without asking. Build output is
regenerable, and a full disk stops every job on that machine.

## Measurement

- Model quality and speed are measured only through `inference/evaluation`.
- Timed measurements need the host to themselves, with the evaluation flow's lock and thermal gate
  in place.
- A small difference from the baseline taken in separate sessions is not evidence. Confirm it with
  both sides alternating in one session.
- Make sure a run measures the tree you intend: check its recorded source identity.
