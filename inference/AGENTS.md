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

## Validation

**Run every validation in the minimum time needed to obtain its information.** The information
dictates the length, not you. If you could break it up or run it shorter, it is too long.

**Spend time only on information you need to implement further.** Obtain exactly that. Collect
anything more only when it costs you no additional time; never spend time on information you do not
need to continue.

You can check your results and decide your next step from them, so validate in short runs that each
answer what you need now. A sweep or matrix pays up front for answers to questions you do not yet
know you have, and blocks you and every other agent on the machine while it runs.

## Machines

**Use every machine available.** Builds, tests and device runs go on the remote machines, spread
across all of them, so that independent work runs at the same time. An idle machine is time lost.
Keep expensive work off the local machine.

**Take a host through its lock, by pool.** Run on the machines only through the evaluation flow,
never over ssh directly, and name a pool rather than one host unless the work needs that specific
machine. A pool run takes whichever member frees first. `magnitude-remote status` shows who holds
and who waits on each host: the agent session, its checkout, and its run.

**Hold a machine only while it is doing work you need.** Every other agent waits while you hold it.
Know how long each of your runs should take, and set its maximum hold to that. A run that goes
past it is broken: stop it and find the cause. You are responsible for every run you have
started, including ones whose results you no longer need; stop those at once.

## Autonomy

**Act with full autonomy.** Make the decisions the work requires and keep going. Do what has been
authorised without asking again, and bring something back only when it is not yours to decide.
Stopping to ask halts the work until someone answers.

**Clear space with `cargo clean` automatically.** When a disk is full or filling, run `cargo clean`
on stale target directories at once, without hesitation and without asking. Build output is
regenerable, and a full disk stops every job on that machine.

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

- Model quality and speed are measured only through `inference/evaluation`.
- Timed measurements need the host to themselves, with the evaluation flow's lock and thermal gate
  in place.
- A small difference from the baseline taken in separate sessions is not evidence. Confirm it with
  both sides alternating in one session.
- Make sure a run measures the tree you intend: check its recorded source identity.
