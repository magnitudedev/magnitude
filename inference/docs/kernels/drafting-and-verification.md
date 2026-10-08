# Efficient drafting and verification

Speculative decoding drafts tokens cheaply and verifies them with the target model. For a given
drafter, its acceptance rate is the drafter's property; the engine's job is to make everything
else free. Three guarantees do that, for every speculative method (an embedded MTP head, a
separate drafter such as DFlash, a drafter that reads the target's history) and on every device.
The governing contract is
[speculative generation](../../../design/inference/engine/speculative-generation.md).

## The three guarantees

1. **Prefill pays nothing for drafting.** A prompt's prefill and time to first token with a
   drafter equal plain's.
2. **Verification costs one decode step.** Verifying a round's drafted tokens costs the same as
   decoding one token, however many tokens it checks.
3. **Drafting costs only the drafter's own reads.** A draft step costs the bytes the drafter must
   read, and nothing more.

Together they fix decode speed: a round costs one target step plus the drafter's reads, and
emits the accepted drafts plus one target token. Nothing else is left to lose.

The one obligation on acceptance itself: on the same weights and prompt, a drafter's acceptance
matches the reference implementation's. A shortfall is a defect in how the engine runs the
drafter, not a property of the drafter.

## 1. Prefill pays nothing for drafting

**Why it is achievable.** During prefill, a prompt row's only lasting product is its state: the
target's attention history and recurrent state, and the drafter's own history. Outputs (logits,
features, drafts) are read only for the final prompt row. So a drafter's prefill work reduces to
writing its own state, which is small, and that small work can be placed where the device is
idle.

**How it is achieved**, in order of preference:

- **Don't need it.** A drafter that reads the target's history writes nothing for prompt rows.
- **Write state only.** A history row computes only what writes state a later step reads: a
  drafter head's input projection and key/value write, never its attention, feed-forward or
  readout. The same rule holds for the target: layers after the last one that writes state, and
  that layer's output work, do not run on prompt rows.
- **Demand per row.** Each method declares what each row needs (state, features or logits), and
  the executor computes exactly the union. No method demands outputs from every prompt row.
- **Decide forms statically.** State-only work is its own launch form, chosen when the launch is
  chosen; never a full launch that returns early.
- **Run it in idle capacity.** The drafter's state write is issued on the device alongside the
  prompt chunk, batched over the chunk's rows, never row by row and never through a host read.
- **Defer it.** Whatever cannot be hidden in prefill moves into decode, where arithmetic is idle:
  the first rounds go undrafted while the drafter catches up.

The only case where the cost cannot be hidden is a device saturated in arithmetic during both
prefill and decode, such as a large concurrent decode batch. There the cost is the drafter's
state share, and speculation's value is low anyway.

**How it is validated.** Every speculative benchmark row is paired with a plain row on the same
build, device and request. Prefill throughput and time to first token must match within
measurement noise. A per-chunk attribution separates the drafter's entry from the target's work;
any entry time on the critical path is a defect.

## 2. Verification costs one decode step

**Why it is achievable.** Decode is bound by memory bandwidth: a step reads every weight to
produce one token, and most arithmetic is idle. Verifying several tokens is a decode step with
more rows. Every row uses the same weights and the same history, so the step reads the same
bytes; the extra rows cost only arithmetic, which is idle.

**How it is achieved.**

- **Read shared data once.** Every multi-row kernel reads weights and history once for all its
  rows. This is a property of every form a kernel offers, never a tuning outcome: the default for
  a multi-row class already shares, and tuning only chooses among forms that share.
- **Stay bandwidth-bound across the widths used.** Extra rows are free only while their
  arithmetic fits in the time the memory reads take. Where a scalar form stops fitting, the
  kernel switches to a form on the device's matrix units; the switch point is measured per
  device.
- **Tune verify widths like decode.** The row counts verification uses are tuned with the same
  priority as single-row decode, because they serve the same share of decode time.
- **Read out only what is checked.** The readout over verify rows reads no more of the output
  head than single-row decode does.

**How it is validated.** A per-round attribution compares the verify step with a plain step at the
same history length; the ratio should be close to one. Per kernel, achieved bandwidth is compared
with the device's limit. A verify kernel whose time grows with rows while its bandwidth stays
below the limit is either re-reading shared data or bound by its arithmetic form.

## 3. Drafting costs only the drafter's own reads

**Why it is achievable.** A draft step is a small pass of the drafter: its own layers, its own
history, a readout over its vocabulary. Its necessary cost is the bytes it reads. Draft steps are
serial, each depending on the last, so what inflates them is overhead between steps, not
arithmetic.

**How it is achieved.**

- **Keep the chain on the device.** Drafts are proposed device-resident, one step feeding the
  next, with no host read or synchronization inside a round.
- **Minimize launches.** A draft step is fused into as few launches as possible; gaps between
  small dependent launches cost more than their arithmetic.
- **Minimize the drafter's bytes.** The drafter reads the smallest vocabulary and weight form that
  keep its acceptance.
- **Overlap what does not depend on the outcome.** Work that does not depend on the current
  round's verification, such as drafting the next round along the likely path, runs alongside
  it.
- **Draft as wide as pays.** A further draft is added while the expected value of its acceptance
  exceeds its draft cost plus its verify row. Width follows from measured acceptance and measured
  costs on the device in use; it is never a fixed constant.

**How it is validated.** On a workload where nearly every draft is accepted (prose-repeat), only
cost remains. A per-round attribution gives each draft step's time; it should equal the drafter's
bytes at the device's bandwidth. The measured decode gain is compared with the gain the three
guarantees imply for that width; any shortfall is assigned to the guarantee it breaks.
