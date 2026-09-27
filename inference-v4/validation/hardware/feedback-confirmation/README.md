# Production feedback confirmation qualification

These experiments execute the production Rust median-interval and confirmation
state machine. They do not use the standalone Python tuning prototype or a model
workload. The simulation observes no device and makes no hardware speed claim.

From `inference-v4`, run sequentially:

```sh
cargo test -p seismic-compiler --lib feedback
mkdir -p validation/results/feedback-search
cargo test -p seismic-compiler --lib seeded_iid_and_adversarial_confirmation_qualification -- --ignored --nocapture > validation/results/feedback-search/confirmation-statistics.log 2>&1
```

The deterministic simulation uses 256 independent seeded replications for each
of six distributions. Each replication starts a fresh campaign and evaluates two
content events, numbered 1 and 2. Each event uses separate pseudo-random streams
for incumbent observations, challenger observations, and pair ordering. These are
synthetic IID draws in the first four cases; no measured latency enters them.
All samples are integer nanoseconds.

| Distribution | Construction | Interpretation |
|---|---|---|
| Equal uniform | Both arms uniform over integers 500 through 1500 | Null: equal median |
| Equal skewed | Both arms use the same monotone cubic transform of uniform draws | Null with right skew and ties |
| Faster | Challenger is 0.8 times an independent uniform draw, rounded down | Real median improvement |
| Slower | Challenger is 1.2 times an independent uniform draw, rounded down | Inferior challenger |
| Correlated | Incumbent is 1000; challenger repeats one latent uniform draw for the entire campaign | Equal marginal medians, violated within-event independence |
| Drift | Both arms acquire an extra 1000 after the first 20 observations | Violated stationarity; no confidence guarantee |

The report records both-case promotions, any single-case improvements,
inconclusive events, and median/maximum total observations across the two cases.
A single-case improvement is a false finding only in the equal or slower cases.
Every simulation runs to a decision or the production cap of 4096 observations
per arm, rather than truncating difficult cases to make power look better.

The quantitative results are deterministic regression evidence, not a proof of
coverage or an estimate of device-wide speedup. The formal error accounting is
conditional on fixed IID arm/event distributions: the binomial order-statistic
interval is allocated `delta / (2*i*(i+1)*2^(k+1))` per arm, with campaign delta
0.05. Summing over both arms, all events and all stages is at most 0.05.
The tests independently verify the finite telescoping sum and small exact
binomial intervals. The reported adversarial cases deliberately violate the
premise; their findings must never be described as covered by that guarantee.

Additional normal tests exercise sample caps, interruption/resumption of pair
ordering, separate event/seed sample pools, and actual controller behavior:
winning the first content case alone cannot promote; losing the second defeats
promotion; re-nomination starts fresh events; environment changes between cases
discard the comparison without resetting campaign event numbering.
