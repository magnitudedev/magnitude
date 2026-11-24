---
"@magnitudedev/cli": patch
---

Fix streaming deadlocks, short-prompt decode panics, CPU livelocks, and high-acceptance MTP head crashes across all backends.

- Fixed `LandingList::append` overflow panics (`landing entry K of a list of K`) on short multi-token prompts and multi-row verification batches, and aligned decode bank floors, boundary-parked staging, and chained head site slots to worst-case decode capacities.
- Fixed mid-generation stalls and deadlocks during elastic expert-bank resizes and memory reclaim (including synchronous host/expert reclaim during active decoding, queued link promotions and write-backs during traffic holds, and unconditional liveness publication across resize turns).
- Fixed `expert-requests`, cross-layer lookahead, and teardown CPU livelocks after host-tier reclaim, eliminating unnecessary background CPU burn and NVMe read churn.
- Fixed `seismic-io` read-completion lost wakeups under lock contention and eliminated unbounded CPU-site wait spins.
- Fixed native head routing buffer sizing for chained MTP drafting steps under high draft acceptance.
