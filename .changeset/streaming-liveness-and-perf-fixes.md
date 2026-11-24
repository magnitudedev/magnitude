---
"@magnitudedev/cli": patch
---

Fix streaming deadlocks, CPU livelocks, and high-acceptance MTP head crashes across all backends.

- Fixed mid-generation stalls and deadlocks during elastic expert-bank resizes and memory reclaim (including synchronous host/expert reclaim during active decoding, queued link promotions and write-backs during traffic holds, and unconditional liveness publication across resize turns).
- Fixed `expert-requests`, cross-layer lookahead, and teardown CPU livelocks after host-tier reclaim, eliminating unnecessary background CPU burn and NVMe read churn.
- Fixed async-DMA prefill deadlocks when trailing prompt steps are skipped on late streamed layers.
- Fixed `seismic-io` read-completion lost wakeups under lock contention and eliminated unbounded CPU-site wait spins.
- Fixed native head routing buffer sizing for chained MTP drafting steps under high draft acceptance.
