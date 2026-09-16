# Performance comparison

Compare ordinary production Qwen 3.5 35B A3B Q4_K_M forwards on the same Apple M4 Max, artifact, compiler libraries, input token IDs, physical row capacity, and state horizon. The baseline uses the saved pre-edit kernel sources; its compiler streaming module is the original HEAD version. The candidate uses working sources. No compiler or device tuning is enabled.

Each run creates fresh sequences, executes 32 prompt tokens followed by two single-token decode advances, excludes two complete warmup sequences, and records nine samples per stage. Timing covers model preparation through completion, including ordinary dispatch and allocation. Reading logits and committing state happen afterward. Native instrumentation is disabled.

Investigate a median slowdown above 5%, or median absolute deviation above 5% of the median, with another paired run before accepting the change. Compare the complete saved logits as well as timings. Do not count a speedup unless repeatable; this check is a regression gate. Any unexplained numerical difference or repeatable slowdown beyond that threshold fails the gate.

The original implementation rejects 16-token MoE prefill and Qwen dense four-head groups. Those newly supported cases receive correctness/smoke qualification, not a speed comparison to incomplete or rejected execution. The smaller standalone audit timings include host submission overhead and are supporting evidence only.

This qualification covers this device and these workloads. It does not establish performance on other devices, all shapes, or all model families.
