# Remote benchmarking

- Edit locally and sync `inference-v2/` to an idle benchmark machine with `rsync`, excluding environments, caches, models, and results.
- Use configured SSH aliases to install dependencies with `uv sync --frozen` and run existing benchmark commands. Cache model weights on each machine.
- Run one measurement at a time per machine; parallelize across machines. Do not sync while a run is active. Compare baseline and candidate on the same machine.
- Copy results and logs back locally, preserving which source was measured.
