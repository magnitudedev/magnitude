# Magnitude testing lab

Test unpublished local changes and GitHub pull requests through the same API. The lab snapshots
source, builds and packages it on a disposable producer, then installs and exercises those exact
packages on clean test machines. Results include native diagnostics, Playwright traces, JSON and
JUnit reports; owned workers are removed after success, failure or cancellation.

- [Run tests locally or in CI](USAGE.md): authentication, commands, nine suites, reports and cancellation.
- [Current coverage](COVERAGE.md): actual passes, product failures and external blockers for each platform.
- [Provider setup](infra/README.md): coordinator and worker infrastructure.
- [System contract](../../design/testing-lab.md): ownership, isolation, source integrity and acceptance guarantees.
- [Historical implementation notes](IMPLEMENTATION-NOTES.md): retained checkpoints, not current status.

The supported execution mode is clean `verify`. Coverage selections choose targets and cases;
they do not change the test engine. A listed target is not proof that its provider or backend has
passed qualification. Missing, blocked or failed cases cannot produce a passing report.

The focus is application behavior: native packaging and installation, resilient UI interaction,
real endpoint and Pi/OpenCode/Hermes generation, bundled CLI, recovery, updates and uninstall.
Performance benchmarks and new Linux packaging formats are outside this implementation.
