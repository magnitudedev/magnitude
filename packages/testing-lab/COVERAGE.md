# Testing lab coverage

Updated 2026-09-20. A catalog entry is a planned target, not a qualification result.
Full qualification requires real package installation, UI interaction, generation through the
requested backend, all selected harnesses, CLI, recovery, update, uninstall and cleanup.

| Machines and coverage | Implementation and observed result | Remaining work |
|---|---|---|
| Azure Ubuntu 24.04, x64 Intel, CPU | Complete unpublished-source flow exercised locally: 65/66 passed. Same full flow in GitHub Actions: 64/66 passed. All nine suites ran; both runs cleaned up successfully. | Hermes file-edit correctness fails. GitHub Pi terminal readiness incorrectly required uppercase output; fix under verification. Neither report is green. |
| Azure Fedora 44, x64 Intel / AMD, CPU | All 66 checks passed on both machines, including all nine suites and Pi/OpenCode/Hermes. Workers removed; evidence verified. | This qualifies the existing RPM graph; compilation was performed by the canonical Ubuntu producer. |
| Azure Debian 13, x64 Intel / AMD, and Ubuntu 24.04 x64 AMD, CPU | Debian Intel 65/66 passed; Debian AMD and Ubuntu AMD 64/66 passed. No blocked checks or cleanup errors. | Remaining failures concern Hermes tool editing and terminal generation/interruption. Reports are not green. |
| Azure Red Hat 10, x64 Intel / AMD, CPU | Native headless Wayland desktop previously verified; the full batch blocked both targets during dependency preparation. Workers removed and failure evidence retained. | Native diagnosis found the image’s 1 GiB home volume; the preparation fix expanded it to 64 GiB on a disposable VM. Deploy and rerun package qualification. No app coverage is claimed from the earlier desktop probe. |
| Azure Ubuntu / Debian / Fedora / Red Hat, arm64 CPU | Native images prepared; source producer failed because default GCC cannot compile the engine’s SME variants. All four consumers were blocked. Worker removed; evidence verified. | Clang selection now matches the release workflow. Deploy the prepared correction and rerun the ARM producer and consumers. |
| Azure Windows 10 / 11, x64 CPU | Interactive desktop launcher, native process/login removal, pinned tooling installation, real installer-helper compilation, runtime dependencies, Hermes, native terminal startup and one-shot interactive desktop readiness exercised on Windows Server diagnostics. Runtime, Hermes, Rust, terminal subprocess and workspace writes also passed as the admitted desktop user (session 1). This does not qualify client app behavior. | Automatic preparation is connected to the allocator; a fresh Server diagnostic completed all stages and reconciled with one reboot, then was deleted. Readiness refresh uses a new command identity to exclude stale Azure observations. Confirm client licensing eligibility, then run actual Windows 10/11. |
| Azure Linux / Windows, x64 NVIDIA A10 / RTX PRO 6000, CUDA | Backend/device attestation and generation cases implemented. Correct-subscription GPU quota requests require support; case 2609200010000010. | GPU capacity, driver preparation and native execution remain. CUDA SDK preparation is implemented for Linux x64/ARM64 and Windows x64; Linux x64 natively compiled PTX and linked a shared library against cuBLAS on an Azure CPU VM. ARM/Windows SDK execution and full CUDA package builds are not yet qualified. CPU fallback cannot pass CUDA. |
| Namespace macOS 15 / 26, arm64 CPU / Metal | Native macOS probes and provider adapter exist; Mac removal checks implemented. Namespace Metal startup calibration fails despite device discovery. | Automatic runtime preparation and reconciliation passed on fresh macOS 15 and 26 machines. macOS 26 uses the image’s Python to bootstrap pinned CPython 3.12.13 for Hermes. Deploy the coordinator integration and qualify the packaged CPU app independently; resolve actual Metal startup failure. Production signing/update trust remains unqualified. |
| Office DGX Spark, Ubuntu arm64 CUDA | Explicit opt-in target and backend checks implemented. | Coordinate limited native qualification without disturbing the shared machine. |

Excluded from this implementation: macOS Intel, Amazon Linux, Vulkan, new SUSE/Arch/Omarchy/Alpine
packaging and performance benchmarks. `iterate` warm reuse and historical-release migration are
unfinished; use clean `verify` mode. Private same-source update testing is not production release
trust or historical migration qualification.

## Retained evidence

- Local complete run `run-57775e74-63a7-44f2-a95f-85b0c444c626`: 65 passed, one Hermes H5 failure,
  zero blocked, no cleanup errors. After worker deletion, all 180 unique evidence objects were
  retrieved and verified against SHA-256 and length (109,529,496 bytes).
- [Complete GitHub run 35505215545](https://github.com/magnitudedev/magnitude/actions/runs/35505215545):
  lab run `run-7218eda4-d223-4c72-ae35-c815141b6ec7`, 64 passed, Hermes H5 and Pi H7 failed,
  zero blocked, no cleanup errors. Downloaded GitHub artifact contains all 180 verified evidence
  objects (121,483,697 bytes), JSON report and JUnit report.
- Both complete runs compiled their admitted unpublished source, built private update pairs,
  released producer machines, installed packages on separate clean consumers and retained failures.
  Their failures demonstrate reporting, not complete product acceptance.
- Debian/Fedora run `run-ae1f7775-1b60-4182-9dc3-813e554e3d46`: Debian 65 passed/one blocked; Fedora 66 blocked at allocation; no cleanup errors. All 173 unique evidence objects were retrieved and verified after cleanup (103,856,757 bytes).
- Earlier smaller Ubuntu run passed locally and in GitHub Actions; a second GitHub attempt verified
  cancellation. These narrower passes do not supersede the complete reports above.

Retrieve report/evidence using the commands in [USAGE.md](USAGE.md). Evidence is scoped to the run
owner. Do not rerun a failed case until a concrete change or a new diagnostic question justifies it.

Linux qualification runs: x64 `run-45350833-4305-4a1a-91fe-d918aed37992` (seven targets, existing package graph: 325 passed, 5 failed, 132 blocked, no cleanup errors); arm64 `run-cb657d3f-dd0d-440b-adbc-837c64e300df` (producer failed; no consumer qualification). All four ARM evidence objects were verified after cleanup (2,180,069 bytes), and Azure inventory contains no resources for that run. Do not treat admission or progress as passing qualification.

The completed x64 batch retained 625 unique evidence objects (505,640,921 bytes); all were retrieved and verified after cleanup. Azure inventory contains no resources for that run.
