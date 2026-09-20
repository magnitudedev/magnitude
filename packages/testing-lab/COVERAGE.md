# Testing lab coverage

Updated 2026-09-20. A catalog entry is a planned target, not a qualification result.
Full qualification requires real package installation, UI interaction, generation through the
requested backend, all selected harnesses, CLI, recovery, update, uninstall and cleanup.

| Machines and coverage | Implementation and observed result | Remaining work |
|---|---|---|
| Azure Ubuntu 24.04, x64 Intel, CPU | Complete unpublished-source flow exercised locally: 65/66 passed. Same full flow in GitHub Actions: 64/66 passed. All nine suites ran; both runs cleaned up successfully. | Hermes file-edit correctness fails. GitHub Pi terminal readiness incorrectly required uppercase output; the correction passed in subsequent Fedora qualification. Neither report is green. |
| Azure Fedora 44, x64 Intel / AMD, CPU | All 66 checks passed on both machines, including all nine suites and Pi/OpenCode/Hermes. Workers removed; evidence verified. | This qualifies the existing RPM graph; compilation was performed by the canonical Ubuntu producer. |
| Azure Debian 13, x64 Intel / AMD, and Ubuntu 24.04 x64 AMD, CPU | Debian Intel 65/66 passed; Debian AMD and Ubuntu AMD 64/66 passed. No blocked checks or cleanup errors. | Remaining failures concern Hermes tool editing and terminal generation/interruption. Reports are not green. |
| Azure Red Hat 10, x64 Intel / AMD, CPU | Intel passed all 66 checks. AMD passed 64/66 with no blocked checks. Both completed all nine suites and cleanup. The 1 GiB home-volume failure was fixed and natively verified. | AMD Hermes H5 reported unresolved tools; H7 did not render streamed output. All 296 evidence objects (185,693,726 bytes) were retrieved and verified after cleanup; Azure inventory contains no resources for this run. |
| Azure Ubuntu / Debian / Fedora / Red Hat, arm64 CPU | The corrected Clang source producer built and packaged successfully. All four native consumers are currently running. | Collect consumer results, verify retained evidence and audit cleanup; running is not passing. |
| Azure Windows 10 / 11, x64 CPU | Interactive desktop launcher, native process/login removal, pinned tooling installation, real installer-helper compilation, runtime dependencies, Hermes, native terminal startup and one-shot interactive desktop readiness exercised on Windows Server diagnostics. Runtime, Hermes, Rust, terminal subprocess and workspace writes also passed as the admitted desktop user (session 1). This does not qualify client app behavior. | Automatic preparation is connected to the allocator; a fresh Server diagnostic completed all stages and reconciled with one reboot, then was deleted. Readiness refresh uses a new command identity to exclude stale Azure observations. The signed-in user and tenant both returned empty Microsoft Graph license inventories; this does not establish any separate Visual Studio entitlement. Confirm client licensing eligibility, then run actual Windows 10/11. |
| Azure Linux / Windows, x64 NVIDIA A10 / RTX PRO 6000, CUDA | Backend/device attestation and generation cases implemented. Correct-subscription GPU quota requests require support; case 2609200010000010. | Pinned driver preparation and post-install device/version checks are implemented for Ubuntu 24.04 and Windows 11 (plus Server diagnostics). Actual GPU installation and generation remain unverified pending capacity. The current Azure driver documentation does not establish support for Debian 13, Fedora 44, Red Hat 10 or Windows 10; those combinations are rejected before allocation. CUDA SDK preparation is implemented for Linux x64/ARM64 and Windows x64; Linux x64 natively compiled PTX and linked a shared library against cuBLAS on an Azure CPU VM. ARM/Windows SDK execution and full CUDA package builds are not yet qualified. CPU fallback cannot pass CUDA. |
| Namespace macOS 15 / 26, arm64 CPU / Metal | Native macOS probes and provider adapter exist; Mac removal checks implemented. Namespace Metal startup calibration fails despite device discovery. | Automatic runtime preparation and reconciliation passed on fresh macOS 15 and 26 machines. macOS 26 uses the image’s Python to bootstrap pinned CPython 3.12.13 for Hermes. The exact Linux coordinator image authenticated to Namespace in an Azure job, and that diagnostic job was deleted. Deploy the coordinator integration and qualify the packaged CPU app independently; resolve actual Metal startup failure. Production signing/update trust remains unqualified. |
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

Namespace coordinator credential proof: Azure job execution `ml-namespace-auth-proof-mygx5sp` succeeded using image `sha256:14ee56dc39edd3ccb9065513dc6cf43d6e878057711412d21e95a2805c03a451`. The first diagnostic failed because its parser omitted Devbox’s empty-inventory notice; the real provider already handles that notice. Successful authentication is not packaged Mac qualification. Both job executions and their parent job were removed; subsequent job inventory was empty.
