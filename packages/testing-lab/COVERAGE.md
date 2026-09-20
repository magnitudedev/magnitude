# Testing lab coverage

Updated 2026-09-20. A catalog entry is a planned target, not a qualification result.
Full qualification requires real package installation, UI interaction, generation through the
requested backend, all selected harnesses, CLI, recovery, update, uninstall and cleanup.

| Machines and coverage | Implementation and observed result | Remaining work |
|---|---|---|
| Azure Ubuntu 24.04, x64 Intel, CPU | Complete unpublished-source flow exercised locally: 65/66 passed. Same full flow in GitHub Actions: 64/66 passed. All nine suites ran; both runs cleaned up successfully. | Hermes file-edit correctness fails. GitHub Pi terminal readiness incorrectly required uppercase output; fix under verification. Neither report is green. |
| Azure Debian 13 / Fedora 44, x64 Intel, CPU | Existing DEB/RPM graph exercised: Debian 65 passed, one CLI check blocked by missing `lsof`. Fedora VM creation rejected because its SCSI image was paired with an NVMe-only VM. Both allocations released. | Deploy the added `lsof` dependency and compatible Fedora v5 VM sizes, then qualify. This run reuses artifacts and does not establish another source build. |
| Azure Red Hat 10, x64 Intel, CPU | Native headless Wayland desktop and failure cleanup verified on Red Hat 10.2. VM removed. | Deploy prepared worker recipe after active runs drain, then qualify packaged app and generation. |
| Azure Ubuntu / Debian / Fedora / Red Hat, x64 AMD and arm64 CPU | Native architecture-specific images and tooling pinned; next coordinator configuration prepared. | Deploy and exercise each target. Image availability is not app qualification. |
| Azure Windows 10 / 11, x64 CPU | Interactive desktop launcher, native process/login removal, pinned tooling installation and compilation of the real installer helper exercised on Windows Server diagnostics. This does not qualify client app behavior. | Integrate runtime/Rust/Hermes preparation and one-shot user login with allocation, confirm client licensing eligibility, then run actual Windows 10/11. |
| Azure Linux / Windows, x64 NVIDIA A10 / RTX PRO 6000, CUDA | Backend/device attestation and generation cases implemented. Correct-subscription GPU quota requests require support; case 2609200010000010. | GPU capacity, driver/toolchain preparation and native execution. CPU fallback cannot pass CUDA. |
| Namespace macOS 15 / 26, arm64 CPU / Metal | Native macOS probes and provider adapter exist; Mac removal checks implemented. Namespace Metal startup calibration fails despite device discovery. | Integrate coordinator allocation/runtime; qualify CPU independently; resolve actual Metal startup failure. Production signing/update trust remains unqualified. |
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
