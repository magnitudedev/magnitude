# Distribution

**The engine is an embeddable Rust library with CLI and server compositions.**
Installed execution uses packaged programs and native code plus declared OS and
driver facilities.

## Products and dependencies

| Component | Contents and boundary |
| --- | --- |
| Engine library | Loading, model execution, state, and optional service composition; independent of HTTP |
| Seismic | Compiler/runtime and selected backend implementations |
| Program libraries | Embedded checked standard and model programs, with generated host bindings |
| CLI / server | Application composition over the same engine and chat facilities |
| Native dependencies | Owned template/parser extraction and required host libraries, with version identity and licenses |
| Development tools | Source overrides, independent references, benchmarks, and build tooling; not customer runtime requirements |

- Backend features permit CPU-only builds without loading GPU drivers.
- Build hosts can check and embed programs without possessing each target accelerator.
- Device-specific compilation occurs through the selected backend's supported route.
- Packaged execution requires no Python preparer, source checkout, CMake, external
  compiler/linker, vendor headers, or CUDA toolkit.
- Owned native libraries resolve relative to the installation; ambient development
  paths and loader overrides are not part of the runtime contract.
- Local execution requires no network after installation of engine and model artifacts.

## Platform architecture

| Host family | Base execution | Accelerator composition |
| --- | --- | --- |
| macOS arm64 | CPU | Metal |
| macOS x64 | CPU | Additional accelerator support requires an explicit platform contract |
| GNU Linux arm64 / x64 | CPU | CUDA and Vulkan where device/driver capabilities apply |
| Windows x64 MSVC | CPU | Accelerator support is an explicitly declared build/runtime combination |

Artifacts declare OS/ABI floors, CPU ISA requirements, backend features, and driver
and target-code compatibility. Capability detection prevents unsupported instruction
use. Supported architecture does not imply that every device or artifact combination
has been qualified. Release matrices and measurements belong in release records.
Per-backend floors and tiers are defined in [compatibility](compatibility.md).

## Identity and reuse

- Program, compiler, native dependency, and backend identities are reproducible.
- Model artifacts remain separately identified; equal geometry permits code reuse
  without conflating weight resources.
- Compiled-cache compatibility follows [runtime identities](seismic/runtime.md),
  rather than invalidating every kernel for unrelated engine edits.
- Corrupt or incompatible caches are recoverable misses.
- Development source overrides pass the same program checks as embedded libraries.
