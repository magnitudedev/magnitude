---
"@magnitudedev/cli": patch
---

fail fast with clear guidance on unsupported hosts instead of installing a broken launcher

Host support is now enforced at run time by a single policy in lib/download.js, shared by the launcher and by direct consumers of ensureBinary: macOS and Linux on x64/arm64, glibc only (musl hosts such as Alpine are refused up front instead of failing the download probe later). Native Windows users get WSL guidance, including the case where WSL PATH interop runs the Windows Node inside WSL. Setting MAGNITUDE_ALLOW_UNSUPPORTED_HOST (any value except "0"/"false") skips the check, e.g. to exercise windows-x64-msvc prerelease artifacts served from MAGNITUDE_RELEASE_BASE_URL on native Windows x64; the launcher mentions this whenever MAGNITUDE_RELEASE_BASE_URL is set.

The npm "os" field is intentionally not used: it hard-fails the entire `npm install` of any project that depends on this package on Windows, before any guidance can print. Known limitation: the wrappers npm generates on native Windows bake in the launcher's `#!/bin/sh` interpreter, so `magnitude` from cmd or PowerShell still fails with a raw "/bin/sh is not recognized" error before the guard can run. The sh shebang cannot change, because the launcher must work on node-only and bun-only machines (enforced by accept-public-cli.ts); the guard does run for node, bun, bunx, Git Bash, and WSL-interop invocations. Launch failures are reported instead of exiting silently, and a child killed by a signal exits with 128 plus the signal number, like a shell, instead of a flat 1.
