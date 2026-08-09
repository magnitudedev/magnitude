---
"@magnitudedev/cli": patch
---

fail fast with clear guidance on unsupported hosts instead of installing a broken launcher

The launcher and lib/download.js now enforce a shared host allowlist (macOS/Linux on x64/arm64) at run time: native Windows users get WSL guidance, other unsupported hosts get a clear error, and MAGNITUDE_ALLOW_UNSUPPORTED_HOST=1 bypasses the check for testing prerelease artifacts via MAGNITUDE_RELEASE_BASE_URL. The npm "os" field is intentionally not used because it hard-fails the entire `npm install` of any dependent project on Windows before the guidance can print. The launcher shebang is now `env node` so npm's Windows cmd shim can reach the guard (the sh header with the bun fallback still applies when the script runs under sh), spawn failures are reported instead of exiting silently, and a child killed by a signal re-raises that signal instead of exiting 1.
