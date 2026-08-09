"use strict";

const runtime = require("./release-runtime.cjs");

// Canonical launcher-side host policy, shared with bin/magnitude.js. The
// artifacts themselves are defined by releaseHosts in
// packages/release/src/targets.ts; keep the two in sync.
// MAGNITUDE_ALLOW_UNSUPPORTED_HOST (any value except "0"/"false") skips the
// check, e.g. to exercise windows-x64-msvc prerelease artifacts served from
// MAGNITUDE_RELEASE_BASE_URL on native Windows x64.

const bypassRequested = () => {
  const value = (process.env.MAGNITUDE_ALLOW_UNSUPPORTED_HOST || "").toLowerCase();
  return value !== "" && value !== "0" && value !== "false";
};

// Returns null when this host can run published release artifacts, otherwise a
// short reason ("win32-x64 is not supported").
const unsupportedHostReason = () => {
  if (bypassRequested()) return null;
  const host = `${process.platform}-${process.arch}`;
  if (
    !["darwin", "linux"].includes(process.platform) ||
    !["x64", "arm64"].includes(process.arch)
  ) {
    return `${host} is not supported`;
  }
  if (
    process.platform === "linux" &&
    !process.versions.bun &&
    typeof process.report?.getReport === "function" &&
    process.report.getReport().header?.glibcVersionRuntime === undefined
  ) {
    return `${host} uses musl libc, and Magnitude's Linux builds require glibc`;
  }
  return null;
};

// The release runtime's only export is ensureBinary; wrap it so direct
// consumers get the same host policy as the launcher.
const ensureBinary = (version) => {
  const reason = unsupportedHostReason();
  if (reason) {
    let message = `Magnitude releases support macOS and Linux on x64/arm64 with glibc; ${reason}.`;
    if (process.platform === "win32") {
      message += " On Windows, run Magnitude inside WSL: https://learn.microsoft.com/windows/wsl/install";
    }
    if (process.env.MAGNITUDE_RELEASE_BASE_URL) {
      message += " Set MAGNITUDE_ALLOW_UNSUPPORTED_HOST=1 to bypass this check.";
    }
    return Promise.reject(new Error(message));
  }
  return runtime.ensureBinary(version);
};

module.exports = { ensureBinary, unsupportedHostReason };
