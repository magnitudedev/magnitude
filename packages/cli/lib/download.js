"use strict";

const runtime = require("./release-runtime.cjs");

// Releases only ship macOS and Linux artifacts for x64/arm64. Keep this list in
// sync with bin/magnitude.js and releaseHosts in packages/release/src/targets.ts.
// MAGNITUDE_ALLOW_UNSUPPORTED_HOST=1 skips the check, e.g. to test prerelease
// artifacts served from MAGNITUDE_RELEASE_BASE_URL.
const supportedHost = () =>
  Boolean(process.env.MAGNITUDE_ALLOW_UNSUPPORTED_HOST) ||
  (["darwin", "linux"].includes(process.platform) &&
    ["x64", "arm64"].includes(process.arch));

const ensureBinary = (version) => {
  if (!supportedHost()) {
    return Promise.reject(new Error(
      `Magnitude releases support macOS and Linux on x64/arm64; ${process.platform}-${process.arch} is not supported. ` +
      "On Windows, run Magnitude inside WSL: https://learn.microsoft.com/windows/wsl/install"
    ));
  }
  return runtime.ensureBinary(version);
};

module.exports = { ...runtime, ensureBinary };
