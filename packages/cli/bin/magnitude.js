#!/usr/bin/env node
':' //; if command -v node >/dev/null 2>&1; then exec node "$0" "$@"; fi
':' //; if command -v bun >/dev/null 2>&1; then exec bun "$0" "$@"; fi
':' //; echo "Magnitude requires Node.js or Bun to start." >&2; exit 127

// Releases only ship macOS and Linux artifacts for x64/arm64. Keep this list in
// sync with lib/download.js and releaseHosts in packages/release/src/targets.ts.
// MAGNITUDE_ALLOW_UNSUPPORTED_HOST=1 skips the check, e.g. to test prerelease
// artifacts served from MAGNITUDE_RELEASE_BASE_URL.
const supportedHost =
  Boolean(process.env.MAGNITUDE_ALLOW_UNSUPPORTED_HOST) ||
  (['darwin', 'linux'].includes(process.platform) &&
    ['x64', 'arm64'].includes(process.arch));

async function main() {
  try {
    const { ensureBinary } = require('../lib/download.js');
    const version = require('../package.json').version;

    const binaryPath = await ensureBinary(version);

    // Spawn the binary with inherited stdio
    const result = require('child_process').spawnSync(binaryPath, process.argv.slice(2), {
      stdio: 'inherit',
      env: process.env,
    });

    if (result.error) throw result.error;

    if (result.signal) {
      // Re-raise so callers see the same signal that killed the CLI binary;
      // the exit code is a fallback in case the signal does not terminate us.
      process.kill(process.pid, result.signal);
      process.exitCode = 128 + (require('os').constants.signals[result.signal] ?? 0);
      return;
    }

    process.exitCode = result.status ?? 1;
  } catch (err) {
    console.error('Failed to launch Magnitude:', err.message);
    process.exitCode = 1;
  }
}

if (supportedHost) {
  main();
} else if (process.platform === 'win32') {
  console.error('Magnitude supports macOS and Linux (x64/arm64). On Windows, run Magnitude inside WSL:');
  console.error('  https://learn.microsoft.com/windows/wsl/install');
  console.error('Already inside WSL? Then this Node.js is the Windows one; install Node.js or Bun inside your distro and try again.');
  process.exitCode = 1;
} else {
  console.error(`Magnitude supports macOS and Linux (x64/arm64); ${process.platform}-${process.arch} is not supported.`);
  process.exitCode = 1;
}
