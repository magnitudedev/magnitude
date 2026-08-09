#!/usr/bin/env node
':' //; if command -v node >/dev/null 2>&1; then exec node "$0" "$@"; fi
':' //; if command -v bun >/dev/null 2>&1; then exec bun "$0" "$@"; fi
':' //; echo "Magnitude requires Node.js or Bun to start." >&2; exit 127

const { ensureBinary, unsupportedHostReason } = require('../lib/download.js');
const version = require('../package.json').version;

async function main() {
  try {
    const binaryPath = await ensureBinary(version);

    // Spawn the binary with inherited stdio
    const result = require('child_process').spawnSync(binaryPath, process.argv.slice(2), {
      stdio: 'inherit',
      env: process.env,
    });

    if (result.error) throw result.error;

    if (result.signal) {
      // Encode the child's fatal signal the way shells do (128 + number).
      // Deliberately not re-raised: re-raising SIGUSR1 would start Node's
      // inspector instead of terminating the launcher.
      process.exit(128 + (require('os').constants.signals[result.signal] ?? 0));
    }

    process.exit(result.status ?? 1);
  } catch (err) {
    console.error('Failed to launch Magnitude:', err.message);
    process.exit(1);
  }
}

const reason = unsupportedHostReason();
if (reason === null) {
  main();
} else {
  if (process.platform === 'win32') {
    console.error('Magnitude supports macOS and Linux (x64/arm64). On Windows, run Magnitude inside WSL:');
    console.error('  https://learn.microsoft.com/windows/wsl/install');
    console.error('Already inside WSL? Then this Node.js is the Windows one; install Node.js or Bun inside your distro and try again.');
  } else {
    console.error(`Magnitude supports macOS and Linux on x64/arm64 with glibc; ${reason}.`);
  }
  if (process.env.MAGNITUDE_RELEASE_BASE_URL) {
    console.error('Set MAGNITUDE_ALLOW_UNSUPPORTED_HOST=1 to bypass this check and use artifacts from MAGNITUDE_RELEASE_BASE_URL.');
  }
  // exitCode instead of process.exit so the message above always flushes on
  // Windows; nothing else is pending, so the process exits immediately.
  process.exitCode = 1;
}
