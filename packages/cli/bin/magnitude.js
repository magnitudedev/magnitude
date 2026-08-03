#!/bin/sh
':' //; if command -v node >/dev/null 2>&1; then exec node "$0" "$@"; fi
':' //; if command -v bun >/dev/null 2>&1; then exec bun "$0" "$@"; fi
':' //; echo "Magnitude requires Node.js or Bun to start." >&2; exit 127

const { ensureBinary } = require('../lib/download.js');

const version = require('../package.json').version;

async function main() {
  try {
    const binaryPath = await ensureBinary(version);
    
    // Spawn the binary with inherited stdio
    const result = require('child_process').spawnSync(binaryPath, process.argv.slice(2), {
      stdio: 'inherit',
      env: process.env,
    });
    
    process.exit(result.status ?? 1);
  } catch (err) {
    console.error('Failed to launch Magnitude:', err.message);
    process.exit(1);
  }
}

main();
