#!/usr/bin/env node

const { ensureBinaries } = require('../lib/download.js');

const version = require('../package.json').version;

async function main() {
  try {
    const { binaryPath, acnError } = await ensureBinaries(version, {
      prefetchAcn: !process.argv.slice(2).some((arg) => arg === '--version' || arg === '-V' || arg === '--help' || arg === '-h'),
      requireAcn: process.env.MAGNITUDE_REQUIRE_ACN_PREFETCH === '1',
    });
    if (acnError) {
      console.warn(`Warning: failed to prefetch Magnitude ACN: ${acnError.message}`);
    }
    
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
