#!/usr/bin/env node

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
