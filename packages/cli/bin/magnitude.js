#!/bin/sh
':' //; if command -v node >/dev/null 2>&1; then exec node "$0" "$@"; fi
':' //; if command -v bun >/dev/null 2>&1; then exec bun "$0" "$@"; fi
':' //; echo "Magnitude requires Node.js or Bun to start." >&2; exit 127

// Releases only ship macOS and Linux artifacts; Windows users run Magnitude through WSL.
if (process.platform === 'win32') {
  console.error('Magnitude supports macOS and Linux. On Windows, run Magnitude inside WSL: https://learn.microsoft.com/windows/wsl/install');
  process.exit(1);
}

const { ensureBinary } = require('../lib/download.js');
const { existsSync, realpathSync } = require('fs');
const path = require('path');

const version = require('../package.json').version;
const magnitudePackageRoot = realpathSync(path.join(__dirname, '..'));

function isPnpmOwnedMagnitudeInstall(nodeModulesDir) {
  if (!existsSync(path.join(nodeModulesDir, '.modules.yaml'))) {
    return false;
  }

  try {
    return realpathSync(path.join(nodeModulesDir, '@magnitudedev', 'cli')) ===
      magnitudePackageRoot;
  } catch {
    return false;
  }
}

function detectPackageManager() {
  const entrypointDir = path.dirname(path.resolve(process.argv[1]));
  for (const startDir of new Set([magnitudePackageRoot, entrypointDir])) {
    const filesystemRoot = path.parse(startDir).root;
    for (
      let currentDir = startDir;
      currentDir !== filesystemRoot;
      currentDir = path.dirname(currentDir)
    ) {
      if (isPnpmOwnedMagnitudeInstall(path.join(currentDir, 'node_modules'))) {
        return 'pnpm';
      }
    }

    if (isPnpmOwnedMagnitudeInstall(path.join(filesystemRoot, 'node_modules'))) {
      return 'pnpm';
    }
  }

  const userAgent = process.env.npm_config_user_agent || '';
  if (/\bbun\//.test(userAgent)) return 'bun';

  const execPath = process.env.npm_execpath || '';
  if (execPath.includes('bun')) return 'bun';

  if (
    __dirname.includes('.bun/install/global') ||
    __dirname.includes('.bun\\install\\global')
  ) return 'bun';

  return userAgent ? 'npm' : null;
}

function managedEnvironment() {
  const packageManager = detectPackageManager();
  const variable = packageManager === 'bun'
    ? 'MAGNITUDE_MANAGED_BY_BUN'
    : packageManager === 'pnpm'
      ? 'MAGNITUDE_MANAGED_BY_PNPM'
      : 'MAGNITUDE_MANAGED_BY_NPM';
  const environment = {
    ...process.env,
    MAGNITUDE_MANAGED_PACKAGE_ROOT: magnitudePackageRoot,
  };
  delete environment.MAGNITUDE_MANAGED_BY_NPM;
  delete environment.MAGNITUDE_MANAGED_BY_BUN;
  delete environment.MAGNITUDE_MANAGED_BY_PNPM;
  environment[variable] = '1';
  return environment;
}

async function main() {
  try {
    const binaryPath = await ensureBinary(version);
    
    // Spawn the binary with inherited stdio
    const result = require('child_process').spawnSync(binaryPath, process.argv.slice(2), {
      stdio: 'inherit',
      env: managedEnvironment(),
    });
    
    process.exit(result.status ?? 1);
  } catch (err) {
    console.error('Failed to launch Magnitude:', err.message);
    process.exit(1);
  }
}

main();
