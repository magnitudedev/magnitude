$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
Set-StrictMode -Version Latest
# Preparation owns these paths; no candidate-supplied executable enters the launcher.
$launch = Get-Content -LiteralPath 'C:\MagnitudeLab\Runtime\launch.json' -Raw | ConvertFrom-Json
if (-not $env:LAB_WORKER_ROOT -or -not $env:LAB_WORKER_TOKEN -or -not $env:LAB_URL) { throw 'Missing scoped worker assignment' }
# Bootstrap owns logs in the attempt directory; the worker requires its own fresh child.
$env:LAB_WORKER_ROOT = Join-Path $env:LAB_WORKER_ROOT 'worker'
$env:PATH = $launch.path
$env:CARGO_HOME = $launch.cargo
$env:RUSTUP_HOME = $launch.rustup
$env:LAB_TERMINAL_NODE_EXECUTABLE = $launch.node
$env:LAB_PI_EXECUTABLE = $launch.pi
$env:LAB_OPENCODE_EXECUTABLE = $launch.opencode
$env:LAB_HERMES_EXECUTABLE = $launch.hermes
$env:LAB_DEPENDENCIES_EXECUTABLE = $launch.dependencies
Set-Location $launch.workspace
& $launch.bun (Join-Path $launch.workspace 'packages\testing-lab\scripts\verify-update-prerequisites.ts')
if ($LASTEXITCODE -ne 0) { throw 'Native update prerequisites failed before candidate execution' }
& $launch.bun (Join-Path $launch.workspace 'packages\testing-lab\src\outward-worker.ts')
exit $LASTEXITCODE
