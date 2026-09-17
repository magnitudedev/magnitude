param(
  [Parameter(Mandatory = $true)][string]$PackId,
  [Parameter(Mandatory = $true)][string]$OutputDirectory
)
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

. (Join-Path $PSScriptRoot 'windows-engine-toolchain.ps1')
foreach ($tool in @('cl.exe', 'dumpbin.exe', 'cmake.exe', 'ninja.exe', 'bun.exe', 'cargo.exe')) {
  Get-Command $tool -ErrorAction Stop | Out-Null
}

$inputDirectory = Join-Path ([IO.Path]::GetTempPath()) ('magnitude-backend-signing-' + [guid]::NewGuid())
New-Item -ItemType Directory $inputDirectory | Out-Null
try {
  if ($env:MAGNITUDE_WINDOWS_DISTRIBUTION -eq 'artifact-signing') {
    . (Join-Path $PSScriptRoot 'windows-signing-setup.ps1') -Directory $inputDirectory
  }
  & bun.exe (Join-Path $PSScriptRoot 'backend.ts') $PackId $OutputDirectory
  if ($LASTEXITCODE -ne 0) { throw "Windows backend build failed with exit $LASTEXITCODE." }
} finally {
  Remove-Item -LiteralPath $inputDirectory -Recurse -Force
}
