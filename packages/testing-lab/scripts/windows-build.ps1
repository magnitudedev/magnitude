param(
  [Parameter(Mandatory=$true)][string]$BunExecutable,
  [Parameter(Mandatory=$true)][string]$ArgumentsBase64
)
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$root = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..\..'))
# Initialize from the admitted source's release helper for every phase, including native
# dependency installation. The coordinator never supplies an opaque compiler environment.
. (Join-Path $root 'packages\release\scripts\build\windows-engine-toolchain.ps1')
$arguments = @([Text.Encoding]::UTF8.GetString([Convert]::FromBase64String($ArgumentsBase64)) | ConvertFrom-Json)
if (-not $arguments.Count -or @($arguments | Where-Object { $_ -isnot [string] }).Count) { throw 'Invalid build argument vector' }
$directory = Join-Path ([IO.Path]::GetTempPath()) ('Magnitude lab native input ' + [Guid]::NewGuid())
New-Item -ItemType Directory -Path $directory | Out-Null
$previousLibrary = $env:MAGNITUDE_NODE_LIBRARY
$exitCode = 1
try {
  $nodeVersion = (& node.exe -p process.versions.node).Trim()
  if ($LASTEXITCODE -ne 0 -or $nodeVersion -notmatch '^\d+\.\d+\.\d+$') { throw 'Cannot identify native Node build input' }
  $base = "https://nodejs.org/download/release/v$nodeVersion"
  $library = Join-Path $directory 'node.lib'
  Invoke-WebRequest "$base/win-x64/node.lib" -OutFile $library -TimeoutSec 120
  $checksums = (Invoke-WebRequest "$base/SHASUMS256.txt" -TimeoutSec 120).Content
  $checksum = [regex]::Match($checksums, '(?m)^([a-f0-9]{64})\s+win-x64/node\.lib\r?$')
  if (-not $checksum.Success -or (Get-FileHash -LiteralPath $library -Algorithm SHA256).Hash.ToLowerInvariant() -ne $checksum.Groups[1].Value) { throw 'Node import library integrity check failed' }
  $env:MAGNITUDE_NODE_LIBRARY = $library
  & $BunExecutable @arguments
  $exitCode = $LASTEXITCODE
} finally {
  $env:MAGNITUDE_NODE_LIBRARY = $previousLibrary
  Remove-Item -LiteralPath $directory -Recurse -Force
}
exit $exitCode
