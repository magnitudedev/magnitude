param(
  [Parameter(Mandatory = $true)][string]$CatalogDirectory,
  [Parameter(Mandatory = $true)][string]$OutputDirectory
)
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

. (Join-Path $PSScriptRoot 'windows-engine-toolchain.ps1')
$env:PATH = "${env:ProgramFiles(x86)}\NSIS;$env:PATH"
foreach ($tool in @('cmake.exe', 'ninja.exe', 'makensis.exe', 'pwsh.exe', 'node.exe', 'bun.exe', 'cargo.exe')) {
  Get-Command $tool -ErrorAction Stop | Out-Null
}

# Node-API headers are a workspace dependency; the matching Windows import library
# is a build input, never an installed application dependency.
$inputDirectory = Join-Path ([IO.Path]::GetTempPath()) ('magnitude-windows-build-' + [guid]::NewGuid())
New-Item -ItemType Directory $inputDirectory | Out-Null
$previousLibrary = $env:MAGNITUDE_NODE_LIBRARY
try {
  if ($env:MAGNITUDE_WINDOWS_DISTRIBUTION -eq 'artifact-signing') {
    . (Join-Path $PSScriptRoot 'windows-signing-setup.ps1') -Directory $inputDirectory
  }
  $nodeVersion = (& node.exe -p process.versions.node).Trim()
  if ($LASTEXITCODE -ne 0) { throw 'Could not determine the Node build version.' }
  $nodeBase = "https://nodejs.org/download/release/v$nodeVersion"
  $library = Join-Path $inputDirectory 'node.lib'
  Invoke-WebRequest -UseBasicParsing "$nodeBase/win-x64/node.lib" -OutFile $library
  $checksums = (Invoke-WebRequest -UseBasicParsing "$nodeBase/SHASUMS256.txt").Content
  $checksum = [regex]::Match($checksums, '(?m)^([a-f0-9]{64})\s+win-x64/node\.lib\r?$')
  if (!$checksum.Success -or (Get-FileHash $library -Algorithm SHA256).Hash.ToLowerInvariant() -ne $checksum.Groups[1].Value) {
    throw 'Node import library integrity check failed.'
  }
  $env:MAGNITUDE_NODE_LIBRARY = $library
  & bun.exe (Join-Path $PSScriptRoot 'host.ts') windows-x64-msvc $CatalogDirectory $OutputDirectory
  if ($LASTEXITCODE -ne 0) { throw "Windows host build failed with exit $LASTEXITCODE." }
} finally {
  $env:MAGNITUDE_NODE_LIBRARY = $previousLibrary
  Remove-Item -LiteralPath $inputDirectory -Recurse -Force
}
