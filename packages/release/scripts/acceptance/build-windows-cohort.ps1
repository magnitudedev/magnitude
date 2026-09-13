$ErrorActionPreference = 'Stop'
$root = Join-Path $env:RUNNER_TEMP 'magnitude-windows-update-acceptance'
New-Item -ItemType Directory -Force $root | Out-Null
$nodeVersion = (node -p process.versions.node).Trim()
$nodeBase = "https://nodejs.org/download/release/v$nodeVersion"
$nodeLibrary = Join-Path $root 'node.lib'
Invoke-WebRequest "$nodeBase/win-x64/node.lib" -OutFile $nodeLibrary
$checksums = (Invoke-WebRequest "$nodeBase/SHASUMS256.txt").Content
$checksum = [regex]::Match($checksums, '(?m)^([a-f0-9]{64})\s+win-x64/node\.lib\r?$')
if (!$checksum.Success -or (Get-FileHash $nodeLibrary -Algorithm SHA256).Hash.ToLowerInvariant() -ne $checksum.Groups[1].Value) {
  throw 'Node import library integrity check failed'
}
$env:MAGNITUDE_NODE_LIBRARY = $nodeLibrary
$certificate = $null
try {
  $certificate = New-SelfSignedCertificate -Type CodeSigningCert -Subject 'CN=Magnitude Update Acceptance, O=Magnitude Update Acceptance' `
    -CertStoreLocation Cert:\CurrentUser\My -KeyAlgorithm RSA -KeyLength 2048 -HashAlgorithm SHA256 -NotAfter (Get-Date).AddDays(2)
  $publicCertificate = Join-Path $root 'acceptance-publisher.cer'
  Export-Certificate -Cert $certificate -FilePath $publicCertificate | Out-Null
  Import-Certificate -FilePath $publicCertificate -CertStoreLocation Cert:\LocalMachine\Root | Out-Null
  $env:MAGNITUDE_ACCEPTANCE_WINDOWS_CERTIFICATE = $certificate.Thumbprint
  $env:MAGNITUDE_ACCEPTANCE_NSIS = Join-Path ${env:ProgramFiles(x86)} 'NSIS\makensis.exe'
  if (!(Test-Path -LiteralPath $env:MAGNITUDE_ACCEPTANCE_NSIS)) {
    & choco install nsis --yes --no-progress
    if ($LASTEXITCODE -ne 0) { throw 'NSIS acquisition failed' }
  }
  foreach ($version in @('0.0.32','0.0.33')) {
    $env:MAGNITUDE_ACCEPTANCE_VERSION = $version
    $env:MAGNITUDE_ACCEPTANCE_OUTPUT = Join-Path $root $version
    & bun (Join-Path $PSScriptRoot 'build-desktop.ts')
    if ($LASTEXITCODE -ne 0) { throw "Windows application build failed for $version" }
    $env:MAGNITUDE_ACCEPTANCE_ARTIFACTS = Join-Path $env:MAGNITUDE_ACCEPTANCE_OUTPUT 'artifacts'
    & bun (Join-Path $PSScriptRoot 'prepare-artifacts.ts')
    if ($LASTEXITCODE -ne 0) { throw "Windows artifact transfer verification failed for $version" }
  }
} finally {
  Remove-Item Env:MAGNITUDE_ACCEPTANCE_WINDOWS_CERTIFICATE -ErrorAction SilentlyContinue
  if ($certificate) {
    Remove-Item -LiteralPath "Cert:\LocalMachine\Root\$($certificate.Thumbprint)" -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath "Cert:\CurrentUser\My\$($certificate.Thumbprint)" -ErrorAction SilentlyContinue
  }
}
