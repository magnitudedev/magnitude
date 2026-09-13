$ErrorActionPreference = 'Stop'
$root = Join-Path $env:RUNNER_TEMP 'magnitude-windows-update-acceptance'
New-Item -ItemType Directory -Force $root | Out-Null
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
  foreach ($version in @('0.0.30','0.0.31')) {
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
