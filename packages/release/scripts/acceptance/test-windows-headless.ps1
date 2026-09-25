$ErrorActionPreference = 'Stop'
$certificate = $null
try {
  if (!$env:MAGNITUDE_HEADLESS_ACCEPTANCE_OUTPUT) {
    $env:MAGNITUDE_HEADLESS_ACCEPTANCE_OUTPUT = Join-Path $env:USERPROFILE ('magnitude-test-' + [Guid]::NewGuid().ToString('N').Substring(0, 8))
  }
  New-Item -ItemType Directory -Force $env:MAGNITUDE_HEADLESS_ACCEPTANCE_OUTPUT | Out-Null
  $certificate = New-SelfSignedCertificate -Type CodeSigningCert -Subject 'CN=Magnitude Update Acceptance, O=Magnitude Update Acceptance' `
    -CertStoreLocation Cert:\CurrentUser\My -KeyAlgorithm RSA -KeyLength 2048 -HashAlgorithm SHA256 -NotAfter (Get-Date).AddDays(2)
  $publicCertificate = Join-Path $env:MAGNITUDE_HEADLESS_ACCEPTANCE_OUTPUT 'publisher.cer'
  Export-Certificate -Cert $certificate -FilePath $publicCertificate | Out-Null
  Import-Certificate -FilePath $publicCertificate -CertStoreLocation Cert:\LocalMachine\Root | Out-Null
  $env:MAGNITUDE_ACCEPTANCE_WINDOWS_CERTIFICATE = $certificate.Thumbprint
  $env:MAGNITUDE_ACCEPTANCE_NSIS = Join-Path ${env:ProgramFiles(x86)} 'NSIS\makensis.exe'
  if (!(Test-Path -LiteralPath $env:MAGNITUDE_ACCEPTANCE_NSIS)) { throw 'NSIS must be installed before packaged update acceptance.' }
  & bun (Join-Path $PSScriptRoot 'test-windows-headless-installer.ts')
  if ($LASTEXITCODE -ne 0) { throw "Windows packaged update acceptance failed with exit $LASTEXITCODE" }
  $receipt = Join-Path $env:MAGNITUDE_HEADLESS_ACCEPTANCE_OUTPUT 'result.json'
  if (!(Test-Path -LiteralPath $receipt) -or (Get-Item -LiteralPath $receipt).Length -eq 0) { throw 'Packaged update acceptance did not write its completion receipt.' }
} finally {
  Remove-Item Env:MAGNITUDE_ACCEPTANCE_WINDOWS_CERTIFICATE -ErrorAction SilentlyContinue
  if ($certificate) {
    Remove-Item -LiteralPath "Cert:\LocalMachine\Root\$($certificate.Thumbprint)" -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath "Cert:\CurrentUser\My\$($certificate.Thumbprint)" -ErrorAction SilentlyContinue
  }
}
