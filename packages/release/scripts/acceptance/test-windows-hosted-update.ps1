$ErrorActionPreference = 'Stop'
$root = Join-Path $env:RUNNER_TEMP 'magnitude-windows-update-acceptance'
$certificate = Import-Certificate -FilePath (Join-Path $root 'acceptance-publisher.cer') -CertStoreLocation Cert:\LocalMachine\Root
try {
  $env:MAGNITUDE_WINDOWS_ACCEPTANCE_ROOT = $root
  & node (Join-Path $PSScriptRoot '..\..\..\..\desktop\src\fixtures\windows-hosted-update.mjs')
  if ($LASTEXITCODE -ne 0) { throw 'Installed Windows hosted-update acceptance failed' }
} finally {
  Remove-Item Env:MAGNITUDE_WINDOWS_ACCEPTANCE_ROOT -ErrorAction SilentlyContinue
  Remove-Item -LiteralPath "Cert:\LocalMachine\Root\$($certificate.Thumbprint)" -ErrorAction SilentlyContinue
}
