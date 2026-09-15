param([Parameter(Mandatory = $true)][string]$Path)
$ErrorActionPreference = 'Stop'

foreach ($name in @('MAGNITUDE_WINDOWS_SIGNTOOL', 'MAGNITUDE_WINDOWS_SIGNING_DLIB', 'MAGNITUDE_WINDOWS_SIGNING_METADATA', 'MAGNITUDE_WINDOWS_PUBLISHER')) {
  if ([string]::IsNullOrWhiteSpace([Environment]::GetEnvironmentVariable($name))) { throw "Missing signing configuration: $name" }
}
if ($env:MAGNITUDE_WINDOWS_DISTRIBUTION -ne 'artifact-signing') { throw 'Production signing must be explicitly enabled.' }
foreach ($file in @($Path, $env:MAGNITUDE_WINDOWS_SIGNTOOL, $env:MAGNITUDE_WINDOWS_SIGNING_DLIB, $env:MAGNITUDE_WINDOWS_SIGNING_METADATA)) {
  if (!(Test-Path -LiteralPath $file -PathType Leaf)) { throw "Missing signing input: $file" }
}

# Keep the desktop's bundled CLI and service byte-identical to their signed archives.
$existing = Get-AuthenticodeSignature -LiteralPath $Path
if ($existing.Status -eq 'Valid' -and $existing.TimeStamperCertificate -and
    $existing.SignerCertificate.GetNameInfo([Security.Cryptography.X509Certificates.X509NameType]::SimpleName, $false) -ceq $env:MAGNITUDE_WINDOWS_PUBLISHER) {
  & $env:MAGNITUDE_WINDOWS_SIGNTOOL verify /pa /all /tw $Path
  if ($LASTEXITCODE -ne 0) { throw "Existing signature verification failed for $Path." }
  exit 0
}

& $env:MAGNITUDE_WINDOWS_SIGNTOOL sign /fd SHA256 /tr http://timestamp.acs.microsoft.com /td SHA256 /dlib $env:MAGNITUDE_WINDOWS_SIGNING_DLIB /dmdf $env:MAGNITUDE_WINDOWS_SIGNING_METADATA $Path
if ($LASTEXITCODE -ne 0) { throw "Artifact Signing failed for $Path (exit $LASTEXITCODE)." }
& $env:MAGNITUDE_WINDOWS_SIGNTOOL verify /pa /all /tw $Path
if ($LASTEXITCODE -ne 0) { throw "Signature verification failed for $Path (exit $LASTEXITCODE)." }
$signature = Get-AuthenticodeSignature -LiteralPath $Path
if ($signature.Status -ne 'Valid' -or !$signature.TimeStamperCertificate -or
    $signature.SignerCertificate.GetNameInfo([Security.Cryptography.X509Certificates.X509NameType]::SimpleName, $false) -cne $env:MAGNITUDE_WINDOWS_PUBLISHER) {
  throw "Signature publisher or timestamp did not match for $Path."
}
