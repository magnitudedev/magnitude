param([Parameter(Mandatory=$true)][string]$Path, [Parameter(Mandatory=$true)][string]$Thumbprint, [string]$TimestampServer)
$ErrorActionPreference = 'Stop'
if ($Thumbprint -notmatch '^[A-Fa-f0-9]{40}$') { throw 'Invalid acceptance certificate fingerprint' }
$certificate = Get-Item -LiteralPath "Cert:\CurrentUser\My\$Thumbprint"
if ($certificate.Subject -ne 'CN=Magnitude Update Acceptance, O=Magnitude Update Acceptance') { throw 'This signer is only for the isolated acceptance publisher' }
$files = if (Test-Path -LiteralPath $Path -PathType Container) {
  Get-ChildItem -LiteralPath $Path -File -Recurse | Where-Object { $_.Extension -in '.exe','.dll','.node' }
} else { Get-Item -LiteralPath $Path }
foreach ($file in $files) {
  $parameters = @{ LiteralPath = $file.FullName; Certificate = $certificate; HashAlgorithm = 'SHA256' }
  if ($TimestampServer) { $parameters.TimestampServer = $TimestampServer }
  $result = Set-AuthenticodeSignature @parameters
  if ($result.Status -ne 'Valid') { throw "Acceptance signing failed for $($file.Name): $($result.Status)" }
  if ($TimestampServer -and !$result.TimeStamperCertificate) { throw "Acceptance timestamp missing for $($file.Name)" }
}
Write-Output "Signed $(@($files).Count) acceptance binaries"
