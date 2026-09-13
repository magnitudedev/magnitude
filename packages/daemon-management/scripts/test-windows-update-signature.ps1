$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'windows-toolchain.ps1')
$source = Join-Path (Split-Path -Parent $PSScriptRoot) 'native'
$fixture = Join-Path ([IO.Path]::GetTempPath()) ('Magnitude update signatures ' + [Guid]::NewGuid())
New-Item -ItemType Directory $fixture | Out-Null
$certificate = $null
Push-Location $fixture
try {
  & cl.exe /nologo /W4 /WX /O2 /std:c11 /D_WIN32_WINNT=0x0A00 /D_CRT_SECURE_NO_WARNINGS `
    (Join-Path $source 'windows-update-signature.c') (Join-Path $source 'windows-update-signature-test.c') `
    /Fe:verify.exe /link wintrust.lib crypt32.lib
  if ($LASTEXITCODE -ne 0) { throw 'Native signature verifier compilation failed' }
  $binary = Join-Path $fixture 'verify.exe'
  & $binary $binary 'Magnitude Update Acceptance'
  if ($LASTEXITCODE -eq 0) { throw 'Unsigned installer was accepted' }
  Write-Output 'Creating isolated fixture publisher'
  $certificate = New-SelfSignedCertificate -Type CodeSigningCert -Subject 'CN=Magnitude Update Acceptance, O=Magnitude Update Acceptance' `
    -CertStoreLocation Cert:\CurrentUser\My -KeyAlgorithm RSA -KeyLength 2048 -HashAlgorithm SHA256 -NotAfter (Get-Date).AddDays(1)
  Export-Certificate -Cert $certificate -FilePath (Join-Path $fixture 'publisher.cer') | Out-Null
  # CurrentUser Root displays an interactive trust prompt, even on a headless runner.
  # The disposable CI machine is elevated; remove this exact test root in finally.
  Write-Output 'Trusting isolated fixture publisher on the disposable runner'
  Import-Certificate -FilePath (Join-Path $fixture 'publisher.cer') -CertStoreLocation Cert:\LocalMachine\Root | Out-Null
  Write-Output 'Signing the fixture executable'
  $signed = Set-AuthenticodeSignature -FilePath $binary -Certificate $certificate -HashAlgorithm SHA256
  if ($signed.Status -ne 'Valid') { throw "Fixture signing failed: $($signed.Status)" }
  Write-Output 'Verifying the trusted fixture publisher'
  & $binary $binary 'Magnitude Update Acceptance'
  if ($LASTEXITCODE -ne 0) { throw 'Trusted fixture publisher was rejected' }
  & $binary $binary 'Different Publisher'
  if ($LASTEXITCODE -eq 0) { throw 'Wrong publisher was accepted' }
  $tampered = Join-Path $fixture 'tampered.exe'
  $bytes = [IO.File]::ReadAllBytes($binary)
  $bytes[4096] = $bytes[4096] -bxor 1
  [IO.File]::WriteAllBytes($tampered, $bytes)
  & $binary $tampered 'Magnitude Update Acceptance'
  if ($LASTEXITCODE -eq 0) { throw 'Tampered installer was accepted' }
  Write-Output 'PASS unsigned, trusted, wrong-publisher and tampered native installer checks'
  $global:LASTEXITCODE = 0
} finally {
  if ($certificate) {
    Remove-Item -LiteralPath "Cert:\LocalMachine\Root\$($certificate.Thumbprint)" -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath "Cert:\CurrentUser\My\$($certificate.Thumbprint)" -ErrorAction SilentlyContinue
  }
  Pop-Location
  Remove-Item -LiteralPath $fixture -Recurse -Force
}
