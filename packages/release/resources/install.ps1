param([ValidateSet('stable', 'beta', 'alpha')][string]$Channel = 'stable')
$ErrorActionPreference = 'Stop'
$origin = '@MAGNITUDE_INSTALL_ORIGIN@'
$publisher = '@MAGNITUDE_WINDOWS_PUBLISHER@'
if ($env:OS -ne 'Windows_NT' -or -not [Environment]::Is64BitOperatingSystem) { throw 'This installer requires 64-bit Windows.' }
if (-not $origin.StartsWith('https://') -or $publisher.StartsWith('@MAGNITUDE_')) { throw 'The installer has no publisher configuration.' }
Add-Type -AssemblyName System.Net.Http
$client = [Net.Http.HttpClient]::new()
$client.Timeout = [TimeSpan]::FromMinutes(10)
$scratch = Join-Path ([IO.Path]::GetTempPath()) ('magnitude-install-' + [Guid]::NewGuid().ToString('N'))
[IO.Directory]::CreateDirectory($scratch) | Out-Null
function Get-InstallerFile([string]$Url, [string]$Path, [long]$Limit) {
  if (-not $Url.StartsWith('https://')) { throw 'Downloads require HTTPS.' }
  $cancellation = [Threading.CancellationTokenSource]::new(600000)
  $response = $null
  try {
    $response = $client.GetAsync($Url, [Net.Http.HttpCompletionOption]::ResponseHeadersRead, $cancellation.Token).GetAwaiter().GetResult()
    $response.EnsureSuccessStatusCode() | Out-Null
    if ($response.RequestMessage.RequestUri.Scheme -ne 'https') { throw 'Unexpected download redirect.' }
    if ($response.Content.Headers.ContentLength -gt $Limit) { throw 'Download exceeds the installation limit.' }
    $inputStream = $response.Content.ReadAsStreamAsync().GetAwaiter().GetResult()
    $outputStream = $null
    try {
      $outputStream = [IO.File]::Open($Path, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write, [IO.FileShare]::None)
      $buffer = New-Object byte[] 65536
      [long]$total = 0
      while (($count = $inputStream.ReadAsync($buffer, 0, $buffer.Length, $cancellation.Token).GetAwaiter().GetResult()) -gt 0) {
        $total += $count
        if ($total -gt $Limit) { throw 'Download exceeds the installation limit.' }
        $outputStream.Write($buffer, 0, $count)
      }
    } finally { if ($null -ne $outputStream) { $outputStream.Dispose() }; $inputStream.Dispose() }
  } finally { if ($null -ne $response) { $response.Dispose() }; $cancellation.Dispose() }
}
function Assert-Publisher([string]$Path) {
  $signature = Get-AuthenticodeSignature -LiteralPath $Path
  if ($signature.Status -ne 'Valid' -or $null -eq $signature.TimeStamperCertificate -or
      $signature.SignerCertificate.GetNameInfo([Security.Cryptography.X509Certificates.X509NameType]::SimpleName, $false) -cne $publisher) {
    throw 'The downloaded executable does not match the Magnitude publisher.'
  }
}
try {
  $offerPath = Join-Path $scratch 'offer.json'
  Get-InstallerFile "$origin/install/$Channel/windows-x64-windows-exe.json" $offerPath 16384
  $offer = Get-Content -LiteralPath $offerPath -Raw | ConvertFrom-Json
  $version = [string]$offer.release.version
  if ($version.Length -gt 96 -or $version -notmatch '^[0-9]+\.[0-9]+\.[0-9]+(?:-[A-Za-z0-9.-]+)?(?:\+[A-Za-z0-9.-]+)?$') { throw 'Invalid application version.' }
  $download = [string]$offer.download
  if ($download -notmatch '^https://github\.com/magnitudedev/magnitude/releases/download/[^\s?#]+$') { throw 'Unexpected installer download location.' }
  $archive = Join-Path $scratch 'cli.tar.gz'
  $tag = [Uri]::EscapeDataString('@magnitudedev') + '/' + [Uri]::EscapeDataString('cli@' + $version)
  Get-InstallerFile "https://github.com/magnitudedev/magnitude/releases/download/$tag/magnitude-cli-windows-x64-msvc.tar.gz" $archive 268435456
  # Extract only the expected command, then authenticate it before executing any bootstrap code.
  & tar.exe -xf $archive -C $scratch 'bin/magnitude-cli.exe'
  if ($LASTEXITCODE -ne 0) { throw 'The application verifier could not be extracted.' }
  $verifier = Join-Path $scratch 'bin\magnitude-cli.exe'
  Assert-Publisher $verifier
  $installer = Join-Path $scratch 'magnitude-setup.exe'
  Get-InstallerFile $download $installer 2147483648
  & $verifier _verify-windows-installation $offerPath $installer $Channel
  if ($LASTEXITCODE -ne 0) { throw 'The installer failed release verification.' }
  Assert-Publisher $installer
  $process = Start-Process -FilePath $installer -ArgumentList '/S' -Wait -PassThru
  if ($process.ExitCode -ne 0) { throw "Magnitude installation failed with exit code $($process.ExitCode)." }
  $commandDirectory = Join-Path ([Environment]::GetFolderPath('LocalApplicationData')) 'Programs\Magnitude CLI'
  if (-not (($env:PATH -split ';') | Where-Object { $_.TrimEnd('\') -ieq $commandDirectory.TrimEnd('\') })) {
    $env:PATH = $commandDirectory + ';' + $env:PATH
  }
  Write-Output 'Magnitude was installed. Run magnitude serve to start the server.'
} finally {
  $client.Dispose()
  Remove-Item -LiteralPath $scratch -Recurse -Force -ErrorAction SilentlyContinue
}
