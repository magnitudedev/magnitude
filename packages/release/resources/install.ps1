# irm https://magnitude.dev/install.ps1 | iex - installs the Magnitude desktop app for this user. It takes no options.
$ErrorActionPreference = 'Stop'
$Channel = 'stable'
$origin = '@MAGNITUDE_INSTALL_ORIGIN@'
$publisher = '@MAGNITUDE_WINDOWS_PUBLISHER@'
if ($env:OS -ne 'Windows_NT' -or -not [Environment]::Is64BitOperatingSystem) { Write-Host 'Magnitude requires 64-bit Windows.' -ForegroundColor Red; return }
if (-not $origin.StartsWith('https://') -or $publisher.StartsWith('@MAGNITUDE_')) { Write-Host 'This install script has no publisher configuration. Download it again from https://magnitude.dev.' -ForegroundColor Red; return }
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
$retry = 'Check your internet connection, then run the install command again.'
$report = 'Run the install command again. If this keeps happening, something on your network may be changing downloads; report it at https://github.com/magnitudedev/magnitude/issues.'
$step = @{}
# Names the step that is running, so a failure says what went wrong and what to do next.
function Start-Step([string]$Progress, [string]$Failure, [string]$Hint) {
  Write-Host $Progress
  $step.Failure = $Failure
  $step.Hint = $Hint
}
function Assert-Publisher([string]$Path) {
  $signature = Get-AuthenticodeSignature -LiteralPath $Path
  if ($signature.Status -ne 'Valid' -or $null -eq $signature.TimeStamperCertificate -or
      $signature.SignerCertificate.GetNameInfo([Security.Cryptography.X509Certificates.X509NameType]::SimpleName, $false) -cne $publisher) {
    throw 'The downloaded executable does not match the Magnitude publisher.'
  }
}
try {
  Start-Step 'Finding the latest version of Magnitude...' "Couldn't get the latest version of Magnitude from $origin." $retry
  $offerPath = Join-Path $scratch 'offer.json'
  # The landing server counts each request and answers with the publisher-signed offer.
  Get-InstallerFile "$origin/api/installer?os=windows&arch=x64&package=windows-exe&offer=1" $offerPath 16384
  $offer = Get-Content -LiteralPath $offerPath -Raw | ConvertFrom-Json
  $version = [string]$offer.release.version
  if ($version.Length -gt 96 -or $version -notmatch '^[0-9]+\.[0-9]+\.[0-9]+(?:-[A-Za-z0-9.-]+)?(?:\+[A-Za-z0-9.-]+)?$') { throw 'Invalid application version.' }
  $download = [string]$offer.download
  if ($download -notmatch '^https://github\.com/magnitudedev/magnitude/releases/download/[^\s?#]+$') { throw 'Unexpected installer download location.' }
  Start-Step "Downloading Magnitude $version..." "Couldn't download Magnitude $version from GitHub." 'Check your internet connection and that github.com is reachable, then run the install command again.'
  $archive = Join-Path $scratch 'cli.tar.gz'
  $tag = [Uri]::EscapeDataString('@magnitudedev') + '/' + [Uri]::EscapeDataString('cli@' + $version)
  Get-InstallerFile "https://github.com/magnitudedev/magnitude/releases/download/$tag/magnitude-cli-windows-x64-msvc.tar.gz" $archive 268435456
  Start-Step 'Checking the download is signed by Magnitude...' "The downloaded Magnitude command couldn't be confirmed as signed by $publisher, so it wasn't run." $report
  # Extract only the expected command, then authenticate it before executing any bootstrap code.
  & tar.exe -xf $archive -C $scratch 'bin/magnitude-cli.exe'
  if ($LASTEXITCODE -ne 0) { throw 'The application verifier could not be extracted.' }
  $verifier = Join-Path $scratch 'bin\magnitude-cli.exe'
  Assert-Publisher $verifier
  Start-Step 'Downloading the installer...' "Couldn't download the Magnitude $version installer." 'Check your internet connection and that github.com is reachable, then run the install command again.'
  $installer = Join-Path $scratch 'magnitude-setup.exe'
  Get-InstallerFile $download $installer 2147483648
  Start-Step 'Verifying the installer...' "The installer didn't pass verification, so it wasn't run." $report
  & $verifier _verify-windows-installation $offerPath $installer $Channel
  if ($LASTEXITCODE -ne 0) { throw 'The installer failed release verification.' }
  Assert-Publisher $installer
  Start-Step 'Installing Magnitude...' 'The Magnitude installer failed.' 'If Magnitude is open, quit it from its tray menu, then run the install command again.'
  $process = Start-Process -FilePath $installer -ArgumentList '/S' -Wait -PassThru
  if ($process.ExitCode -ne 0) { throw "Magnitude installation failed with exit code $($process.ExitCode)." }
  $commandDirectory = Join-Path ([Environment]::GetFolderPath('LocalApplicationData')) 'Programs\Magnitude CLI'
  if (-not (($env:PATH -split ';') | Where-Object { $_.TrimEnd('\') -ieq $commandDirectory.TrimEnd('\') })) {
    $env:PATH = $commandDirectory + ';' + $env:PATH
  }
  Write-Host "Magnitude $version is installed. Open it from the Start menu, or run ``magnitude`` in a terminal." -ForegroundColor Green
} catch {
  Write-Host ($step.Failure) -ForegroundColor Red
  Write-Host ($step.Hint)
  Write-Host "Details: $($_.Exception.GetBaseException().Message)" -ForegroundColor DarkGray
} finally {
  $client.Dispose()
  Remove-Item -LiteralPath $scratch -Recurse -Force -ErrorAction SilentlyContinue
}
