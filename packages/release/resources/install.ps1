# irm https://magnitude.dev/install.ps1 | iex - installs the Magnitude desktop app for this user. It takes no options.
$ErrorActionPreference = 'Stop'
$Channel = 'stable'
$origin = '@MAGNITUDE_INSTALL_ORIGIN@'
$publisher = '@MAGNITUDE_WINDOWS_PUBLISHER@'
if ($env:OS -ne 'Windows_NT' -or -not [Environment]::Is64BitOperatingSystem) { throw 'This installer requires 64-bit Windows.' }
if (-not $origin.StartsWith('https://') -or $publisher.StartsWith('@MAGNITUDE_')) { throw 'The installer has no publisher configuration.' }
Add-Type -AssemblyName System.Net.Http
$client = [Net.Http.HttpClient]::new()
$client.Timeout = [TimeSpan]::FromMinutes(10)
$scratch = Join-Path ([IO.Path]::GetTempPath()) ('magnitude-install-' + [Guid]::NewGuid().ToString('N'))
[IO.Directory]::CreateDirectory($scratch) | Out-Null
# Keep status off the success stream. Animated output is only used by an interactive console.
$script:InstallerProgress = $null
function Stop-InstallerProgress {
  $state = $script:InstallerProgress
  if ($null -ne $state -and $state.Interactive) {
    [Console]::Error.Write("`r" + (' ' * $state.LineWidth) + "`r")
  }
  $script:InstallerProgress = $null
}
function Start-InstallerProgress([string]$Label) {
  Stop-InstallerProgress
  $interactive = -not [Console]::IsErrorRedirected -and $env:TERM -ne 'dumb'
  $script:InstallerProgress = [pscustomobject]@{
    Label = $Label; Clock = [Diagnostics.Stopwatch]::StartNew(); LastUpdate = 0.0
    Interactive = $interactive; LineWidth = $Label.Length + 5
  }
  if ($interactive) { [Console]::Error.Write("  $Label...") }
  else { [Console]::Error.WriteLine("$Label...") }
}
function Update-InstallerProgress([long]$ReceivedBytes = -1, [long]$ExpectedBytes = 0) {
  $state = $script:InstallerProgress
  if ($null -eq $state) { return }
  $elapsed = $state.Clock.Elapsed.TotalSeconds
  $interval = if ($state.Interactive) { 0.2 } else { 10.0 }
  if ($elapsed - $state.LastUpdate -lt $interval) { return }
  $state.LastUpdate = $elapsed
  if ($ReceivedBytes -lt 0) {
    $frame = @('|', '/', '-', '\')[[int][Math]::Floor($elapsed * 4) % 4]
    $text = "  $frame $($state.Label) $([char]0xb7) $([int][Math]::Floor($elapsed))s"
  } else {
    $received = ($ReceivedBytes / 1000000.0).ToString('0.0', [Globalization.CultureInfo]::InvariantCulture)
    if ($ExpectedBytes -gt 0) {
      $percent = [Math]::Min(100, [int][Math]::Floor($ReceivedBytes * 100.0 / $ExpectedBytes))
      $filled = [int][Math]::Floor($percent * 12.0 / 100)
      $bar = ('=' * $filled) + ('-' * (12 - $filled))
      $total = ($ExpectedBytes / 1000000.0).ToString('0.0', [Globalization.CultureInfo]::InvariantCulture)
      $width = 80
      if ($state.Interactive) { try { $width = [Console]::WindowWidth } catch {} }
      if ($width -ge 75) { $text = "  [$bar] $percent%  $received/$total MB" }
      else { $text = "  $percent%  $received/$total MB" }
    } else { $text = "  $received MB" }
    if ($elapsed -ge 1 -and $ReceivedBytes -gt 0) {
      $speed = $ReceivedBytes / $elapsed
      $rate = ($speed / 1000000.0).ToString('0.0', [Globalization.CultureInfo]::InvariantCulture)
      $text += "  $rate MB/s"
      if ($ExpectedBytes -gt 0) {
        $remaining = [int][Math]::Ceiling([Math]::Max(0, $ExpectedBytes - $ReceivedBytes) / $speed)
        $text += "  ~$($remaining)s left"
      }
    } else { $text += '  estimating time remaining' }
  }
  if ($state.Interactive) {
    [Console]::Error.Write("`r" + $text.PadRight($state.LineWidth))
    $state.LineWidth = $text.Length
  } elseif ($ReceivedBytes -lt 0) {
    [Console]::Error.WriteLine("$($state.Label) $([char]0xb7) $([int][Math]::Floor($elapsed))s elapsed")
  } else { [Console]::Error.WriteLine("$($state.Label) $([char]0xb7) $($text.Trim())") }
}
function Complete-InstallerProgress([string]$Message) {
  Stop-InstallerProgress
  [Console]::Error.WriteLine("$([char]0x2713) $Message")
}
function Wait-InstallerTask($Task, [long]$ReceivedBytes = -1, [long]$ExpectedBytes = 0) {
  while (-not $Task.IsCompleted) {
    Update-InstallerProgress $ReceivedBytes $ExpectedBytes
    Start-Sleep -Milliseconds 100
  }
  return $Task.GetAwaiter().GetResult()
}
function Invoke-InstallerProcess([string]$File, [string[]]$Arguments, [string]$Label, [string]$Completed) {
  Start-InstallerProgress $Label
  $process = [Diagnostics.Process]::new()
  try {
    $process.StartInfo.FileName = $File
    # Windows PowerShell 5.1 has no ProcessStartInfo.ArgumentList. Quote each argument for Windows.
    $process.StartInfo.Arguments = ($Arguments | ForEach-Object {
      '"' + [regex]::Replace([regex]::Replace($_, '(\\*)"', '$1$1\"'), '(\\+)$', '$1$1') + '"'
    }) -join ' '
    $process.StartInfo.UseShellExecute = $false
    $process.StartInfo.CreateNoWindow = $true
    $process.StartInfo.RedirectStandardOutput = $true
    $process.StartInfo.RedirectStandardError = $true
    if (-not $process.Start()) { throw "Could not start $Label." }
    $stdout = $process.StandardOutput.ReadToEndAsync()
    $stderr = $process.StandardError.ReadToEndAsync()
    while (-not $process.WaitForExit(200)) { Update-InstallerProgress }
    $output = $stdout.GetAwaiter().GetResult()
    $errorOutput = $stderr.GetAwaiter().GetResult()
    if ($process.ExitCode -ne 0) {
      throw "$Label failed with exit code $($process.ExitCode). $($errorOutput.Trim())"
    }
    Complete-InstallerProgress $Completed
    if ($output) { [Console]::Out.Write($output) }
    if ($errorOutput) { [Console]::Error.Write($errorOutput) }
  } finally {
    Stop-InstallerProgress
    $process.Dispose()
  }
}
function Get-InstallerFile([string]$Url, [string]$Path, [long]$Limit, [string]$Label, [string]$Completed, [long]$ExpectedBytes = 0) {
  if (-not $Url.StartsWith('https://')) { throw 'Downloads require HTTPS.' }
  Start-InstallerProgress $Label
  $cancellation = [Threading.CancellationTokenSource]::new(600000)
  $response = $null
  try {
    $response = Wait-InstallerTask ($client.GetAsync($Url, [Net.Http.HttpCompletionOption]::ResponseHeadersRead, $cancellation.Token)) 0 $ExpectedBytes
    $response.EnsureSuccessStatusCode() | Out-Null
    if ($response.RequestMessage.RequestUri.Scheme -ne 'https') { throw 'Unexpected download redirect.' }
    if ($response.Content.Headers.ContentLength -gt $Limit) { throw 'Download exceeds the installation limit.' }
    if ($ExpectedBytes -le 0 -and $response.Content.Headers.ContentLength -gt 0) { $ExpectedBytes = $response.Content.Headers.ContentLength }
    $inputStream = Wait-InstallerTask ($response.Content.ReadAsStreamAsync()) 0 $ExpectedBytes
    $outputStream = $null
    try {
      $outputStream = [IO.File]::Open($Path, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write, [IO.FileShare]::None)
      $buffer = New-Object byte[] 65536
      [long]$total = 0
      while (($count = Wait-InstallerTask ($inputStream.ReadAsync($buffer, 0, $buffer.Length, $cancellation.Token)) $total $ExpectedBytes) -gt 0) {
        $total += $count
        if ($total -gt $Limit) { throw 'Download exceeds the installation limit.' }
        $outputStream.Write($buffer, 0, $count)
        Update-InstallerProgress $total $ExpectedBytes
      }
      $size = ($total / 1000000.0).ToString('0.0', [Globalization.CultureInfo]::InvariantCulture)
      $duration = [int][Math]::Floor($script:InstallerProgress.Clock.Elapsed.TotalSeconds)
      Complete-InstallerProgress "$Completed $([char]0xb7) $size MB $([char]0xb7) $($duration)s"
    } finally { if ($null -ne $outputStream) { $outputStream.Dispose() }; $inputStream.Dispose() }
  } finally {
    Stop-InstallerProgress
    if ($null -ne $response) { $response.Dispose() }
    $cancellation.Dispose()
  }
}
function Assert-Publisher([string]$Path) {
  $verification = [PowerShell]::Create()
  try {
    $null = $verification.AddCommand('Get-AuthenticodeSignature').AddParameter('LiteralPath', $Path).AddParameter('ErrorAction', 'Stop')
    $pending = $verification.BeginInvoke()
    while (-not $pending.IsCompleted) {
      Update-InstallerProgress
      Start-Sleep -Milliseconds 100
    }
    $signatures = $verification.EndInvoke($pending)
    if ($verification.HadErrors -or $signatures.Count -ne 1) { throw 'Could not verify the downloaded executable signature.' }
    $signature = $signatures[0]
  } finally { $verification.Dispose() }
  if ($signature.Status -ne 'Valid' -or $null -eq $signature.TimeStamperCertificate -or
      $signature.SignerCertificate.GetNameInfo([Security.Cryptography.X509Certificates.X509NameType]::SimpleName, $false) -cne $publisher) {
    throw 'The downloaded executable does not match the Magnitude publisher.'
  }
}
try {
  $offerPath = Join-Path $scratch 'offer.json'
  # The landing server counts each request and answers with the publisher-signed offer.
  [Console]::Error.WriteLine("`nInstalling Magnitude`nWindows $([char]0xb7) x64`n")
  Get-InstallerFile "$origin/api/installer?os=windows&arch=x64&package=windows-exe&offer=1" $offerPath 16384 'Checking latest release' 'Checked latest release'
  $offer = Get-Content -LiteralPath $offerPath -Raw | ConvertFrom-Json
  $version = [string]$offer.release.version
  if ($version.Length -gt 96 -or $version -notmatch '^[0-9]+\.[0-9]+\.[0-9]+(?:-[A-Za-z0-9.-]+)?(?:\+[A-Za-z0-9.-]+)?$') { throw 'Invalid application version.' }
  [Console]::Error.WriteLine("Magnitude $version")
  $download = [string]$offer.download
  if ($download -notmatch '^https://github\.com/magnitudedev/magnitude/releases/download/[^\s?#]+$') { throw 'Unexpected installer download location.' }
  $archive = Join-Path $scratch 'cli.tar.gz'
  $tag = [Uri]::EscapeDataString('@magnitudedev') + '/' + [Uri]::EscapeDataString('cli@' + $version)
  Get-InstallerFile "https://github.com/magnitudedev/magnitude/releases/download/$tag/magnitude-cli-windows-x64-msvc.tar.gz" $archive 268435456 'Downloading installer helper' 'Downloaded installer helper'
  # Extract only the expected command, then authenticate it before executing any bootstrap code.
  Invoke-InstallerProcess 'tar.exe' @('-xf', $archive, '-C', $scratch, 'bin/magnitude-cli.exe') 'Extracting installer helper' 'Extracted installer helper'
  $verifier = Join-Path $scratch 'bin\magnitude-cli.exe'
  Start-InstallerProgress 'Verifying helper signature'
  Assert-Publisher $verifier
  Complete-InstallerProgress 'Verified helper signature'
  $installer = Join-Path $scratch 'magnitude-setup.exe'
  Get-InstallerFile $download $installer 2147483648 'Downloading app' 'Downloaded app' ([long]$offer.release.bytes)
  Invoke-InstallerProcess $verifier @('_verify-windows-installation', $offerPath, $installer, $Channel) 'Verifying installer release and checksum' 'Verified installer release and checksum'
  Start-InstallerProgress 'Verifying installer signature'
  Assert-Publisher $installer
  Complete-InstallerProgress 'Verified installer signature'
  Invoke-InstallerProcess $installer @('/S') 'Installing app and CLI' 'Installed app and CLI'
  $commandDirectory = Join-Path ([Environment]::GetFolderPath('LocalApplicationData')) 'Programs\Magnitude CLI'
  if (-not (($env:PATH -split ';') | Where-Object { $_.TrimEnd('\') -ieq $commandDirectory.TrimEnd('\') })) {
    $env:PATH = $commandDirectory + ';' + $env:PATH
  }
  Write-Output 'Magnitude was installed. Open it from the Start menu.'
} finally {
  Stop-InstallerProgress
  $client.Dispose()
  Remove-Item -LiteralPath $scratch -Recurse -Force -ErrorAction SilentlyContinue
}
