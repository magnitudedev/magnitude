# irm https://magnitude.dev/install.ps1 | iex - installs the Magnitude CLI only (for servers and headless use) for this user.
# & ([scriptblock]::Create((irm https://magnitude.dev/install.ps1))) -Desktop - installs the desktop app, which includes the CLI.
# CmdletBinding refuses unknown parameters, so a mistyped -Desktop never silently installs the CLI only.
[CmdletBinding()]
param([switch]$Desktop)
$ErrorActionPreference = 'Stop'
$Channel = 'stable'
$origin = '@MAGNITUDE_INSTALL_ORIGIN@'
$publisher = '@MAGNITUDE_WINDOWS_PUBLISHER@'
if ($env:OS -ne 'Windows_NT' -or -not [Environment]::Is64BitOperatingSystem) { throw 'This installer requires 64-bit Windows.' }
if (-not $origin.StartsWith('https://') -or $publisher.StartsWith('@MAGNITUDE_')) { throw 'The installer has no publisher configuration.' }
Add-Type -AssemblyName System.Net.Http
$client = [Net.Http.HttpClient]::new()
# Downloads end only when they stall; see Get-InstallerFile.
$client.Timeout = [Threading.Timeout]::InfiniteTimeSpan
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
  $width = 80
  if ($state.Interactive) { try { $width = [Math]::Max(2, [Console]::WindowWidth) } catch {} }
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
    if ($text.Length -ge $width) { $text = $text.Substring(0, $width - 1) }
    # Overwrite and pad in one write, without a blank frame or terminal line wrapping.
    [Console]::Error.Write("`r" + $text.PadRight([Math]::Min($state.LineWidth, $width - 1)))
    $state.LineWidth = $text.Length
  } elseif ($ReceivedBytes -lt 0) {
    [Console]::Error.WriteLine("$($state.Label) $([char]0xb7) $([int][Math]::Floor($elapsed))s elapsed")
  } else { [Console]::Error.WriteLine("$($state.Label) $([char]0xb7) $($text.Trim())") }
}
function Complete-InstallerProgress([string]$Message) {
  Stop-InstallerProgress
  [Console]::Error.WriteLine("$([char]0x2713) $Message")
}
function Wait-InstallerTask($Task, [long]$ReceivedBytes = -1, [long]$ExpectedBytes = 0, [Threading.CancellationToken]$Cancel, [IDisposable]$Abort) {
  # Wake as soon as the task completes; the timeout only paces progress redraws.
  while (-not ([IAsyncResult]$Task).AsyncWaitHandle.WaitOne(100)) {
    Update-InstallerProgress $ReceivedBytes $ExpectedBytes
    # .NET Framework network reads ignore cancellation; closing the stream ends a stalled read.
    if ($null -ne $Abort -and $Cancel.IsCancellationRequested) { $Abort.Dispose() }
  }
  return $Task.GetAwaiter().GetResult()
}
function Invoke-InstallerProcess([string]$File, [string[]]$Arguments, [string]$Label, [string]$Completed, [switch]$Quiet) {
  Start-InstallerProgress $Label
  $process = [Diagnostics.Process]::new()
  try {
    $process.StartInfo.FileName = $File
    # Windows PowerShell 5.1 has no ProcessStartInfo.ArgumentList. Quote only arguments that need it:
    # NSIS reads switches such as /S only when they are unquoted.
    $process.StartInfo.Arguments = ($Arguments | ForEach-Object {
      if ($_ -match '^[^\s"]+$') { $_ }
      else { '"' + [regex]::Replace([regex]::Replace($_, '(\\*)"', '$1$1\"'), '(\\+)$', '$1$1') + '"' }
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
    if ($output -and -not $Quiet) { [Console]::Out.Write($output) }
    if ($errorOutput) { [Console]::Error.Write($errorOutput) }
  } finally {
    Stop-InstallerProgress
    $process.Dispose()
  }
}
# A download that receives no data for this long is cancelled.
$StallMilliseconds = 60000
function Get-InstallerFile([string]$Url, [string]$Path, [long]$Limit, [string]$Label, [string]$Completed, [long]$ExpectedBytes = 0) {
  if (-not $Url.StartsWith('https://')) { throw 'Downloads require HTTPS.' }
  Start-InstallerProgress $Label
  # Slow connections may take as long as they need; only a connection that stops delivering data fails.
  $cancellation = [Threading.CancellationTokenSource]::new()
  $cancellation.CancelAfter($StallMilliseconds)
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
      while ($true) {
        $cancellation.CancelAfter($StallMilliseconds)
        $count = Wait-InstallerTask ($inputStream.ReadAsync($buffer, 0, $buffer.Length, $cancellation.Token)) $total $ExpectedBytes $cancellation.Token $response
        if ($count -le 0) { break }
        $total += $count
        if ($total -gt $Limit) { throw 'Download exceeds the installation limit.' }
        $outputStream.Write($buffer, 0, $count)
        Update-InstallerProgress $total $ExpectedBytes
      }
      $size = ($total / 1000000.0).ToString('0.0', [Globalization.CultureInfo]::InvariantCulture)
      $duration = [int][Math]::Floor($script:InstallerProgress.Clock.Elapsed.TotalSeconds)
      Complete-InstallerProgress "$Completed $([char]0xb7) $size MB $([char]0xb7) $($duration)s"
    } finally { if ($null -ne $outputStream) { $outputStream.Dispose() }; $inputStream.Dispose() }
  } catch {
    if ($cancellation.IsCancellationRequested) {
      throw "$Label stopped: no data arrived for $($StallMilliseconds / 1000) seconds. Check your network connection and run the installer again."
    }
    throw
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
# The CLI-only install: the payload in %USERPROFILE%\.magnitude\cli\<version>, `current` naming the version,
# and the launcher on PATH in the slot the desktop installer also uses. It never needs administrator rights.
$cliRoot = Join-Path ([Environment]::GetFolderPath('UserProfile')) '.magnitude\cli'
$commandDirectory = Join-Path ([Environment]::GetFolderPath('LocalApplicationData')) 'Programs\Magnitude CLI'
$desktopExecutable = Join-Path ([Environment]::GetFolderPath('LocalApplicationData')) 'Programs\Magnitude\Magnitude.exe'
$cliFiles = @('magnitude.exe', 'magnitude-service.exe', 'desktop-host.node', 'magnitude-launcher.exe', 'update-configuration.json', 'update-trust.json')
$cliSignedFiles = @('magnitude.exe', 'magnitude-service.exe', 'desktop-host.node', 'magnitude-launcher.exe')
$script:Staging = $null
function Get-CliVersion([string]$Executable) {
  if (-not (Test-Path -LiteralPath $Executable -PathType Leaf)) { return '' }
  try { $reported = & $Executable --version 2>$null } catch { return '' }
  if ($LASTEXITCODE -ne 0) { return '' }
  return ([string]$reported).Trim()
}
function Test-PathEntry([string]$Entry, [string]$Directory) {
  $value = $Entry.Trim(" `t")
  if ($value.Length -ge 2 -and $value.StartsWith('"') -and $value.EndsWith('"')) { $value = $value.Substring(1, $value.Length - 2) }
  return $value.TrimEnd('\', '/') -ieq $Directory.TrimEnd('\', '/')
}
# Mirrors the desktop installer's PATH registration: prepend to the user Path, keeping its registry type, and
# tell running programs. An existing entry is left alone.
function Add-UserPath([string]$Directory) {
  $environment = [Microsoft.Win32.Registry]::CurrentUser.CreateSubKey('Environment')
  try {
    $current = [string]$environment.GetValue('Path', '', [Microsoft.Win32.RegistryValueOptions]::DoNotExpandEnvironmentNames)
    $kind = [Microsoft.Win32.RegistryValueKind]::ExpandString
    if ($null -ne $environment.GetValue('Path')) { $kind = $environment.GetValueKind('Path') }
    if ($kind -ne [Microsoft.Win32.RegistryValueKind]::String -and $kind -ne [Microsoft.Win32.RegistryValueKind]::ExpandString) {
      throw 'The user Path in the registry is not text.'
    }
    if ($current -split ';' | Where-Object { Test-PathEntry $_ $Directory }) { return }
    $next = if ($current) { "$Directory;$current" } else { $Directory }
    $environment.SetValue('Path', $next, $kind)
  } finally { $environment.Dispose() }
  try {
    Add-Type -Namespace MagnitudeInstaller -Name Broadcast -MemberDefinition @'
[DllImport("user32.dll", CharSet = CharSet.Unicode)]
public static extern IntPtr SendMessageTimeout(IntPtr window, uint message, UIntPtr wParam, string lParam, uint flags, uint timeout, out UIntPtr result);
'@
    $ignored = [UIntPtr]::Zero
    # HWND_BROADCAST, WM_SETTINGCHANGE, SMTO_ABORTIFHUNG: Explorer then starts new terminals with the new Path.
    [void][MagnitudeInstaller.Broadcast]::SendMessageTimeout([IntPtr]0xffff, 0x1A, [UIntPtr]::Zero, 'Environment', 2, 2000, [ref]$ignored)
  } catch {
    [Console]::Error.WriteLine('Could not announce the PATH change; sign out and back in if new terminals do not find magnitude.')
  }
}
# The desktop installer publishes its launcher only into a folder that this user alone may access (a protected
# DACL with one inheritable full-control entry), so the CLI-only install gives the shared slot the same DACL.
function Initialize-CommandDirectory {
  [IO.Directory]::CreateDirectory($commandDirectory) | Out-Null
  $user = [Security.Principal.WindowsIdentity]::GetCurrent().User
  $security = New-Object Security.AccessControl.DirectorySecurity
  $security.SetOwner($user)
  $security.SetAccessRuleProtection($true, $false)
  $security.AddAccessRule((New-Object Security.AccessControl.FileSystemAccessRule($user, 'FullControl', 'ContainerInherit, ObjectInherit', 'None', 'Allow')))
  Set-Acl -LiteralPath $commandDirectory -AclObject $security
}
# A running launcher (a serve started at login, say) cannot be overwritten, but it can be renamed away.
function Install-Launcher([string]$Source) {
  Initialize-CommandDirectory
  $target = Join-Path $commandDirectory 'magnitude.exe'
  Get-ChildItem -LiteralPath $commandDirectory -Force | Where-Object { $_.Name -like 'magnitude.exe.new-*' -or $_.Name -like 'magnitude.exe.old-*' } |
    Remove-Item -Force -ErrorAction SilentlyContinue
  if ((Test-Path -LiteralPath $target -PathType Leaf) -and
      (Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash -eq (Get-FileHash -LiteralPath $Source -Algorithm SHA256).Hash) { return }
  $suffix = [Guid]::NewGuid().ToString('N')
  $incoming = "$target.new-$suffix"
  Copy-Item -LiteralPath $Source -Destination $incoming
  if (Test-Path -LiteralPath $target) {
    $retired = "$target.old-$suffix"
    Move-Item -LiteralPath $target -Destination $retired
    Remove-Item -LiteralPath $retired -Force -ErrorAction SilentlyContinue
  }
  Move-Item -LiteralPath $incoming -Destination $target
}
# Names the version in one replacement, so the launcher reads either the old or the new version.
function Set-CurrentVersion([string]$Version) {
  $current = Join-Path $cliRoot 'current'
  $temporary = Join-Path $cliRoot ('.current-' + [Guid]::NewGuid().ToString('N'))
  [IO.File]::WriteAllText($temporary, "$Version`n")
  if (Test-Path -LiteralPath $current) { [IO.File]::Replace($temporary, $current, $null) }
  else { [IO.File]::Move($temporary, $current) }
}
function Assert-CliFiles([string]$Directory) {
  $allowed = $cliFiles + @('LICENSE')
  foreach ($entry in Get-ChildItem -LiteralPath $Directory -Force) {
    if ($allowed -notcontains $entry.Name) { throw "The CLI download contains an unexpected entry: $($entry.Name)" }
    if ($entry.PSIsContainer -or ($entry.Attributes -band [IO.FileAttributes]::ReparsePoint)) { throw "The CLI download entry $($entry.Name) is not a regular file." }
  }
  foreach ($name in $cliFiles) {
    if (-not (Test-Path -LiteralPath (Join-Path $Directory $name) -PathType Leaf)) { throw "The CLI download is missing $name." }
  }
}
function Install-MagnitudeCli {
  [Console]::Error.WriteLine('Installing the Magnitude CLI only (for servers and headless use). For the desktop app, press Ctrl-C and rerun with -Desktop:')
  [Console]::Error.WriteLine('  & ([scriptblock]::Create((irm https://magnitude.dev/install.ps1))) -Desktop')
  [Console]::Error.WriteLine("`nInstalling the Magnitude CLI`nWindows $([char]0xb7) x64`n")
  if (Test-Path -LiteralPath $desktopExecutable -PathType Leaf) {
    Write-Output 'The Magnitude desktop app is installed and already includes the magnitude command. Nothing was changed.'
    return
  }
  $offerPath = Join-Path $scratch 'offer.json'
  # The landing server counts each request and answers with the publisher-signed offer.
  Get-InstallerFile "$origin/api/installer?os=windows&arch=x64&package=cli-only&offer=1" $offerPath 16384 'Checking latest release' 'Checked latest release'
  $offer = Get-Content -LiteralPath $offerPath -Raw | ConvertFrom-Json
  $version = [string]$offer.release.version
  if ($version.Length -gt 96 -or $version -notmatch '^[0-9]+\.[0-9]+\.[0-9]+(?:-[A-Za-z0-9.-]+)?(?:\+[A-Za-z0-9.-]+)?$') { throw 'Invalid application version.' }
  $download = [string]$offer.download
  if ($download -notmatch '^https://github\.com/magnitudedev/magnitude/releases/download/[^\s?#]+$') { throw 'Unexpected CLI download location.' }
  $bytes = [long]$offer.release.bytes
  if ($bytes -le 0 -or $bytes -gt 2147483648) { throw 'Invalid application size.' }
  $digest = [string]$offer.release.sha256
  if ($digest -cnotmatch '^[a-f0-9]{64}$') { throw 'Invalid application checksum.' }
  $currentPath = Join-Path $cliRoot 'current'
  $previous = ''
  if (Test-Path -LiteralPath $currentPath -PathType Leaf) { $previous = ([IO.File]::ReadAllText($currentPath)).Trim() }
  $target = Join-Path $cliRoot $version
  $updated = $false
  if ($previous -eq $version -and (Get-CliVersion (Join-Path $target 'magnitude.exe')) -eq $version) {
    [Console]::Error.WriteLine("Magnitude CLI $version")
  } else {
    if ($previous) { [Console]::Error.WriteLine("Magnitude CLI $version (installed: $previous)") }
    else { [Console]::Error.WriteLine("Magnitude CLI $version") }
    $archive = Join-Path $scratch 'magnitude-cli-only.tar.gz'
    Get-InstallerFile $download $archive 2147483648 'Downloading CLI' 'Downloaded CLI' $bytes
    Start-InstallerProgress 'Verifying download checksum'
    if ((Get-Item -LiteralPath $archive).Length -ne $bytes) { throw 'The CLI download is incomplete.' }
    if ((Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLowerInvariant() -cne $digest) { throw 'The CLI checksum does not match.' }
    Complete-InstallerProgress 'Verified download checksum'
    [IO.Directory]::CreateDirectory($cliRoot) | Out-Null
    $script:Staging = Join-Path $cliRoot ".staging-$version"
    if (Test-Path -LiteralPath $script:Staging) { Remove-Item -LiteralPath $script:Staging -Recurse -Force }
    [IO.Directory]::CreateDirectory($script:Staging) | Out-Null
    # Windows' bsdtar refuses members that climb out of the folder; only regular top-level files are kept.
    Invoke-InstallerProcess (Join-Path $env:SystemRoot 'System32\tar.exe') @('-xzf', $archive, '-C', $script:Staging) 'Extracting CLI' 'Extracted CLI'
    Assert-CliFiles $script:Staging
    $staged = Join-Path $script:Staging 'magnitude.exe'
    Start-InstallerProgress 'Verifying CLI signatures'
    foreach ($name in $cliSignedFiles) { Assert-Publisher (Join-Path $script:Staging $name) }
    Complete-InstallerProgress 'Verified CLI signatures'
    # The publisher-signed CLI checks the offer's publisher signature against this archive.
    Invoke-InstallerProcess $staged @('_verify-windows-installation', $offerPath, $archive, $Channel, 'cli-only') 'Verifying CLI release' 'Verified CLI release'
    $reported = Get-CliVersion $staged
    if ($reported -ne $version) { throw "The downloaded magnitude command reports version '$reported', not $version." }
    if (Test-Path -LiteralPath $target) { Remove-Item -LiteralPath $target -Recurse -Force }
    Move-Item -LiteralPath $script:Staging -Destination $target
    $script:Staging = $null
    Set-CurrentVersion $version
    # Keep the version a running serve may still use; a folder still in use stays for the next run.
    Get-ChildItem -LiteralPath $cliRoot -Force | Where-Object { @('current', $version, $previous) -notcontains $_.Name } |
      Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
    $updated = $true
  }
  Install-Launcher (Join-Path $target 'magnitude-launcher.exe')
  $onPath = [bool](($env:PATH -split ';') | Where-Object { Test-PathEntry $_ $commandDirectory })
  Add-UserPath $commandDirectory
  if (-not $onPath) { $env:PATH = $commandDirectory + ';' + $env:PATH }
  if ($updated) {
    Write-Output "$([char]0x2713) Installed the Magnitude CLI $version (CLI only, no desktop app)"
    Write-Output ''
    Write-Output 'Next: run `magnitude docs` for how to use `magnitude serve` and how to'
    Write-Output 'keep it running in the background at login on macOS, Linux and Windows.'
  } else {
    Write-Output "The Magnitude CLI $version is already up to date (CLI only, no desktop app)."
  }
  if (-not $onPath) { Write-Output 'Open a new terminal to use the magnitude command.' }
}
function Install-MagnitudeDesktop {
  # Setup refuses to replace a running app, and its own message is hidden by /S. Say so before downloading.
  $installed = Join-Path ([Environment]::GetFolderPath('LocalApplicationData')) 'Programs\Magnitude\'
  if (Get-Process | Where-Object { $_.Path -and $_.Path.StartsWith($installed, [StringComparison]::OrdinalIgnoreCase) }) {
    throw 'Magnitude is running. Quit it using Quit from its tray menu, then run the installer again.'
  }
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
  # The desktop app replaces a CLI-only install; its launcher already took the command slot.
  $cliInstall = Join-Path ([Environment]::GetFolderPath('UserProfile')) '.magnitude\cli'
  if (Test-Path -LiteralPath $cliInstall) {
    try { Remove-Item -LiteralPath $cliInstall -Recurse -Force }
    catch { [Console]::Error.WriteLine("Could not remove the CLI-only install at $cliInstall; remove it once nothing uses it.") }
  }
  Write-Output 'Magnitude was installed.'
  Open-InstalledDesktop (Join-Path $commandDirectory 'magnitude.exe')
}
function Open-InstalledDesktop([string]$Command) {
  # Services, SSH and remote automation do not have a desktop to show the window on.
  if (-not [Environment]::UserInteractive -or (Get-Process -Id $PID).SessionId -eq 0 -or
      $env:SSH_CONNECTION -or $env:SSH_CLIENT -or $env:SSH_TTY) { return }
  try {
    Invoke-InstallerProcess $Command @('app', 'open') 'Opening Magnitude' 'Opened Magnitude' -Quiet
  } catch {
    [Console]::Error.WriteLine('Magnitude is installed, but could not open automatically. Open it from the Start menu.')
  }
}
try {
  if ($Desktop) { Install-MagnitudeDesktop } else { Install-MagnitudeCli }
} finally {
  Stop-InstallerProgress
  $client.Dispose()
  if ($script:Staging) { Remove-Item -LiteralPath $script:Staging -Recurse -Force -ErrorAction SilentlyContinue }
  Remove-Item -LiteralPath $scratch -Recurse -Force -ErrorAction SilentlyContinue
}
