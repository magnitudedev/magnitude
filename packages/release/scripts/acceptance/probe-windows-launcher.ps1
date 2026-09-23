param([Parameter(Mandatory=$true)][string]$Root)
$ErrorActionPreference = 'Stop'
$installation = Join-Path $env:LOCALAPPDATA 'Programs\Magnitude'
if (Test-Path $installation) { throw 'Launcher probe requires an unused test installation' }
$env:MAGNITUDE_DEV_DATA_DIR = Join-Path $Root 'launcher-profile'
New-Item -ItemType Directory -Force $env:MAGNITUDE_DEV_DATA_DIR | Out-Null
$env:MAGNITUDE_DESKTOP_STATE_DIR = Join-Path $env:MAGNITUDE_DEV_DATA_DIR 'state'
$script:uncertain = $false
$foreground = $null
$stage = Join-Path $env:LOCALAPPDATA 'Programs\Magnitude-installation-stage'
$registration = 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall\MagnitudeDesktop'
function Uninstall-Fixture([int]$Expected) {
  # Run the exact self-copy directly so its retained process reports the removal result.
  # The normal NSIS bootstrap exits before its temporary uninstaller completes.
  $copy = Join-Path $Root ('fixture-uninstall-' + [Guid]::NewGuid() + '.exe')
  Copy-Item -LiteralPath (Join-Path $installation 'Uninstall Magnitude.exe') -Destination $copy -Force
  $process = Start-Process -FilePath $copy -ArgumentList @('/S', "_?=$installation") -PassThru
  try {
    if (!$process.WaitForExit(60000)) {
      $script:uncertain = $true
      throw "Uninstaller $($process.Id) has not finished; do not overlap cleanup"
    }
    if ($process.ExitCode -ne $Expected) { throw "Uninstaller returned $($process.ExitCode), expected $Expected" }
  } finally { $process.Dispose() }
}
function Assert-Removed {
  if ((Test-Path $installation) -or (Test-Path $stage) -or (Test-Path $registration)) { throw 'Uninstall left installation or recovery state behind' }
}
function Install-Version([string]$Version) {
  $installer = Join-Path $Root "$Version\magnitude-desktop-windows-x64-$Version.exe"
  $process = Start-Process -FilePath $installer -ArgumentList '/S' -PassThru
  try {
    if (!$process.WaitForExit(60000)) {
      $script:uncertain = $true
      throw "Installer $($process.Id) has not finished; do not overlap replacement or cleanup"
    }
    if ($process.ExitCode -ne 0) { throw "Installer $Version failed with $($process.ExitCode)" }
  } finally { $process.Dispose() }
}
function Read-Ready($Process) {
  $line = $Process.StandardOutput.ReadLineAsync()
  if (!$line.Wait(15000)) { throw "Foreground CLI readiness timed out (launcher exited: $($Process.HasExited))" }
  if ($line.Result -ne 'ready') { throw "Unexpected foreground output: $($line.Result)" }
  $line = $Process.StandardOutput.ReadLineAsync()
  if (!$line.Wait(15000)) { throw 'Missing command context' }
  $context = $line.Result | ConvertFrom-Json
  $expected = @('serve', '--launcher-probe', '', 'space value', 'quote"value', 'trailing\', ('Unicode-' + [char]0x03bb))
  if ($context.args.Count -ne $expected.Count) { throw 'Argument count changed' }
  for ($i=0; $i -lt $expected.Count; ++$i) {
    if ($context.args[$i] -cne $expected[$i]) { throw "Argument $i changed" }
  }
  if ($context.cwd -cne $Root) { throw 'Working directory changed' }
}
function Start-Foreground {
  $start = [Diagnostics.ProcessStartInfo]::new((Join-Path $Root 'magnitude-launcher.exe'))
  $start.UseShellExecute = $false
  $start.WorkingDirectory = $Root
  $start.Arguments = 'serve --launcher-probe "" "space value" "quote\"value" "trailing\\" "Unicode-' + [char]0x03bb + '"'
  $start.RedirectStandardInput = $true
  $start.RedirectStandardOutput = $true
  $start.StandardOutputEncoding = [Text.Encoding]::UTF8
  # Native known-folder lookup must remain authoritative.
  $start.EnvironmentVariables['LOCALAPPDATA'] = 'Z:\not-the-installation'
  return [Diagnostics.Process]::Start($start)
}
try {
  Install-Version '1.2.3'
  foreach ($version in @('1.2.4', '1.2.5')) {
    $foreground = Start-Foreground
    Read-Ready $foreground
    Write-Output "Initial CLI ready before $version"
    Install-Version $version
    $prior = Join-Path $stage 'previous'
    if (!(Test-Path $prior)) { throw 'Replacement did not retain the mapped old payload' }
    $pathBefore = [Environment]::GetEnvironmentVariable('Path', 'User')
    $unknown = Join-Path $prior 'unrelated-fixture.txt'
    [IO.File]::WriteAllText($unknown, 'preserve this file')
    try {
      Uninstall-Fixture 1
      if ([IO.File]::ReadAllText($unknown) -ne 'preserve this file') { throw 'Removal changed an unrelated previous file' }
    } finally { Remove-Item -LiteralPath $unknown }
    Uninstall-Fixture 1
    if ((Get-ItemProperty $registration).DisplayVersion -ne $version -or
        [IO.File]::ReadAllText((Join-Path $installation 'resources\fixture-version.txt')) -ne $version -or
        [Environment]::GetEnvironmentVariable('Path', 'User') -cne $pathBefore -or
        !(Test-Path (Join-Path $installation 'Uninstall Magnitude.exe'))) { throw 'Deferred removal changed the current installation' }
    Write-Output 'PASS unexpected and mapped previous files defer uninstall without changing current installation'
    Write-Output "Installed $version; requesting continuation"
    $foreground.StandardInput.WriteLine('continue')
    $foreground.StandardInput.Flush()
    Read-Ready $foreground
    $foreground.StandardInput.Close()
    if (!$foreground.WaitForExit(15000) -or $foreground.ExitCode -ne 0) { throw 'Continued foreground command did not exit successfully' }
    $foreground.Dispose(); $foreground = $null
    Write-Output "PASS native foreground continuation into $version with original arguments and cwd"
  }
  $foreground = Start-Foreground
  Read-Ready $foreground
  $foreground.StandardInput.WriteLine('continue')
  $foreground.StandardInput.Flush()
  if (!$foreground.WaitForExit(15000) -or $foreground.ExitCode -ne 1) { throw 'Unchanged executable was allowed to continue' }
  $foreground.Dispose(); $foreground = $null
  Write-Output 'PASS unchanged payload refuses continuation'
  Uninstall-Fixture 0
  Assert-Removed
  Install-Version '1.2.3'
  Uninstall-Fixture 0
  Assert-Removed
  Write-Output 'PASS retained payload cleanup, fresh reinstall and repeated uninstall'
} finally {
  if ($foreground) {
    if (!$foreground.HasExited) {
      $foreground.StandardInput.Close()
      if (!$foreground.WaitForExit(15000)) { $foreground.Kill(); $foreground.WaitForExit() }
    }
    $foreground.Dispose()
  }
  $uninstaller = Join-Path $installation 'Uninstall Magnitude.exe'
  if (!$script:uncertain -and (Test-Path $uninstaller)) {
    Uninstall-Fixture 0
    Assert-Removed
  }
}
