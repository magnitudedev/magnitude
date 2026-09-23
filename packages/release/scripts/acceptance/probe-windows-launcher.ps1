param([Parameter(Mandatory=$true)][string]$Root)
$ErrorActionPreference = 'Stop'
$installation = Join-Path $env:LOCALAPPDATA 'Programs\Magnitude'
if (Test-Path $installation) { throw 'Launcher probe requires an unused test installation' }
$env:MAGNITUDE_DEV_DATA_DIR = Join-Path $Root 'launcher-profile'
New-Item -ItemType Directory -Force $env:MAGNITUDE_DEV_DATA_DIR | Out-Null
$env:MAGNITUDE_DESKTOP_STATE_DIR = Join-Path $env:MAGNITUDE_DEV_DATA_DIR 'state'
$script:uncertain = $false
$foreground = $null
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
  $expected = @('--launcher-probe', '', 'space value', 'quote"value', 'trailing\', ('Unicode-' + [char]0x03bb))
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
  $start.Arguments = '--launcher-probe "" "space value" "quote\"value" "trailing\\" "Unicode-' + [char]0x03bb + '"'
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
    $process = Start-Process -FilePath $uninstaller -ArgumentList '/S' -PassThru
    try { if (!$process.WaitForExit(60000) -or $process.ExitCode -ne 0) { throw 'Fixture uninstallation failed' } }
    finally { $process.Dispose() }
  }
}
