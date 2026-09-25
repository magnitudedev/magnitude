param([Parameter(Mandatory=$true)][string]$Root)
$ErrorActionPreference = 'Stop'
$installation = Join-Path $env:LOCALAPPDATA 'Programs\Magnitude'
if (Test-Path $installation) { throw 'Continuation probe requires an unused test installation' }
$data = Join-Path $Root 'foreground-profile'
New-Item -ItemType Directory -Force $data | Out-Null
$env:MAGNITUDE_DEV_DATA_DIR = $data
$env:MAGNITUDE_DESKTOP_STATE_DIR = Join-Path $data 'state'
$script:installerUncertain = $false
function Install-Version([string]$Version) {
  $installer = Join-Path $Root "$Version\magnitude-desktop-windows-x64-$Version.exe"
  $process = Start-Process -FilePath $installer -ArgumentList '/S' -PassThru
  try {
    if (!$process.WaitForExit(60000)) {
      $script:installerUncertain = $true
      throw "Installer process $($process.Id) has not finished; inspect it before attempting cleanup or another installation"
    }
    return $process.ExitCode
  } finally { $process.Dispose() }
}
function Installed-Version {
  return [IO.File]::ReadAllText((Join-Path $installation 'resources\fixture-version.txt'))
}
$foreground = $null
try {
  if ((Install-Version '1.2.3') -ne 0) { throw 'Initial fixture installation failed' }
  $start = [Diagnostics.ProcessStartInfo]::new((Join-Path $installation 'resources\magnitude.exe'))
  $start.UseShellExecute = $false
  $start.WorkingDirectory = $Root
  $start.RedirectStandardInput = $true
  $start.RedirectStandardOutput = $true
  $foreground = [Diagnostics.Process]::Start($start)
  $ready = $foreground.StandardOutput.ReadLineAsync()
  if (!$ready.Wait(15000) -or $ready.Result -ne 'ready') { throw 'Compiled foreground runtime did not load its native addon' }
  $first = Install-Version '1.2.4'
  $afterFirst = Installed-Version
  $second = Install-Version '1.2.5'
  $afterSecond = Installed-Version
  if ($foreground.HasExited) { throw 'Installer interrupted the mapped foreground process' }
  $foreground.StandardInput.Close()
  if (!$foreground.WaitForExit(15000)) { throw 'Foreground probe did not finish after input closed' }
  if ($foreground.ExitCode -ne 0) { throw 'Foreground process failed after replacement' }
  $afterExit = Install-Version '1.2.5'
  $result = [ordered]@{
    firstExit=$first; firstVersion=$afterFirst; secondExit=$second; secondVersion=$afterSecond
    afterParentExit=$afterExit; finalVersion=(Installed-Version)
    repeatedReplacementWithMappedParent=($first -eq 0 -and $second -eq 0 -and $afterSecond -eq '1.2.5')
  }
  $result | ConvertTo-Json | Tee-Object -FilePath (Join-Path $Root 'foreground-continuation.json')
  if ($afterExit -ne 0 -or (Installed-Version) -ne '1.2.5') { throw 'Installation did not recover after the mapped parent exited' }
} finally {
  if ($foreground) {
    if (!$foreground.HasExited) {
      $foreground.StandardInput.Close()
      if (!$foreground.WaitForExit(15000)) { $foreground.Kill(); $foreground.WaitForExit() }
    }
    $foreground.Dispose()
  }
  $uninstaller = Join-Path $installation 'Uninstall Magnitude.exe'
  if (!$script:installerUncertain -and (Test-Path $uninstaller)) {
    $process = Start-Process -FilePath $uninstaller -ArgumentList '/S' -PassThru
    try { if (!$process.WaitForExit(60000) -or $process.ExitCode -ne 0) { throw 'Fixture uninstallation failed' } }
    finally { $process.Dispose() }
  }
}
