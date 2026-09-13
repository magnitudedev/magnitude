param([Parameter(Mandatory=$true)][string]$Root)
$ErrorActionPreference = 'Stop'
$installation = Join-Path $env:LOCALAPPDATA 'Programs\Magnitude'
$registration = 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall\MagnitudeDesktop'
if (Test-Path $installation) { throw 'Installer fixture requires an unused installation path' }
if (Test-Path $registration) { throw 'Installer fixture requires an unused registration' }
function Invoke-Installer([string]$Path, [int]$Expected) {
  $process = Start-Process -FilePath $Path -ArgumentList '/S' -PassThru
  if (!$process.WaitForExit(60000)) {
    Stop-Process -Id $process.Id -Force
    throw 'Installer fixture timed out'
  }
  if ($process.ExitCode -ne $Expected) { throw "Installer returned $($process.ExitCode), expected $Expected" }
}
function Assert-Version([string]$Version) {
  if ((Get-ItemProperty $registration).DisplayVersion -ne $Version) { throw 'Registered version differs' }
  if ([IO.File]::ReadAllText((Join-Path $installation 'resources\fixture-version.txt')) -ne $Version) { throw 'Installed payload version differs' }
}
$old = Join-Path $Root '1.2.3\magnitude-desktop-windows-x64-1.2.3.exe'
$next = Join-Path $Root '1.2.4\magnitude-desktop-windows-x64-1.2.4.exe'
Invoke-Installer $old 0
Assert-Version '1.2.3'
$shortcut = (New-Object -ComObject WScript.Shell).CreateShortcut((Join-Path ([Environment]::GetFolderPath('Programs')) 'Magnitude.lnk'))
if ($shortcut.WorkingDirectory -ne $installation) { throw 'Application shortcut has an invalid working directory' }
$unknown = Join-Path $installation 'unrelated user file.txt'
[IO.File]::WriteAllText($unknown, 'preserve this file')
Invoke-Installer $next 1
Assert-Version '1.2.3'
if ([IO.File]::ReadAllText($unknown) -ne 'preserve this file') { throw 'Unexpected file was changed' }
Remove-Item -LiteralPath $unknown
$run = 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Run'
$startup = '"' + (Join-Path $installation 'Magnitude.exe') + '" --background'
New-Item -Path $run -Force | Out-Null
New-ItemProperty -Path $run -Name 'dev.magnitude.desktop' -Value $startup -PropertyType String -Force | Out-Null
$state = Join-Path $Root 'update-state'
$prepared = Join-Path $state ('application-updates\prepared-' + [Guid]::NewGuid())
New-Item -ItemType Directory -Force $prepared | Out-Null
$helper = Join-Path $prepared 'magnitude-update.exe'
& bun (Join-Path $PSScriptRoot '..\..\..\version\scripts\generate-version.ts')
if ($LASTEXITCODE -ne 0) { throw 'Update helper build identity generation failed' }
& bun build (Join-Path $PSScriptRoot 'windows-update-handoff-entry.ts') --compile "--outfile=$helper"
if ($LASTEXITCODE -ne 0) { throw 'Update handoff bootstrap compilation failed' }
Copy-Item -LiteralPath $next -Destination (Join-Path $prepared 'magnitude-setup.exe')
$request = @{ stateDirectory=$state; preparedDirectory=$prepared; applicationPath=(Join-Path $installation 'Magnitude.exe'); version='1.2.4'; showWindow=$false; envelope=@{keyId='fixture';payload='';signature=''} }
$start = [Diagnostics.ProcessStartInfo]::new($helper)
$start.WorkingDirectory = $prepared
$start.UseShellExecute = $false
$start.CreateNoWindow = $true
$start.RedirectStandardInput = $true
$start.RedirectStandardOutput = $true
$process = [Diagnostics.Process]::Start($start)
try {
  $process.StandardInput.WriteLine(($request | ConvertTo-Json -Compress))
  $ready = $process.StandardOutput.ReadLineAsync()
  if (!$ready.Wait(10000) -or $ready.Result -ne 'ready') { throw 'Update helper did not acknowledge readiness' }
  if ($process.WaitForExit(100)) { throw 'Update helper exited before its owner' }
  Assert-Version '1.2.3'
  $process.StandardInput.Close()
  if (!$process.WaitForExit(60000)) { throw 'Update handoff did not finish installation' }
  if ($process.ExitCode -ne 0) { throw 'Update helper failed' }
  $result = Get-Content -Raw (Join-Path $state 'update-result.json') | ConvertFrom-Json
  if ($result.request.version -ne '1.2.4' -or $result.error) { throw 'Update helper did not record installer success' }
} finally {
  if (!$process.HasExited) { $process.Kill(); $process.WaitForExit() }
  $process.Dispose()
}
Assert-Version '1.2.4'
if ((Get-ItemProperty $run).'dev.magnitude.desktop' -ne $startup) { throw 'Update changed startup preference' }
Invoke-Installer (Join-Path $installation 'Uninstall Magnitude.exe') 0
$deadline = (Get-Date).AddSeconds(30)
while ((Test-Path $installation) -and (Get-Date) -lt $deadline) { Start-Sleep -Milliseconds 100 }
if ((Test-Path $installation) -or (Test-Path $registration)) { throw 'Uninstaller did not finish' }
if ((Get-ItemProperty $run -Name 'dev.magnitude.desktop' -ErrorAction SilentlyContinue)) { throw 'Uninstaller retained owned startup' }
Write-Output 'PASS actual NSIS fresh install, unknown-file refusal, owner-exit handoff, upgrade, startup preservation and uninstall'
$global:LASTEXITCODE = 0
