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
Invoke-Installer $next 0
Assert-Version '1.2.4'
if ((Get-ItemProperty $run).'dev.magnitude.desktop' -ne $startup) { throw 'Update changed startup preference' }
Invoke-Installer (Join-Path $installation 'Uninstall Magnitude.exe') 0
$deadline = (Get-Date).AddSeconds(30)
while ((Test-Path $installation) -and (Get-Date) -lt $deadline) { Start-Sleep -Milliseconds 100 }
if ((Test-Path $installation) -or (Test-Path $registration)) { throw 'Uninstaller did not finish' }
if ((Get-ItemProperty $run -Name 'dev.magnitude.desktop' -ErrorAction SilentlyContinue)) { throw 'Uninstaller retained owned startup' }
Write-Output 'PASS actual NSIS fresh install, unknown-file refusal, upgrade, startup preservation and uninstall'
$global:LASTEXITCODE = 0
