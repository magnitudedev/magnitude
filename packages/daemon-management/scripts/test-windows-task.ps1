param(
  [Parameter(Mandatory=$true)][string]$Addon,
  [Parameter(Mandatory=$true)][string]$Helper,
  [Parameter(Mandatory=$true)][string]$FixtureRoot
)
$ErrorActionPreference = 'Stop'
$scheduler = New-Object -ComObject 'Schedule.Service'
$scheduler.Connect()
$folder = $scheduler.GetFolder('\')
function Get-FixtureTask {
  try { return $folder.GetTask('MagnitudeInference') }
  catch {
    $failure = $_.Exception
    while ($null -ne $failure.InnerException) { $failure = $failure.InnerException }
    if ($failure.HResult -eq -2147024894) { return $null }
    throw
  }
}
if ($null -ne (Get-FixtureTask)) { throw 'Refusing to replace an existing MagnitudeInference task.' }
$sid = [Security.Principal.WindowsIdentity]::GetCurrent().User.Value
$service = Join-Path $FixtureRoot 'magnitude-service.exe'
$inputPath = Join-Path $FixtureRoot ('task-input-' + [Guid]::NewGuid() + '.txt')
[IO.File]::WriteAllText($inputPath, '')
$capturedXml = $null
try {
  # Reproduce the historical CLI's task defaults, rather than the different defaults
  # of New-ScheduledTaskSettingsSet. Omit /F so an existing task cannot be replaced.
  # Redirect input to prevent an unexpected overwrite/password prompt from hanging CI.
  $create = Start-Process -FilePath "$env:SystemRoot\System32\schtasks.exe" -ArgumentList @(
    '/Create', '/TN', 'MagnitudeInference', '/TR', ('"\"' + $service + '\" serve"'),
    '/SC', 'ONLOGON', '/RL', 'LIMITED'
  ) -NoNewWindow -Wait -PassThru -RedirectStandardInput $inputPath
  if ($create.ExitCode -ne 0) { throw 'Could not create the historical task fixture.' }
  $capturedXml = (Get-FixtureTask).Xml
  $task = Get-ScheduledTask -TaskName 'MagnitudeInference' -ErrorAction Stop
  $task.Settings.ExecutionTimeLimit = 'PT0S'
  $task.Settings.DisallowStartIfOnBatteries = $false
  $task.Settings.StopIfGoingOnBatteries = $false
  $task.Settings.RestartCount = 999
  $task.Settings.RestartInterval = 'PT1M'
  $task.Settings.Enabled = $false
  Set-ScheduledTask -InputObject $task | Out-Null
  $capturedXml = (Get-FixtureTask).Xml
  $env:MAGNITUDE_TASK_FIXTURE_ADDON = $Addon
  $env:MAGNITUDE_TASK_FIXTURE_HELPER = $Helper
  $env:MAGNITUDE_TASK_FIXTURE_SERVICE = $service
  Push-Location (Split-Path -Parent $PSScriptRoot)
  try {
    & bunx --bun vitest run src/desktop-native/windows-task-native.test.ts
    if ($LASTEXITCODE -ne 0) { throw 'Native Task Scheduler acceptance failed.' }
  } finally { Pop-Location }
  if ($null -ne (Get-FixtureTask)) { throw 'Fixture task survived successful retirement.' }
} finally {
  Remove-Item $inputPath -ErrorAction SilentlyContinue
  Remove-Item Env:MAGNITUDE_TASK_FIXTURE_ADDON, Env:MAGNITUDE_TASK_FIXTURE_HELPER, Env:MAGNITUDE_TASK_FIXTURE_SERVICE -ErrorAction SilentlyContinue
  if ($null -ne $capturedXml) {
    $remaining = Get-FixtureTask
    if ($null -ne $remaining) {
      if ($remaining.Xml -cne $capturedXml) { throw 'Fixture task changed; refusing cleanup of an unknown registration.' }
      $folder.DeleteTask('MagnitudeInference', 0)
    }
  }
}
