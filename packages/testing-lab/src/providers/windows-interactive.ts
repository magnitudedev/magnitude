import { Effect, Schema } from "effect"
import { win32 } from "node:path"
import { InfrastructureFailure } from "../domain"

const Launch = Schema.Struct({ user: Schema.String, executable: Schema.String, args: Schema.Array(Schema.String),
  root: Schema.String, origin: Schema.String, timeoutSeconds: Schema.Int.pipe(Schema.positive()) })

/** The system-side command never tests the app itself. It observes a task in the admitted user's desktop. */
export const windowsInteractiveScript = (input: typeof Launch.Type) => Effect.gen(function* () {
  if (![input.executable, input.root].every(path => /^[a-z]:\\/i.test(path) && win32.isAbsolute(path)) ||
    !input.executable.toLowerCase().endsWith(".exe") || !/^[a-z][a-z0-9]{1,19}$/.test(input.user) ||
    [input.executable, input.root, input.origin, ...input.args].some(value => value.includes("\0"))) {
    return yield* new InfrastructureFailure({ operation: "azure-bootstrap", message: "Windows worker requires local absolute paths, a native executable and valid launch arguments" })
  }
  const encoded = Buffer.from(yield* Schema.encode(Schema.parseJson(Launch))(input).pipe(Effect.mapError(() =>
    new InfrastructureFailure({ operation: "azure-bootstrap", message: "Invalid Windows desktop launch configuration" })))).toString("base64")
  return String.raw`param([Parameter(Mandatory=$true)][string]$LAB_WORKER_TOKEN)
$ErrorActionPreference = 'Stop'
$configuration = [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('${encoded}')) | ConvertFrom-Json
$taskName = 'MagnitudeLab-' + [Guid]::NewGuid().ToString('N')
$directory = Join-Path $env:ProgramData $taskName
$registered = $false
$exitCode = 1
try {
  if (-not [Security.Principal.WindowsIdentity]::GetCurrent().IsSystem) { throw 'Worker delivery requires the system account' }
  $account = Get-LocalUser -Name $configuration.user -ErrorAction Stop
  if (-not $account.Enabled) { throw 'Worker account is disabled' }
  New-Item -ItemType Directory -Path $directory -ErrorAction Stop | Out-Null
  # Candidate code can read its run authority, but cannot replace the system-owned launcher.
  $acl = New-Object Security.AccessControl.DirectorySecurity
  $acl.SetAccessRuleProtection($true, $false)
  $acl.SetOwner([Security.Principal.SecurityIdentifier]::new('S-1-5-18'))
  $inherit = [Security.AccessControl.InheritanceFlags]'ContainerInherit,ObjectInherit'
  $propagate = [Security.AccessControl.PropagationFlags]::None
  $allow = [Security.AccessControl.AccessControlType]::Allow
  $acl.AddAccessRule([Security.AccessControl.FileSystemAccessRule]::new([Security.Principal.SecurityIdentifier]::new('S-1-5-18'), 'FullControl', $inherit, $propagate, $allow))
  $acl.AddAccessRule([Security.AccessControl.FileSystemAccessRule]::new($account.SID, 'ReadAndExecute', $inherit, $propagate, $allow))
  Set-Acl -LiteralPath $directory -AclObject $acl
  $configuration | Add-Member -NotePropertyName token -NotePropertyValue $LAB_WORKER_TOKEN
  $configuration | Add-Member -NotePropertyName sid -NotePropertyValue $account.SID.Value
  $configuration | ConvertTo-Json -Depth 8 -Compress | Set-Content -LiteralPath (Join-Path $directory 'launch.json') -Encoding UTF8
  $wrapper = @'
$ErrorActionPreference = 'Stop'
try {
  $config = Get-Content -LiteralPath (Join-Path $PSScriptRoot 'launch.json') -Raw | ConvertFrom-Json
  $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
  if ($identity.User.Value -ne $config.sid -or [Diagnostics.Process]::GetCurrentProcess().SessionId -eq 0 -or -not [Environment]::UserInteractive) {
    throw 'Worker did not enter the admitted interactive desktop session'
  }
  $env:LAB_WORKER_TOKEN = $config.token
  $env:LAB_WORKER_ROOT = $config.root
  $env:LAB_URL = $config.origin
  New-Item -ItemType Directory -Path $config.root -Force | Out-Null
  function Quote-NativeArgument([string]$Value) {
    $result = [Text.StringBuilder]::new('"')
    $slashes = 0
    foreach ($character in $Value.ToCharArray()) {
      if ($character -eq '\') { $slashes++; continue }
      if ($character -eq '"') { [void]$result.Append('\', 2 * $slashes + 1) }
      else { [void]$result.Append('\', $slashes) }
      [void]$result.Append($character)
      $slashes = 0
    }
    [void]$result.Append('\', 2 * $slashes)
    [void]$result.Append('"')
    return $result.ToString()
  }
  $arguments = @($config.args | ForEach-Object { Quote-NativeArgument $_ }) -join ' '
  $start = @{ FilePath=$config.executable; WorkingDirectory=$config.root; NoNewWindow=$true; PassThru=$true; Wait=$true;
    RedirectStandardOutput=(Join-Path $config.root 'bootstrap.stdout.log'); RedirectStandardError=(Join-Path $config.root 'bootstrap.stderr.log') }
  if ($arguments.Length) { $start.ArgumentList = $arguments }
  $child = Start-Process @start
  if (-not $child.HasExited -or $null -eq $child.ExitCode) { throw 'Worker exit status is unavailable' }
  exit $child.ExitCode
} catch {
  if ($config -and $config.root) {
    New-Item -ItemType Directory -Path $config.root -Force | Out-Null
    $detail = $_.Exception.Message.Replace($config.token, '[REDACTED]')
    $detail.Substring(0, [Math]::Min(1800, $detail.Length)) | Set-Content -LiteralPath (Join-Path $config.root 'bootstrap.launch-error.log') -Encoding UTF8
  }
  exit 1
}
'@
  $script = Join-Path $directory 'worker.ps1'
  $wrapper | Set-Content -LiteralPath $script -Encoding UTF8
  $powerShell = Join-Path $env:SystemRoot 'System32\WindowsPowerShell\v1.0\powershell.exe'
  $action = New-ScheduledTaskAction -Execute $powerShell -Argument ('-NoProfile -NonInteractive -ExecutionPolicy Bypass -File "' + $script + '"')
  $principal = New-ScheduledTaskPrincipal -UserId $account.SID.Value -LogonType Interactive -RunLevel Highest
  $settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit (New-TimeSpan -Seconds $configuration.timeoutSeconds) -MultipleInstances IgnoreNew
  Register-ScheduledTask -TaskName $taskName -Action $action -Principal $principal -Settings $settings | Out-Null
  $registered = $true
  $started = Get-Date
  $deadline = $started.AddSeconds($configuration.timeoutSeconds)
  Start-ScheduledTask -TaskName $taskName
  for (;;) {
    $task = Get-ScheduledTask -TaskName $taskName
    $info = Get-ScheduledTaskInfo -TaskName $taskName
    $ran = $info.LastRunTime -ge $started.AddSeconds(-2)
    if ($ran -and $task.State -ne 'Running' -and $task.State -ne 'Queued') {
      if ($info.LastTaskResult -eq 0) { $exitCode = 0 }
      else { Write-Output ('Interactive worker exit: ' + $info.LastTaskResult) }
      break
    }
    if ((Get-Date) -ge $deadline) { throw 'Interactive worker exceeded its deadline' }
    if (-not $ran -and (Get-Date) -ge $started.AddMinutes(2)) { throw 'No interactive desktop session accepted the worker task' }
    Start-Sleep -Seconds 2
  }
  foreach ($name in @('bootstrap.stdout.log', 'bootstrap.stderr.log', 'bootstrap.launch-error.log')) {
    $log = Join-Path $configuration.root $name
    if (Test-Path -LiteralPath $log) {
      $text = (Get-Content -LiteralPath $log -Tail 80 | Out-String).Replace($LAB_WORKER_TOKEN, '[REDACTED]')
      Write-Output $text.Substring([Math]::Max(0, $text.Length - 12000))
    }
  }
} catch {
  $detail = $_.Exception.Message.Replace($LAB_WORKER_TOKEN, '[REDACTED]')
  Write-Error -ErrorAction Continue ('Windows desktop worker delivery failed: ' + $detail.Substring(0, [Math]::Min(1800, $detail.Length)))
}
finally {
  if ($registered) {
    try {
      Stop-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
      Unregister-ScheduledTask -TaskName $taskName -Confirm:$false -ErrorAction Stop
    } finally {
      if (Test-Path -LiteralPath $directory) { Remove-Item -LiteralPath $directory -Recurse -Force -ErrorAction Stop }
    }
  } elseif (Test-Path -LiteralPath $directory) {
    Remove-Item -LiteralPath $directory -Recurse -Force -ErrorAction Stop
  }
}
exit $exitCode
`
})
