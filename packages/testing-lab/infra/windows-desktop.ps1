$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
if (-not [Security.Principal.WindowsIdentity]::GetCurrent().IsSystem) { throw 'Desktop preparation requires SYSTEM' }
$state = 'C:\MagnitudeLab\State'
$runtime = Get-Content -LiteralPath (Join-Path $state 'runtime.json') -Raw | ConvertFrom-Json
$users = @(Get-LocalUser | Where-Object { $_.SID.Value -eq $runtime.userSid -and $_.Enabled })
if ($users.Count -ne 1) { throw 'Prepared runtime has no enabled admitted user' }
$user = $users[0]
if (Test-Path -LiteralPath (Join-Path $state 'desktop.json')) { throw 'Desktop preparation requires a fresh login' }
$worker = @'
$ErrorActionPreference='Stop'
$config=Get-Content -LiteralPath 'C:\MagnitudeLab\Runtime\launch.json' -Raw | ConvertFrom-Json
$env:PATH=$config.path
$env:CARGO_HOME=$config.cargo
$env:RUSTUP_HOME=$config.rustup
$env:LAB_TERMINAL_NODE_EXECUTABLE=$config.node
$env:LAB_DEPENDENCIES_EXECUTABLE=$config.dependencies
$env:LAB_PI_EXECUTABLE=$config.pi
$env:LAB_OPENCODE_EXECUTABLE=$config.opencode
$env:LAB_HERMES_EXECUTABLE=$config.hermes
Set-Location $config.workspace
& $config.bun packages/testing-lab/src/outward-worker.ts
exit $LASTEXITCODE
'@
$worker | Set-Content -LiteralPath 'C:\MagnitudeLab\Runtime\worker.ps1' -Encoding UTF8
$finish = @'
$ErrorActionPreference='Stop'
$state='C:\MagnitudeLab\State'
try {
  $path='HKLM:\SOFTWARE\Microsoft\Windows NT\CurrentVersion\Winlogon'
  foreach($name in @('DefaultPassword','AutoAdminLogon','AutoLogonCount')) { Remove-ItemProperty -Path $path -Name $name -ErrorAction SilentlyContinue }
  $login=Get-ItemProperty -Path $path
  foreach($name in @('DefaultPassword','AutoAdminLogon','AutoLogonCount')) { if($login.PSObject.Properties[$name]) { throw 'One-shot login state was not removed' } }
  $runtime=Get-Content -LiteralPath (Join-Path $state 'runtime.json') -Raw | ConvertFrom-Json
  $deadline=(Get-Date).AddSeconds(90)
  do {
    $desktop=@(Get-CimInstance Win32_Process -Filter "Name='explorer.exe'" | Where-Object {
      $_.SessionId -gt 0 -and (Invoke-CimMethod -InputObject $_ -MethodName GetOwnerSid).Sid -eq $runtime.userSid
    })
    if($desktop.Count){break}
    Start-Sleep -Milliseconds 250
  } while((Get-Date) -lt $deadline)
  if(-not $desktop.Count){throw 'Admitted user did not acquire a desktop session'}
  @{runtimeSha256=$runtime.runtimeSha256;userSid=$runtime.userSid;sessionId=$desktop[0].SessionId} | ConvertTo-Json -Compress | Set-Content -LiteralPath (Join-Path $state 'desktop.json') -Encoding UTF8
} catch {
  'Desktop preparation failed; inspect the admitted user session and credential cleanup' | Set-Content -LiteralPath (Join-Path $state 'desktop-error.txt') -Encoding UTF8
  throw
} finally { Unregister-ScheduledTask -TaskName 'MagnitudeLab-FinishDesktop' -Confirm:$false }
'@
$action=New-ScheduledTaskAction -Execute (Join-Path $env:SystemRoot 'System32\WindowsPowerShell\v1.0\powershell.exe') -Argument ('-NoProfile -NonInteractive -EncodedCommand '+[Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($finish)))
$trigger=New-ScheduledTaskTrigger -AtLogOn -User $user.Name
$principal=New-ScheduledTaskPrincipal -UserId 'SYSTEM' -LogonType ServiceAccount -RunLevel Highest
$settings=New-ScheduledTaskSettingsSet -ExecutionTimeLimit (New-TimeSpan -Minutes 3)
Register-ScheduledTask -TaskName 'MagnitudeLab-FinishDesktop' -Action $action -Trigger $trigger -Principal $principal -Settings $settings | Out-Null
$bytes=New-Object byte[] 32
$rng=[Security.Cryptography.RandomNumberGenerator]::Create()
try {$rng.GetBytes($bytes)} finally {$rng.Dispose()}
$password='Aa9!'+[Convert]::ToBase64String($bytes)
Set-LocalUser -Name $user.Name -Password (ConvertTo-SecureString $password -AsPlainText -Force) -PasswordNeverExpires $true
$path='HKLM:\SOFTWARE\Microsoft\Windows NT\CurrentVersion\Winlogon'
@{DefaultUserName=$user.Name;DefaultDomainName=$env:COMPUTERNAME;DefaultPassword=$password;AutoAdminLogon='1'}.GetEnumerator() | ForEach-Object { New-ItemProperty -Path $path -Name $_.Key -Value $_.Value -PropertyType String -Force | Out-Null }
New-ItemProperty -Path $path -Name AutoLogonCount -Value 1 -PropertyType DWord -Force | Out-Null
New-ItemProperty -Path 'HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Policies\System' -Name DisableCAD -Value 1 -PropertyType DWord -Force | Out-Null
$password=$null
# Reboot is an allocator-owned phase after this command has completed successfully.
Write-Output 'One-shot desktop login prepared; reboot and observed desktop readiness are required'
