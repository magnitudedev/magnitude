import { FileSystem } from "@effect/platform"
import { Effect, Schema } from "effect"
import { fileURLToPath } from "node:url"
import { InitializationDownload } from "./linux-initialization"

// Preparation verifies the exact client or Server distribution before candidate delivery.
export const WindowsPreparationDistribution = Schema.Union(
  Schema.Struct({ os: Schema.Literal("windows"), version: Schema.Literal("10", "11") }),
  Schema.Struct({ os: Schema.Literal("windows-server"), version: Schema.Literal("2022", "2025") }),
)
export const WindowsRuntimePreparation = Schema.Struct({
  distribution: WindowsPreparationDistribution,
  adminUsername: Schema.String.pipe(Schema.pattern(/^[a-z][a-z0-9]{1,19}$/)),
  architecture: Schema.Literal("x64"), runtime: InitializationDownload,
})

/** Configuration is delivered separately through a protected native command parameter. */
export const windowsRuntimePreparation = Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  return renderWindowsRuntimePreparation(yield* fs.readFileString(fileURLToPath(new URL("../../infra/windows-runtime.ps1", import.meta.url))))
})

export const renderWindowsRuntimePreparation = (setup: string) => String.raw`param([Parameter(Mandatory=$true)][string]$LAB_INITIALIZATION)
$ErrorActionPreference = 'Stop'
if (-not [Security.Principal.WindowsIdentity]::GetCurrent().IsSystem) { throw 'Runtime delivery requires SYSTEM' }
$tools = Get-Content -LiteralPath 'C:\MagnitudeLab\Tools\paths.json' -Raw | ConvertFrom-Json
$script = 'C:\MagnitudeLab\Preparation\runtime.ps1'
[IO.File]::WriteAllBytes($script,[Convert]::FromBase64String('${Buffer.from(setup).toString("base64")}'))
try {
  $env:LAB_INITIALIZATION = $LAB_INITIALIZATION
  & $tools.powershell -NoProfile -NonInteractive -File $script
  if ($LASTEXITCODE -ne 0) { throw "Runtime preparation exited $LASTEXITCODE" }
} finally { Remove-Item Env:LAB_INITIALIZATION -ErrorAction SilentlyContinue }
`

export const windowsDesktopPreparation = Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  return yield* fs.readFileString(fileURLToPath(new URL("../../infra/windows-desktop.ps1", import.meta.url)))
})

const DesktopExpectation = Schema.Struct({ sha256: InitializationDownload.fields.sha256, user: WindowsRuntimePreparation.fields.adminUsername })
export const windowsDesktopReadiness = (runtimeSha256: string, adminUsername: string) => Effect.gen(function* () {
  const expected = yield* Schema.encode(Schema.parseJson(DesktopExpectation))(
    yield* Schema.decodeUnknown(DesktopExpectation)({ sha256: runtimeSha256, user: adminUsername }),
  )
  return String.raw`$ErrorActionPreference='Stop'
$expected=[Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('${Buffer.from(expected).toString("base64")}')) | ConvertFrom-Json
$root='C:\MagnitudeLab\State'
$deadline=(Get-Date).AddSeconds(120)
do {
  if(Test-Path -LiteralPath (Join-Path $root 'desktop-error.txt')){throw 'Desktop preparation failed'}
  $task=Get-ScheduledTask -TaskName 'MagnitudeLab-FinishDesktop' -ErrorAction SilentlyContinue
  if((Test-Path -LiteralPath (Join-Path $root 'desktop.json')) -and -not $task){break}
  Start-Sleep -Milliseconds 250
} while((Get-Date) -lt $deadline)
if($task){throw 'One-shot desktop task did not complete'}
$runtime=Get-Content -LiteralPath (Join-Path $root 'runtime.json') -Raw | ConvertFrom-Json
$desktop=Get-Content -LiteralPath (Join-Path $root 'desktop.json') -Raw | ConvertFrom-Json
$account=Get-LocalUser -Name $expected.user
if(-not $account.Enabled -or $runtime.runtimeSha256 -ne $expected.sha256 -or $desktop.runtimeSha256 -ne $expected.sha256 -or $desktop.userSid -ne $account.SID.Value -or $runtime.userSid -ne $account.SID.Value -or $desktop.sessionId -le 0){throw 'Desktop receipt differs from the admitted runtime and user'}
$login=Get-ItemProperty -Path 'HKLM:\SOFTWARE\Microsoft\Windows NT\CurrentVersion\Winlogon'
foreach($name in @('DefaultPassword','AutoAdminLogon','AutoLogonCount')){if($login.PSObject.Properties[$name]){throw 'One-shot login state remains'}}
$processes=@(Get-CimInstance Win32_Process -Filter "Name='explorer.exe'" | Where-Object { $_.SessionId -eq $desktop.sessionId -and (Invoke-CimMethod -InputObject $_ -MethodName GetOwnerSid).Sid -eq $account.SID.Value })
if(-not $processes.Count){throw 'Prepared desktop session is no longer active'}
Write-Output 'Verified runtime, admitted interactive desktop and removal of one-shot login authority'
`
})
