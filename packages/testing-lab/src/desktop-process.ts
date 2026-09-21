import { FileSystem } from "@effect/platform"
import { Effect, Option, Schema } from "effect"
import { win32 } from "node:path"
import { LabProcessId } from "./application-identity"
import { AssertionFailure } from "./domain"
import { checkedCommand } from "./process"

const Birth = Schema.String.pipe(Schema.pattern(/^[1-9][0-9]*$/), Schema.brand("DesktopProcessBirth"))
const UserSid = Schema.String.pipe(Schema.pattern(/^S-1-[0-9-]+$/), Schema.brand("DesktopProcessUserSid"))
export const DesktopProcessRequest = Schema.Struct({ launcherPid: LabProcessId, applicationPid: LabProcessId, executable: Schema.NonEmptyString })
export const WindowsDesktopProcesses = Schema.Struct({ userSid: UserSid,
  launcher: Schema.Struct({ pid: LabProcessId, birth: Birth, userSid: UserSid }),
  application: Schema.Struct({ pid: LabProcessId, parentPid: LabProcessId, birth: Birth, userSid: UserSid, executable: Schema.NonEmptyString }),
})
const fail = (message: string) => new AssertionFailure({ message: `Desktop process ownership: ${message}` })

/** Playwright's Windows child is cmd.exe; the inspected Electron process must be its own child. */
export const verifyWindowsDesktopProcesses = (request: typeof DesktopProcessRequest.Type, observed: typeof WindowsDesktopProcesses.Type) => Effect.gen(function* () {
  const { application, launcher } = observed
  if (launcher.pid !== request.launcherPid || application.pid !== request.applicationPid || application.parentPid !== launcher.pid
    || application.pid === launcher.pid) return yield* fail("Electron does not belong to the launched shell")
  if (application.userSid !== observed.userSid || launcher.userSid !== observed.userSid) return yield* fail("launcher or Electron belongs to another user")
  if (BigInt(application.birth) < BigInt(launcher.birth)) return yield* fail("Electron predates its launcher")
  if (!win32.isAbsolute(application.executable) || win32.normalize(application.executable).toLowerCase() !== win32.normalize(request.executable).toLowerCase()) {
    return yield* fail("Electron is not the admitted installed executable")
  }
  return application.pid
})

export const windowsDesktopProcessScript = String.raw`
$ErrorActionPreference = 'Stop'
$request = [Console]::In.ReadToEnd() | ConvertFrom-Json
function Read-OwnedProcess([int]$id) {
  $item = Get-CimInstance Win32_Process -Filter "ProcessId = $id"
  if ($null -eq $item -or $null -eq $item.CreationDate -or -not $item.ExecutablePath) { throw 'Missing live native process identity' }
  $owner = Invoke-CimMethod -InputObject $item -MethodName GetOwnerSid
  if ($owner.ReturnValue -ne 0) { throw 'Cannot inspect native process user' }
  return @{pid=[int]$item.ProcessId;parentPid=[int]$item.ParentProcessId;birth=$item.CreationDate.ToUniversalTime().Ticks.ToString();userSid=$owner.Sid;executable=$item.ExecutablePath}
}
$launcher = Read-OwnedProcess ([int]$request.launcherPid)
$application = Read-OwnedProcess ([int]$request.applicationPid)
foreach ($expected in @($launcher,$application)) {
  $current = Read-OwnedProcess $expected.pid
  if ($current.birth -ne $expected.birth) { throw 'Native process changed during inspection' }
}
@{userSid=[Security.Principal.WindowsIdentity]::GetCurrent().User.Value;launcher=$launcher;application=$application} | ConvertTo-Json -Depth 4 -Compress
`

export const verifyLaunchedDesktop = (request: typeof DesktopProcessRequest.Type) => Effect.gen(function* () {
  if (process.platform !== "win32") {
    if (request.applicationPid !== request.launcherPid) return yield* fail("Electron differs from the launched process")
    return request.applicationPid
  }
  const fs = yield* FileSystem.FileSystem
  const executable = yield* fs.realPath(request.executable).pipe(Effect.mapError(() => fail("Cannot resolve the installed executable")))
  const observed = yield* checkedCommand("powershell.exe", ["-NoProfile", "-NonInteractive", "-Command", windowsDesktopProcessScript], {
    stdin: Option.some(yield* Schema.encode(Schema.parseJson(DesktopProcessRequest))({ ...request, executable })), timeoutMs: 15_000, maxOutputBytes: 16_384,
  }).pipe(Effect.flatMap(result => Schema.decodeUnknown(Schema.parseJson(WindowsDesktopProcesses))(result.stdout)),
    Effect.mapError(() => fail("Cannot inspect the native Electron process")))
  return yield* verifyWindowsDesktopProcesses({ ...request, executable }, observed)
}).pipe(Effect.mapError(error => error._tag === "AssertionFailure" ? error : fail("Invalid native desktop process identity")))
