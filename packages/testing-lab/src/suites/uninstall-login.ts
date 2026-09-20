import { FileSystem } from "@effect/platform"
import { Effect, Option, Schema } from "effect"
import { basename, dirname, join } from "node:path"
import { AssertionFailure, Digest, InfrastructureFailure } from "../domain"
import { sha256 } from "../snapshot"
import { command } from "../process"
import { MacLoginRegistration } from "../login-registration"

const LoginFields = { path: Schema.String, executable: Schema.String, sha256: Digest }
const LinuxLoginEntry = Schema.TaggedStruct("Linux", LoginFields)
const WindowsLoginEntry = Schema.TaggedStruct("Windows", LoginFields)
const MacLoginEntry = Schema.TaggedStruct("MacOS", { path: Schema.String, registration: MacLoginRegistration })
export const RemovalLoginEntry = Schema.Union(LinuxLoginEntry, WindowsLoginEntry, MacLoginEntry)
export const RemovedLoginEntry = Schema.Struct({ path: Schema.String, state: Schema.Literal("Absent", "Dormant", "Unlaunchable") })
const readWindowsLogin = (path: string) => Effect.gen(function* () {
  const result = yield* command("powershell.exe", ["-NoProfile", "-NonInteractive", "-Command", String.raw`
$ErrorActionPreference = 'Stop'
$value = $null
if (Test-Path -LiteralPath $env:LAB_LOGIN_KEY) {
  $key = Get-Item -LiteralPath $env:LAB_LOGIN_KEY
  try {
    if ($key.GetValueNames() -contains 'dev.magnitude.desktop') {
      if ($key.GetValueKind('dev.magnitude.desktop') -ne [Microsoft.Win32.RegistryValueKind]::String) { throw 'Unexpected login command registry type' }
      $value = $key.GetValue('dev.magnitude.desktop', $null, [Microsoft.Win32.RegistryValueOptions]::DoNotExpandEnvironmentNames)
    }
  } finally { $key.Close() }
}
ConvertTo-Json -InputObject @{command=$value} -Compress
`], { env: { LAB_LOGIN_KEY: path }, timeoutMs: 10_000 })
  if (result.exitCode !== 0) return yield* new InfrastructureFailure({ operation: "uninstall-login", message: "Could not inspect Windows login registration" })
  return yield* Schema.decodeUnknown(Schema.parseJson(Schema.Struct({ command: Schema.NullOr(Schema.String) })))(result.stdout).pipe(
    Effect.map(value => value.command), Effect.mapError(() => new InfrastructureFailure({ operation: "uninstall-login", message: "Invalid Windows login registration response" })))
})
/** The UI must first enable this real installed-user entry; an initially absent entry cannot qualify cleanup. */
export const captureRemovalLogin = (environment: Readonly<Record<string, string>>, macOS: Option.Option<MacLoginRegistration> = Option.none()) => Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  if (process.platform === "darwin") {
    if (Option.isNone(macOS)) return yield* new InfrastructureFailure({ operation: "uninstall-login", message: "Native macOS login registration observation is missing" })
    const path = "/Applications/Magnitude.app"
    if (macOS.value.executable !== join(path, "Contents", "MacOS", "Magnitude") || (yield* fs.realPath(path)) !== path || !(yield* fs.exists(macOS.value.executable))) {
      return yield* new AssertionFailure({ message: "Native login registration does not belong to the installed application bundle" })
    }
    return MacLoginEntry.make({ path, registration: macOS.value })
  }
  if (process.platform === "win32") {
    if (!environment.LOCALAPPDATA) return yield* new InfrastructureFailure({ operation: "uninstall-login", message: "Qualified Windows user's LOCALAPPDATA is missing" })
    const path = "HKCU:\\Software\\Microsoft\\Windows\\CurrentVersion\\Run"
    const executable = join(environment.LOCALAPPDATA, "Programs", "Magnitude", "Magnitude.exe")
    const value = yield* readWindowsLogin(path)
    if (value !== `"${executable}" --background` || !(yield* fs.exists(executable))) return yield* new AssertionFailure({ message: "Enabled Windows login entry does not name the installed desktop command" })
    return WindowsLoginEntry.make({ path, executable, sha256: sha256(value) })
  }
  if (process.platform !== "linux" || !environment.HOME) return yield* new InfrastructureFailure({ operation: "uninstall-login", message: "Native login removal verification requires a qualified Linux desktop user" })
  const path = join(environment.XDG_CONFIG_HOME || join(environment.HOME, ".config"), "autostart/dev.magnitude.desktop")
  if ((yield* fs.realPath(path)) !== path || Number((yield* fs.stat(path)).size) > 16 * 1024) return yield* new AssertionFailure({ message: "Login entry is not an owned regular bounded file" })
  const wire = yield* fs.readFileString(path)
  const lines = wire.split(/\r?\n/)
  const executable = "/usr/bin/magnitude-desktop"
  for (const entry of ["Type=Application", "Hidden=false", `TryExec=${executable}`, `Exec=/usr/bin/env "${executable}" --background`]) {
    if (lines.filter(line => line === entry).length !== 1) return yield* new AssertionFailure({ message: "Enabled login entry does not guard the installed desktop executable" })
  }
  if (!(yield* fs.exists(executable))) return yield* new AssertionFailure({ message: "Enabled login entry has no installed executable" })
  return LinuxLoginEntry.make({ path, executable, sha256: sha256(wire) })
})

export const verifyRemovedLogin = (before: typeof RemovalLoginEntry.Type) => Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  if (before._tag === "MacOS") {
    // SMAppService owns its database. Removing the exact registered bundle prevents launch;
    // do not claim its opaque registration was deleted or mutate the user's saved preference.
    for (const path of [before.path, before.registration.executable]) {
      const entries = yield* fs.readDirectory(dirname(path)).pipe(Effect.catchTag("SystemError", error => error.reason === "NotFound" ? Effect.succeed([] as string[]) : Effect.fail(error)))
      if (entries.includes(basename(path))) return yield* new AssertionFailure({ message: "The registered macOS login application still exists after uninstall" })
    }
    return RemovedLoginEntry.make({ path: before.path, state: "Unlaunchable" })
  }
  if (before._tag === "Windows") {
    const value = yield* readWindowsLogin(before.path)
    if (value === null) return RemovedLoginEntry.make({ path: before.path, state: "Absent" })
    if (sha256(value) !== before.sha256) return yield* new AssertionFailure({ message: "Retained Windows login command changed during uninstall" })
    if (yield* fs.exists(before.executable)) return yield* new AssertionFailure({ message: "Uninstalled application still has a runnable Windows login entry" })
    return RemovedLoginEntry.make({ path: before.path, state: "Dormant" })
  }
  const entries = yield* fs.readDirectory(dirname(before.path)).pipe(Effect.catchTag("SystemError", error => error.reason === "NotFound" ? Effect.succeed([] as string[]) : Effect.fail(error)))
  if (!entries.includes(basename(before.path))) return RemovedLoginEntry.make({ path: before.path, state: "Absent" })
  if ((yield* fs.realPath(before.path)) !== before.path || sha256(yield* fs.readFileString(before.path)) !== before.sha256) {
    return yield* new AssertionFailure({ message: "Retained login preference changed during uninstall" })
  }
  if (yield* fs.exists(before.executable)) return yield* new AssertionFailure({ message: "Uninstalled application still has a runnable login entry" })
  return RemovedLoginEntry.make({ path: before.path, state: "Dormant" })
})
