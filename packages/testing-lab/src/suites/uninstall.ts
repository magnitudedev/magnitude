import { FileSystem } from "@effect/platform"
import { Effect, Schema } from "effect"
import { basename, dirname } from "node:path"
import { AssertionFailure, InfrastructureFailure } from "../domain"
import { InstalledApplication } from "../installer"
import { command } from "../process"

/** Inspect directory entries so dangling launcher symlinks cannot look successfully removed. */
export const verifyRemovedPayload = (application: Pick<InstalledApplication, "root" | "executable" | "cli">) => Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  for (const path of new Set([application.root, application.executable, application.cli])) {
    const entries = yield* fs.readDirectory(dirname(path)).pipe(Effect.catchTag("SystemError", error => error.reason === "NotFound" ? Effect.succeed([] as string[]) : Effect.fail(error)))
    if (entries.includes(basename(path))) return yield* new AssertionFailure({ message: `Package-owned path remains after uninstall: ${path}` })
  }
})

export const RemovalReceipt = Schema.Struct({ checks: Schema.Array(Schema.String) })
export const verifyNativeRemoval = (application: InstalledApplication, environment: Readonly<Record<string, string>>) => Effect.gen(function* () {
  yield* verifyRemovedPayload(application)
  const checks = ["Application payload and bundled CLI paths absent"]
  const format = application.candidate.target.packageFormat
  if (format === "deb" || format === "rpm") {
    const result = yield* command(format === "deb" ? "dpkg-query" : "rpm", format === "deb"
      ? ["-W", "-f=${binary:Package}\t${db:Status-Status}\n"] : ["-qa", "--qf", "%{NAME}\n"], { env: { ...environment, LC_ALL: "C" }, inheritEnv: false })
    if (result.exitCode !== 0) return yield* new InfrastructureFailure({ operation: "uninstall-registration", message: "Could not inspect the native package database" })
    const remains = result.stdout.split("\n").some(line => {
      if (format === "rpm") return line.trim() === "magnitude-desktop"
      const [name, status] = line.split("\t")
      return name?.split(":")[0] === "magnitude-desktop" && status?.trim() !== "not-installed" && status?.trim() !== "config-files"
    })
    if (remains) return yield* new AssertionFailure({ message: "Native package database still reports Magnitude installed or partially installed" })
    const desktop = "/usr/share/applications/magnitude-desktop.desktop"
    yield* verifyRemovedPayload({ root: desktop, executable: desktop, cli: desktop })
    checks.push("Native package no longer installed", "Desktop launcher absent")
  } else if (format === "exe") {
    const result = yield* command("powershell.exe", ["-NoProfile", "-NonInteractive", "-Command", String.raw`
$ErrorActionPreference = 'Stop'
$registered = Test-Path 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall\MagnitudeDesktop'
$shortcut = Test-Path -LiteralPath (Join-Path ([Environment]::GetFolderPath('Programs')) 'Magnitude.lnk')
$userPath = if (Test-Path 'HKCU:\Environment') { (Get-ItemProperty 'HKCU:\Environment' -ErrorAction Stop).Path } else { '' }
$cliPath = @($userPath -split ';' | Where-Object { $_.Trim().TrimEnd('\') -ieq $env:LAB_REMOVED_CLI_DIRECTORY.TrimEnd('\') }).Count -gt 0
@{ registered = [bool]$registered; shortcut = [bool]$shortcut; cliPath = [bool]$cliPath } | ConvertTo-Json -Compress
`], { env: { ...environment, LAB_REMOVED_CLI_DIRECTORY: application.root + "\\resources" }, inheritEnv: false })
    if (result.exitCode !== 0) return yield* new InfrastructureFailure({ operation: "uninstall-registration", message: "Could not inspect Windows application registration" })
    const registration = yield* Schema.decodeUnknown(Schema.parseJson(Schema.Struct({ registered: Schema.Boolean, shortcut: Schema.Boolean, cliPath: Schema.Boolean })))(result.stdout).pipe(
      Effect.mapError(() => new InfrastructureFailure({ operation: "uninstall-registration", message: "Invalid Windows registration inspection response" })))
    if (registration.registered || registration.shortcut || registration.cliPath) return yield* new AssertionFailure({ message: "Windows uninstall registration, Start menu shortcut or CLI PATH entry remains" })
    checks.push("Windows uninstall registration absent", "Start menu shortcut absent", "CLI user PATH registration absent")
  }
  return RemovalReceipt.make({ checks })
})
