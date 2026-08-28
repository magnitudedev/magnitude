import * as PlatformCommand from "@effect/platform/Command"
import * as FileSystem from "@effect/platform/FileSystem"
import {
  defaultDataDir,
  AcnHealthResponseSchema,
  resolveBinaryCommand,
  SDK_ACN_TARGET,
  SDK_VERSION,
  MAGNITUDE_SERVICE_ORIGIN,
} from "@magnitudedev/sdk"
import { Data, Effect, Option, Schedule, Schema } from "effect"
import { homedir } from "node:os"
import { dirname, resolve } from "node:path"
import { fileURLToPath } from "node:url"
import { stopTerminalAcn } from "../platform/terminal"
import { isDevelopmentBuild } from "../runtime/environment"
import { writeFileAtomic } from "../utils/atomic-file"

const SERVICE_LABEL = "dev.magnitude.acn"
const PUBLIC_HEALTH = `${MAGNITUDE_SERVICE_ORIGIN}/health`

export class ServerServiceError extends Data.TaggedError("ServerServiceError")<{
  readonly message: string
}> {}

const fail = (message: string) => new ServerServiceError({ message })

const run = (command: ReadonlyArray<string>, allowFailure = false) => {
  const [executable, ...args] = command
  if (executable === undefined) return Effect.fail(fail("Empty service command"))
  return PlatformCommand.make(executable, ...args).pipe(
    PlatformCommand.exitCode,
    Effect.flatMap((code) => code === 0 || allowFailure
      ? Effect.void
      : Effect.fail(fail(`${executable} exited with status ${code}`))),
    Effect.mapError((error) => error instanceof ServerServiceError
      ? error
      : fail(String(error))),
  )
}

const commandSucceeds = (command: ReadonlyArray<string>) => {
  const [executable, ...args] = command
  if (executable === undefined) return Effect.succeed(false)
  return PlatformCommand.make(executable, ...args).pipe(
    PlatformCommand.exitCode,
    Effect.map((code) => code === 0),
    Effect.orElseSucceed(() => false),
  )
}

const xml = (value: string) => value
  .replaceAll("&", "&amp;")
  .replaceAll("<", "&lt;")
  .replaceAll(">", "&gt;")
  .replaceAll('"', "&quot;")

const systemdQuote = (value: string) => `"${value
  .replaceAll("\\", "\\\\")
  .replaceAll('"', '\\"')
  .replaceAll("%", "%%")
  .replaceAll("\n", "\\n")
  .replaceAll("\r", "\\r")}"`

/** Quote one argv member for Windows' CommandLineToArgvW parsing. */
export const windowsCommandQuote = (value: string): string => {
  if (value.length > 0 && !/[\s"]/.test(value)) return value
  let quoted = '"'
  let backslashes = 0
  for (const character of value) {
    if (character === "\\") {
      backslashes += 1
      continue
    }
    if (character === '"') {
      quoted += "\\".repeat(backslashes * 2 + 1) + '"'
      backslashes = 0
      continue
    }
    quoted += "\\".repeat(backslashes) + character
    backslashes = 0
  }
  return quoted + "\\".repeat(backslashes * 2) + '"'
}

export const renderWindowsServerCommand = (command: ReadonlyArray<string>): string =>
  command.map(windowsCommandQuote).join(" ")

export const WINDOWS_RESTART_POLICY_SCRIPT = [
  "$task = Get-ScheduledTask -TaskName 'MagnitudeInference' -ErrorAction Stop",
  "$task.Settings.ExecutionTimeLimit = 'PT0S'",
  "$task.Settings.DisallowStartIfOnBatteries = $false",
  "$task.Settings.StopIfGoingOnBatteries = $false",
  "$task.Settings.RestartCount = 999",
  "$task.Settings.RestartInterval = 'PT1M'",
  "Set-ScheduledTask -InputObject $task | Out-Null",
].join("; ")

export const developmentServerCommand = (
  executable = process.execPath,
): ReadonlyArray<string> => [
  executable,
  resolve(dirname(fileURLToPath(import.meta.url)), "..", "..", "..", "packages", "acn", "src", "binary.ts"),
  "serve",
]

const resolveServiceCommand = isDevelopmentBuild()
  ? Effect.succeed(developmentServerCommand())
  : Effect.gen(function* () {
  const resolved = yield* resolveBinaryCommand({
    version: SDK_VERSION,
    acnRevision: SDK_ACN_TARGET.revision,
    dataDir: defaultDataDir(),
    acquisitionObserver: Option.none(),
  })
  return resolved.command
}).pipe(Effect.mapError((error) => fail(String(error))))

const writeServiceFile = (file: string, contents: string) =>
  writeFileAtomic(file, contents).pipe(Effect.mapError((error) => fail(String(error))))

const macServicePath = () => `${homedir()}/Library/LaunchAgents/${SERVICE_LABEL}.plist`
export const renderMacServerService = (command: ReadonlyArray<string>) => `<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>${SERVICE_LABEL}</string>
  <key>ProgramArguments</key><array>${command.map((part) => `<string>${xml(part)}</string>`).join("")}</array>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><dict><key>SuccessfulExit</key><false/></dict>
  <key>ThrottleInterval</key><integer>2</integer>
  <key>ProcessType</key><string>Background</string>
  <key>StandardOutPath</key><string>${xml(`${defaultDataDir()}/logs/acn-service.log`)}</string>
  <key>StandardErrorPath</key><string>${xml(`${defaultDataDir()}/logs/acn-service.log`)}</string>
</dict></plist>
`

const linuxServicePath = () => `${homedir()}/.config/systemd/user/magnitude.service`
export const renderLinuxServerService = (command: ReadonlyArray<string>) => `[Unit]
Description=Magnitude local inference service
After=network.target

[Service]
Type=simple
ExecStart=${command.map(systemdQuote).join(" ")}
Restart=on-failure
RestartSec=2

[Install]
WantedBy=default.target
`

const installAndStartService = (command: ReadonlyArray<string>) => Effect.gen(function* () {
  if (process.platform === "darwin") {
    const service = macServicePath()
    yield* writeServiceFile(service, renderMacServerService(command))
    const domain = `gui/${process.getuid?.() ?? 0}`
    yield* run(["launchctl", "bootout", domain, service], true)
    yield* run(["launchctl", "enable", `${domain}/${SERVICE_LABEL}`])
    yield* run(["launchctl", "bootstrap", domain, service])
    return
  }
  if (process.platform === "linux") {
    yield* writeServiceFile(linuxServicePath(), renderLinuxServerService(command))
    yield* run(["systemctl", "--user", "daemon-reload"])
    yield* run(["systemctl", "--user", "enable", "--now", "magnitude.service"])
    return
  }
  if (process.platform === "win32") {
    const taskCommand = renderWindowsServerCommand(command)
    yield* run([
      "schtasks", "/Create", "/TN", "MagnitudeInference", "/TR", taskCommand,
      "/SC", "ONLOGON", "/RL", "LIMITED", "/F",
    ])
    yield* run([
      "powershell.exe", "-NoProfile", "-NonInteractive", "-Command",
      WINDOWS_RESTART_POLICY_SCRIPT,
    ])
    yield* run(["schtasks", "/Change", "/TN", "MagnitudeInference", "/ENABLE"])
    yield* run(["schtasks", "/Run", "/TN", "MagnitudeInference"])
    return
  }
  return yield* fail(`Unsupported platform: ${process.platform}`)
})

/** Register the exact release for future user-session startup without
 * replacing the daemon currently serving an interactive setup flow. */
export const installServerOnStartup = Effect.gen(function* () {
  const command = yield* resolveServiceCommand
  const fs = yield* FileSystem.FileSystem
  yield* fs.makeDirectory(`${defaultDataDir()}/logs`, { recursive: true, mode: 0o700 })
  if (process.platform === "darwin") {
    yield* writeServiceFile(macServicePath(), renderMacServerService(command))
    const domain = `gui/${process.getuid?.() ?? 0}`
    yield* run(["launchctl", "enable", `${domain}/${SERVICE_LABEL}`])
    return
  }
  if (process.platform === "linux") {
    yield* writeServiceFile(linuxServicePath(), renderLinuxServerService(command))
    yield* run(["systemctl", "--user", "daemon-reload"])
    yield* run(["systemctl", "--user", "enable", "magnitude.service"])
    return
  }
  if (process.platform === "win32") {
    const taskCommand = renderWindowsServerCommand(command)
    yield* run([
      "schtasks", "/Create", "/TN", "MagnitudeInference", "/TR", taskCommand,
      "/SC", "ONLOGON", "/RL", "LIMITED", "/F",
    ])
    yield* run(["schtasks", "/Change", "/TN", "MagnitudeInference", "/ENABLE"])
    return
  }
  return yield* fail(`Unsupported platform: ${process.platform}`)
}).pipe(Effect.mapError((error) => error instanceof ServerServiceError
  ? error
  : fail(String(error))))

export const stopServer = Effect.gen(function* () {
  if (process.platform === "darwin") {
    const domain = `gui/${process.getuid?.() ?? 0}`
    yield* run(["launchctl", "disable", `${domain}/${SERVICE_LABEL}`], true)
    yield* run(["launchctl", "bootout", domain, macServicePath()], true)
  } else if (process.platform === "linux") {
    yield* run(["systemctl", "--user", "disable", "--now", "magnitude.service"], true)
  } else if (process.platform === "win32") {
    yield* run(["schtasks", "/End", "/TN", "MagnitudeInference"], true)
    yield* run(["schtasks", "/Change", "/TN", "MagnitudeInference", "/DISABLE"], true)
  }
  yield* stopTerminalAcn.pipe(Effect.ignore)
})

const probeReady = Effect.tryPromise({
  try: (signal) => fetch(PUBLIC_HEALTH, { signal }).then(async (response) => {
    if (!response.ok) throw new Error(`health returned ${response.status}`)
    const health = Schema.decodeUnknownSync(AcnHealthResponseSchema)(await response.json())
    if (health.version !== SDK_ACN_TARGET.identity
      || health.revision !== SDK_ACN_TARGET.revision
      || health.state._tag !== "Ready") {
      throw new Error("the fixed port is not serving the requested Magnitude release")
    }
  }),
  catch: (error) => fail(String(error)),
})

const awaitReady = probeReady.pipe(
  Effect.retry(Schedule.spaced("250 millis").pipe(Schedule.intersect(Schedule.recurs(240)))),
)

const managedServiceIsCurrent = (command: ReadonlyArray<string>) => Effect.gen(function* () {
  if (Option.isNone(yield* Effect.option(probeReady))) return false
  const fs = yield* FileSystem.FileSystem
  if (process.platform === "darwin") {
    const service = macServicePath()
    const source = yield* fs.readFileString(service).pipe(Effect.option)
    if (Option.isNone(source) || source.value !== renderMacServerService(command)) return false
    const domain = `gui/${process.getuid?.() ?? 0}`
    return yield* commandSucceeds(["launchctl", "print", `${domain}/${SERVICE_LABEL}`])
  }
  if (process.platform === "linux") {
    const source = yield* fs.readFileString(linuxServicePath()).pipe(Effect.option)
    if (Option.isNone(source) || source.value !== renderLinuxServerService(command)) return false
    return (yield* commandSucceeds(["systemctl", "--user", "is-enabled", "--quiet", "magnitude.service"]))
      && (yield* commandSucceeds(["systemctl", "--user", "is-active", "--quiet", "magnitude.service"]))
  }
  // Scheduled-task inspection is not uniform across supported Windows releases;
  // recreating the one named task remains the authoritative idempotent path.
  return false
})

export const startServer = Effect.gen(function* () {
  const command = yield* resolveServiceCommand
  // Connecting another harness must not bounce a healthy service whose
  // persisted definition already names this exact release.
  if (yield* managedServiceIsCurrent(command)) return
  const fs = yield* FileSystem.FileSystem
  yield* fs.makeDirectory(`${defaultDataDir()}/logs`, { recursive: true, mode: 0o700 })
  // A JIT-owned process may already own the fixed port. Retire it before the
  // platform manager starts the release-matched persistent service.
  yield* stopTerminalAcn.pipe(Effect.ignore)
  yield* installAndStartService(command).pipe(
    Effect.zipRight(awaitReady),
    Effect.onError(() => stopServer.pipe(Effect.ignore)),
  )
}).pipe(Effect.mapError((error) => error instanceof ServerServiceError
  ? error
  : fail(String(error))))
