import { createHash } from "node:crypto"
import { lstat, readFile, unlink } from "node:fs/promises"
import { isAbsolute, join } from "node:path"
import { Effect, Option, Schema } from "effect"
import { LegacyStartupCommands, LegacyStartupFailed } from "./legacy-startup-command"

const Pid = Schema.Int.pipe(Schema.between(1, Number.MAX_SAFE_INTEGER))
export const LegacyLinuxStartup = Schema.TaggedStruct("SystemdUserService", {
  unit: Schema.NonEmptyString, path: Schema.NonEmptyString,
  digest: Schema.String.pipe(Schema.pattern(/^[a-f0-9]{64}$/), Schema.brand("LegacyLinuxStartupDigest")),
  enabled: Schema.Boolean,
  runningPid: Schema.optionalWith(Pid, { as: "Option", exact: true }),
})
export type LegacyLinuxStartup = typeof LegacyLinuxStartup.Type
const UnitState = Schema.Struct({
  LoadState: Schema.String, ActiveState: Schema.String, FragmentPath: Schema.String,
  UnitFileState: Schema.String, MainPID: Schema.NumberFromString.pipe(Schema.int(), Schema.nonNegative()),
  DropInPaths: Schema.String, Transient: Schema.Literal("no"), NeedDaemonReload: Schema.Literal("no"),
})
const missing = (error: unknown) => error instanceof Error && "code" in error && error.code === "ENOENT"
const failed = (message: string) => new LegacyStartupFailed({ message })

/** Recognize the historical generated unit, not arbitrary user service commands or hooks. */
const verifySource = (source: string) => Effect.gen(function* () {
  const fields = new Map<string, string>()
  let section = ""
  for (const raw of source.split(/\r?\n/)) {
    const line = raw.trim()
    if (!line || line.startsWith("#") || line.startsWith(";")) continue
    if (/^\[[^\]]+\]$/.test(line)) { section = line.slice(1, -1); continue }
    const index = line.indexOf("=")
    const key = `${section}.${line.slice(0, index)}`
    if (index < 1 || fields.has(key)) return yield* failed("Legacy user service contains malformed or duplicate settings")
    fields.set(key, line.slice(index + 1))
  }
  const fixed = new Map([
    ["Unit.Description", "Magnitude local inference service"], ["Unit.After", "network.target"],
    ["Service.Type", "simple"], ["Service.Restart", "on-failure"], ["Service.RestartSec", "2"],
    ["Install.WantedBy", "default.target"],
  ])
  if (fields.size !== fixed.size + 1 || [...fixed].some(([key, value]) => fields.get(key) !== value) ||
      !/^"(?:\\.|[^"\\])*\/magnitude-service" "serve"(?: "(?:\\.|[^"\\])*")*$/.test(fields.get("Service.ExecStart") ?? "")) {
    return yield* failed("Legacy user service no longer matches the generated Magnitude service")
  }
})

/** The old CLI always wrote ~/.config/systemd/user, independently of XDG_CONFIG_HOME. */
export const makeLinuxLegacyStartup = (options: {
  readonly home: string
  readonly runtimeDirectory?: string
  readonly unit?: string
}) => Effect.gen(function* () {
  const commands = yield* LegacyStartupCommands
  const unit = options.unit ?? "magnitude.service"
  if (!/^[A-Za-z0-9.-]+\.service$/.test(unit)) return yield* failed("Invalid legacy user-service name")
  const runtime = options.runtimeDirectory ?? (process.env.XDG_RUNTIME_DIR && isAbsolute(process.env.XDG_RUNTIME_DIR)
    ? process.env.XDG_RUNTIME_DIR : `/run/user/${process.getuid!()}`)
  if (!isAbsolute(runtime)) return yield* failed("Legacy user-manager runtime directory must be absolute")
  const path = join(options.home, ".config/systemd/user", unit)
  const manager = Effect.tryPromise({ try: async () => {
    const info = await lstat(join(runtime, "systemd/private")).catch(error => { if (missing(error)) return null; throw error })
    if (info === null) return false
    if (!info.isSocket() || info.uid !== process.getuid!()) throw new Error("User-manager endpoint is not a user-owned socket")
    return true
  }, catch: error => failed(`Cannot inspect legacy user manager: ${String(error)}`) })
  const document = Effect.tryPromise({ try: async () => {
    const info = await lstat(path).catch(error => { if (missing(error)) return null; throw error })
    if (info === null) return Option.none<{ readonly text: string; readonly digest: string }>()
    if (!info.isFile() || info.isSymbolicLink() || info.uid !== process.getuid!() || info.size > 1024 * 1024) throw new Error("Legacy unit must be a bounded regular file owned by this user")
    const bytes = await readFile(path)
    return Option.some({ text: bytes.toString("utf8"), digest: createHash("sha256").update(bytes).digest("hex") })
  }, catch: error => failed(`Cannot read legacy user service: ${String(error)}`) })
  const run = (args: readonly string[]) => commands.run("systemctl", ["--user", "--no-pager", ...args])
  const requireSuccess = (args: readonly string[]) => run(args).pipe(Effect.flatMap(result => result.code === 0
    ? Effect.void : Effect.fail(failed(result.stderr.trim() || `Legacy user-service command exited ${result.code}`))))
  const state = run(["show", "--all", `--property=${Object.keys(UnitState.fields).join(",")}`, unit]).pipe(Effect.flatMap(result => {
    if (result.code !== 0 && result.code !== 4) return Effect.fail(failed(result.stderr.trim() || "Cannot inspect legacy user service"))
    const properties: Record<string, string> = {}
    for (const line of result.stdout.trim().split("\n")) {
      const index = line.indexOf("=")
      const key = line.slice(0, index)
      if (index < 1 || Object.hasOwn(properties, key)) return Effect.fail(failed("Legacy user-manager response is malformed"))
      properties[key] = line.slice(index + 1)
    }
    return Schema.decodeUnknown(UnitState)(properties).pipe(Effect.mapError(() => failed("Legacy user-service state is unavailable, overridden, or needs reloading")))
  }))
  const absent = (value: typeof UnitState.Type) => value.LoadState === "not-found" && value.ActiveState === "inactive" && value.MainPID === 0 && value.FragmentPath === ""
  const verifyUnit = (value: typeof UnitState.Type) => value.LoadState === "loaded" && value.FragmentPath === path && value.DropInPaths === "" &&
      ["enabled", "enabled-runtime", "disabled"].includes(value.UnitFileState)
    ? Effect.void : Effect.fail(failed("Legacy user service has an unverified source, override, or enablement state"))
  const inspect = Effect.gen(function* () {
    const source = yield* document
    if (!(yield* manager)) {
      if (Option.isSome(source)) return yield* failed("The legacy user service exists but its systemd user manager is unavailable")
      return Option.none<LegacyLinuxStartup>()
    }
    const current = yield* state
    if (Option.isNone(source)) {
      if (!absent(current)) return yield* failed("Legacy user service is loaded without its verified source file")
      return Option.none<LegacyLinuxStartup>()
    }
    yield* verifySource(source.value.text)
    yield* verifyUnit(current)
    const latest = yield* document
    if (!Option.exists(latest, value => value.digest === source.value.digest)) return yield* failed("Legacy user service changed during inspection")
    return Option.some(yield* Schema.decodeUnknown(LegacyLinuxStartup)({ _tag: "SystemdUserService", unit, path, digest: source.value.digest,
      enabled: current.UnitFileState === "enabled", ...(current.MainPID > 0 ? { runningPid: current.MainPID } : {}),
    }).pipe(Effect.mapError(() => failed("Legacy user-service identity is malformed"))))
  })
  const unregister = (expected: LegacyLinuxStartup) => Effect.gen(function* () {
    if (expected.unit !== unit || expected.path !== path) return yield* failed("Legacy user service belongs to another installation")
    const source = yield* document
    if (Option.isNone(source)) {
      if (yield* manager) {
        // A crash may follow unlink but precede manager reload; finish that checkpoint on replay.
        yield* requireSuccess(["daemon-reload"])
        if (!absent(yield* state)) return yield* failed("Legacy user-service absence remains unproven")
      }
      return
    }
    if (source.value.digest !== expected.digest) return yield* failed("Legacy user service changed after migration was prepared")
    if (!(yield* manager)) return yield* failed("Cannot unregister legacy startup without its systemd user manager")
    const verifyCurrent = state.pipe(Effect.tap(verifyUnit), Effect.flatMap(current => current.MainPID !== 0 && !Option.contains(expected.runningPid, current.MainPID)
      ? Effect.fail(failed("Legacy user service is running a different process")) : Effect.succeed(current)))
    const before = yield* verifyCurrent
    if (before.UnitFileState === "enabled-runtime") yield* requireSuccess(["disable", "--runtime", unit])
    yield* requireSuccess(["disable", unit])
    if ((yield* verifyCurrent).UnitFileState !== "disabled") return yield* failed("Legacy user service remains enabled")
    yield* requireSuccess(["stop", unit])
    const stopped = yield* state
    yield* verifyUnit(stopped)
    if (stopped.MainPID !== 0 || !["inactive", "failed"].includes(stopped.ActiveState)) return yield* failed("Legacy user service remains active")
    if (stopped.ActiveState === "failed") yield* requireSuccess(["reset-failed", unit])
    const latest = yield* document
    if (!Option.exists(latest, value => value.digest === expected.digest)) return yield* failed("Legacy user service changed during unregistration")
    yield* Effect.tryPromise({ try: () => unlink(path), catch: error => failed(`Cannot remove retired legacy unit: ${String(error)}`) })
    yield* requireSuccess(["daemon-reload"])
    if (!absent(yield* state)) return yield* failed("Legacy user-service absence remains unproven")
  })
  return { inspect, unregister }
})
