import { createHash } from "node:crypto"
import { lstat, readFile, unlink } from "node:fs/promises"
import { basename, join } from "node:path"
import { Effect, Option, Schema } from "effect"

import { CommandResult, LegacyStartupCommands, LegacyStartupFailed } from "./legacy-startup-command"

const StartupPid = Schema.Int.pipe(Schema.between(1, Number.MAX_SAFE_INTEGER))
export const LegacyMacStartup = Schema.TaggedStruct("MacLaunchAgent", {
  label: Schema.NonEmptyString, path: Schema.NonEmptyString,
  digest: Schema.String.pipe(Schema.pattern(/^[a-f0-9]{64}$/), Schema.brand("LegacyStartupDigest")),
  enabled: Schema.Boolean,
  runningPid: Schema.optionalWith(StartupPid, { as: "Option", exact: true }),
})
export type LegacyMacStartup = typeof LegacyMacStartup.Type
const Plist = Schema.Struct({
  Label: Schema.String, ProgramArguments: Schema.NonEmptyArray(Schema.String), RunAtLoad: Schema.Literal(true),
  Disabled: Schema.optionalWith(Schema.Boolean, { as: "Option", exact: true }),
})
const missing = (error: unknown) => error instanceof Error && "code" in error && error.code === "ENOENT"

/** Only the historical launch-agent namespace is migrated; test roots/labels stay isolated. */
export const makeMacLegacyStartup = (home: string, label = "dev.magnitude.acn") => Effect.gen(function* () {
  if (!/^[A-Za-z0-9.-]+$/.test(label)) return yield* new LegacyStartupFailed({ message: "Invalid legacy launch-agent label" })
  const commands = yield* LegacyStartupCommands
  const path = join(home, "Library/LaunchAgents", `${label}.plist`)
  const domain = `gui/${process.getuid!()}`
  const target = `${domain}/${label}`
  const run = (args: readonly string[]) => commands.run("/bin/launchctl", args)
  const requireSuccess = (result: typeof CommandResult.Type) => result.code === 0 ? Effect.succeed(result.stdout)
    : Effect.fail(new LegacyStartupFailed({ message: result.stderr.trim() || `Legacy startup command exited ${result.code}` }))
  const document = Effect.tryPromise({ try: async () => {
    try {
      const info = await lstat(path)
      if (!info.isFile() || info.isSymbolicLink() || info.uid !== process.getuid!()) throw new Error("Legacy launch agent must be a regular file owned by this user")
      const bytes = await readFile(path)
      return Option.some(createHash("sha256").update(bytes).digest("hex"))
    } catch (error) { if (missing(error)) return Option.none<string>(); throw error }
  }, catch: error => new LegacyStartupFailed({ message: String(error) }) })
  const loaded = Effect.gen(function* () {
    const result = yield* run(["print", target])
    if (result.code === 113) return { _tag: "Unloaded" as const }
    const output = yield* requireSuccess(result)
    const registeredPath = output.match(/^\s*path = (.+)$/m)?.[1]?.trim()
    if (registeredPath !== path) return yield* new LegacyStartupFailed({ message: "Loaded legacy launch agent has an unverified source path" })
    const pid = output.match(/^\s*pid = (\d+)$/m)?.[1]
    const identity = pid === undefined ? Option.none<number>() : Option.some(yield* Schema.decodeUnknown(StartupPid)(Number(pid)).pipe(
      Effect.mapError(() => new LegacyStartupFailed({ message: "Loaded legacy job has an invalid process ID" }))))
    return { _tag: "Loaded" as const, pid: identity }
  })
  const inspect = Effect.gen(function* () {
    const digest = yield* document
    const active = yield* loaded
    if (Option.isNone(digest)) {
      if (active._tag === "Loaded") return yield* new LegacyStartupFailed({ message: "Legacy launch agent is loaded but its source file is missing" })
      return Option.none<LegacyMacStartup>()
    }
    const result = yield* commands.run("/usr/bin/plutil", ["-convert", "json", "-o", "-", "--", path]).pipe(Effect.flatMap(requireSuccess))
    const plist = yield* Schema.decodeUnknown(Schema.parseJson(Plist))(result).pipe(
      Effect.mapError(() => new LegacyStartupFailed({ message: "Legacy launch agent is malformed" })))
    if (plist.Label !== label || basename(plist.ProgramArguments[0]) !== "magnitude-service" || plist.ProgramArguments[1] !== "serve") {
      return yield* new LegacyStartupFailed({ message: "Legacy launch agent does not describe the Magnitude service" })
    }
    const disabled = yield* run(["print-disabled", domain]).pipe(Effect.flatMap(requireSuccess))
    const override = disabled.split("\n").map(line => line.trim()).find(line => line.startsWith(`${JSON.stringify(label)} => `))?.split(" => ")[1]
    if (override !== undefined && override !== "enabled" && override !== "disabled") return yield* new LegacyStartupFailed({ message: "Legacy login preference is unrecognized" })
    const latest = yield* document
    if (!Option.contains(latest, digest.value)) return yield* new LegacyStartupFailed({ message: "Legacy launch agent changed during inspection" })
    const enabled = override === "enabled" || (override === undefined && !Option.getOrElse(plist.Disabled, () => false))
    return Option.some(LegacyMacStartup.make({ label, path, digest: LegacyMacStartup.fields.digest.make(digest.value), enabled, runningPid: active._tag === "Loaded" ? active.pid : Option.none() }))
  })
  // The migration owner must durably retain login intent and captured process identities first:
  // bootout may terminate the legacy service before separately grouped descendants are retired.
  const unregister = (expected: LegacyMacStartup) => Effect.gen(function* () {
    if (expected.label !== label || expected.path !== path) return yield* new LegacyStartupFailed({ message: "Legacy registration belongs to another installation" })
    const digest = yield* document
    if (Option.isNone(digest)) {
      if ((yield* loaded)._tag === "Loaded") return yield* new LegacyStartupFailed({ message: "Cannot unregister a loaded legacy job without its verified source file" })
      return
    }
    if (digest.value !== expected.digest) return yield* new LegacyStartupFailed({ message: "Legacy launch agent changed after migration was prepared" })
    const verifyLoaded = loaded.pipe(Effect.flatMap(current => current._tag === "Loaded" && Option.isSome(current.pid) && !Option.contains(expected.runningPid, current.pid.value)
      ? Effect.fail(new LegacyStartupFailed({ message: "Legacy launch agent is running a different process than the migration snapshot" })) : Effect.succeed(current)))
    yield* verifyLoaded
    yield* run(["disable", target]).pipe(Effect.flatMap(requireSuccess))
    if ((yield* verifyLoaded)._tag === "Loaded") yield* run(["bootout", target]).pipe(Effect.flatMap(requireSuccess))
    if ((yield* loaded)._tag === "Loaded") return yield* new LegacyStartupFailed({ message: "Legacy launch agent remains loaded" })
    const latest = yield* document
    if (!Option.contains(latest, expected.digest)) return yield* new LegacyStartupFailed({ message: "Legacy launch agent changed during unregistration" })
    yield* Effect.tryPromise({ try: () => unlink(path), catch: error => new LegacyStartupFailed({ message: String(error) }) })
  })
  return { inspect, unregister }
})
