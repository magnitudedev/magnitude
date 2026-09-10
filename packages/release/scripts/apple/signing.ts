import * as Command from "@effect/platform/Command"
import * as FileSystem from "@effect/platform/FileSystem"
import { Config, Data, Effect, Schema, Stream } from "effect"
import { resolve } from "node:path"

export class AppleDistributionFailed extends Data.TaggedError("AppleDistributionFailed")<{
  readonly message: string
}> {}

const appleCommandResult = (executable: string, ...args: readonly string[]) => Effect.scoped(Effect.gen(function* () {
  const child = yield* Command.make(executable, ...args).pipe(Command.start)
  const read = (stream: typeof child.stdout) => stream.pipe(Stream.decodeText(), Stream.runFoldEffect("", (previous, next) =>
    previous.length + next.length <= 1024 * 1024 ? Effect.succeed(previous + next) : new AppleDistributionFailed({ message: `${executable} output limit exceeded` })))
  const [stdout, stderr, code] = yield* Effect.all([read(child.stdout), read(child.stderr), child.exitCode], { concurrency: "unbounded" })
  return { stdout, stderr, code }
})).pipe(
  Effect.mapError((error) => new AppleDistributionFailed({ message: String(error) })),
)

export const appleCommand = (executable: string, ...args: readonly string[]) => appleCommandResult(executable, ...args).pipe(
  Effect.flatMap(({ stdout, stderr, code }) => code === 0 ? Effect.succeed(stdout)
    : new AppleDistributionFailed({ message: `${executable} exited ${code}: ${stderr.slice(-4000) || stdout.slice(-4000)}` })),
)

export const appleSigning = Effect.gen(function* () {
  const mode = yield* Config.literal("adhoc", "developer-id")("MAGNITUDE_APPLE_DISTRIBUTION").pipe(Config.withDefault("adhoc"))
  if (mode === "adhoc") return { mode, identity: "-", team: "" } as const
  const identity = yield* Config.nonEmptyString("APPLE_SIGNING_IDENTITY")
  const team = yield* Config.string("APPLE_TEAM_ID")
  if (!/^[A-Z0-9]{10}$/.test(team) || !identity.startsWith("Developer ID Application:")) {
    return yield* new AppleDistributionFailed({ message: "Developer ID release requires a valid Team ID and Developer ID Application identity" })
  }
  return { mode, identity, team } as const
})

export const signAppleCode = (file: string, identifier: string, profile: "native" | "bun" | "library" = "native") => Effect.gen(function* () {
  const signing = yield* appleSigning
  const keychain = yield* Config.string("APPLE_RELEASE_KEYCHAIN").pipe(Config.withDefault(""))
  yield* appleCommand("/usr/bin/codesign", "--force", "--sign", signing.identity, "--identifier", identifier,
    ...(keychain ? ["--keychain", keychain] : []),
    ...(signing.mode === "developer-id" ? ["--timestamp"] : ["--timestamp=none"]),
    ...(signing.mode === "developer-id" && profile !== "library" ? ["--options", "runtime"] : []),
    ...(profile === "bun" ? ["--entitlements", resolve(import.meta.dir, "../../resources/macos/bun.entitlements.plist")] : []), file)
  yield* appleCommand("/usr/bin/codesign", "--verify", "--strict", file)
})

const NotaryResult = Schema.Struct({ id: Schema.String, status: Schema.Literal("Accepted") })
/** The archive is immutable before submission; logs and accepted IDs stay outside release assets. */
export const notarize = (archive: string, receipt: string) => Effect.gen(function* () {
  const signing = yield* appleSigning
  if (signing.mode === "adhoc") return
  const fs = yield* FileSystem.FileSystem
  const keychain = yield* Config.string("APPLE_RELEASE_KEYCHAIN")
  const output = yield* appleCommandResult("/usr/bin/xcrun", "notarytool", "submit", archive,
    "--keychain-profile", "magnitude-release", "--keychain", keychain,
    "--wait", "--timeout", "30m", "--output-format", "json").pipe(Effect.timeout("31 minutes"))
  // Persist Apple's response even if it is a rejection, before failing the release.
  yield* fs.writeFileString(receipt, output.stdout || (yield* Schema.encode(Schema.parseJson(Schema.Struct({ error: Schema.String, exitCode: Schema.Int })))({ error: output.stderr, exitCode: Number(output.code) })), { mode: 0o600 })
  const response = yield* Schema.decodeUnknown(Schema.parseJson(Schema.Struct({ id: Schema.String, status: Schema.String })))(output.stdout)
  if (response.status !== "Accepted" || output.code !== 0) {
    yield* appleCommand("/usr/bin/xcrun", "notarytool", "log", response.id, "--keychain-profile", "magnitude-release", "--keychain", keychain).pipe(
      Effect.flatMap((log) => fs.writeFileString(receipt.replace(/\.json$/, ".log.json"), log, { mode: 0o600 })),
      Effect.tapError((error) => Effect.logWarning("Could not retrieve Apple's rejection log", error)), Effect.ignore,
    )
    return yield* new AppleDistributionFailed({ message: `Apple notarization ${response.id}: ${response.status}; see the retained notarization log` })
  }
  return yield* Schema.decodeUnknown(NotaryResult)(response)
})
