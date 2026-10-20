import * as Command from "@effect/platform/Command"
import * as FileSystem from "@effect/platform/FileSystem"
import { Config, Data, Duration, Effect, Option, Schedule, Schema, Stream } from "effect"
import { resolve } from "node:path"
import { AppleTeamId } from "../../src/desktop-distribution"

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

// Apple's notary, ticket and timestamp services drop connections now and then. Only those failures are
// retried; a rejected signature, notarization or missing ticket fails at once.
const transientApple = /network connection was lost|NSURLErrorDomain|request timed out|not connected to the internet|timestamp service is not available/i
const appleRetries = 4
const appleBackoff = (attempt: number) => Duration.seconds(15 * 2 ** attempt)

/** An Apple command, retried with backoff when its connection to Apple's services drops. */
export const appleCommand = (executable: string, ...args: readonly string[]) => appleCommandResult(executable, ...args).pipe(
  Effect.flatMap(({ stdout, stderr, code }) => code === 0 ? Effect.succeed(stdout)
    : new AppleDistributionFailed({ message: `${executable} exited ${code}: ${stderr.slice(-4000) || stdout.slice(-4000)}` })),
  Effect.retry({
    schedule: Schedule.exponential(appleBackoff(0)).pipe(Schedule.intersect(Schedule.recurs(appleRetries))),
    while: error => transientApple.test(error.message),
  }),
)

export const appleSigning = Effect.gen(function* () {
  const mode = yield* Config.literal("adhoc", "developer-id")("MAGNITUDE_APPLE_DISTRIBUTION").pipe(Config.withDefault("adhoc"))
  if (mode === "adhoc") return { mode, identity: "-", team: "" } as const
  const identity = yield* Config.nonEmptyString("APPLE_SIGNING_IDENTITY")
  const team = yield* Schema.decodeUnknown(AppleTeamId)(yield* Config.string("APPLE_TEAM_ID")).pipe(Effect.option)
  if (Option.isNone(team) || !identity.startsWith("Developer ID Application:")) {
    return yield* new AppleDistributionFailed({ message: "Developer ID release requires a valid Team ID and Developer ID Application identity" })
  }
  return { mode, identity, team: team.value } as const
})

/** Apple accepts the stored notary credentials. Checked before building, so a revoked key fails at once. */
export const verifyNotaryCredentials = Effect.gen(function* () {
  if ((yield* appleSigning).mode === "adhoc") return
  const keychain = yield* Config.string("APPLE_RELEASE_KEYCHAIN")
  yield* appleCommand("/usr/bin/xcrun", "notarytool", "history", "--keychain-profile", "magnitude-release", "--keychain", keychain, "--output-format", "json")
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

const submissionId = (text: string) => Option.fromNullable(/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}/i.exec(text)?.[0])

const NotaryResult = Schema.Struct({ id: Schema.String, status: Schema.Literal("Accepted") })
/** The archive is immutable before submission; logs and accepted IDs stay outside release assets. */
export const notarize = (archive: string, receipt: string) => Effect.gen(function* () {
  const signing = yield* appleSigning
  if (signing.mode === "adhoc") return
  const fs = yield* FileSystem.FileSystem
  const keychain = yield* Config.string("APPLE_RELEASE_KEYCHAIN")
  const notary = (...args: readonly string[]) => appleCommandResult("/usr/bin/xcrun", "notarytool", ...args,
    "--keychain-profile", "magnitude-release", "--keychain", keychain, "--timeout", "30m", "--output-format", "json").pipe(Effect.timeout("31 minutes"))
  let output = yield* notary("submit", archive, "--wait")
  // A connection dropped after the upload leaves a submission behind: wait on it instead of uploading again.
  let submission = Option.none<string>()
  for (let attempt = 0; output.code !== 0 && transientApple.test(output.stderr + output.stdout) && attempt < appleRetries; attempt++) {
    submission = Option.orElse(submissionId(output.stdout + output.stderr), () => submission)
    yield* Effect.logWarning(`Apple notarization lost its connection; retrying (${attempt + 1}/${appleRetries})`)
    yield* Effect.sleep(appleBackoff(attempt))
    output = yield* (Option.isSome(submission) ? notary("wait", submission.value) : notary("submit", archive, "--wait"))
  }
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
