import { Command, Options } from "@effect/cli"
import * as PlatformCommand from "@effect/platform/Command"
import { BunContext, BunRuntime } from "@effect/platform-bun"
import { FetchHttpClient } from "@effect/platform"
import { Console, Data, Effect, Layer } from "effect"
import { launchAcnServer } from "./server"
import { ACN_REVISION, ACN_VERSION } from "./version"
import { resolveRgPath } from "@magnitudedev/ripgrep"
import { defaultDataDir } from "./data-dir"

const debug = Options.boolean("debug")
const parentBound = Options.boolean("parent-bound")
const dataDir = Options.text("data-dir").pipe(Options.withDefault(defaultDataDir()))

const launchServer = (options: {
  readonly parentBound: boolean
  readonly debug: boolean
  readonly dataDir: string
}) => launchAcnServer(options)

const serve = Command.make("serve", { parentBound, debug, dataDir }, launchServer).pipe(
  Command.withDescription("Start the ACN server"),
)

const server = Command.make("server", { parentBound, debug, dataDir }, launchServer).pipe(
  Command.withDescription("Alias for serve"),
)

const version = Command.make("version", {}, () => Console.log(ACN_VERSION)).pipe(
  Command.withDescription("Print the ACN version"),
)

const coordinationRevision = Command.make(
  "coordination-revision",
  {},
  () => Console.log(String(ACN_REVISION)),
).pipe(Command.withDescription("Print the embedded scalar ACN coordination revision"))

class RipgrepVerificationError extends Data.TaggedError("RipgrepVerificationError")<{
  readonly cause: unknown
  readonly message: string
}> {}

const toRipgrepVerificationError = (cause: unknown): RipgrepVerificationError =>
  cause instanceof RipgrepVerificationError
    ? cause
    : new RipgrepVerificationError({
        cause,
        message: cause instanceof Error ? cause.message : String(cause),
      })

const resolveRipgrepPath = Effect.tryPromise({
  try: () => resolveRgPath(),
  catch: toRipgrepVerificationError,
})

const verifyRipgrep = Effect.gen(function* () {
  const rgPath = yield* resolveRipgrepPath
  const stdout = yield* PlatformCommand.make(rgPath, "--version").pipe(
    PlatformCommand.string,
    Effect.mapError(toRipgrepVerificationError),
  )
  return { rgPath, version: stdout.split("\n")[0]?.trim() ?? "" }
})

const doctor = Command.make("doctor", {}, () =>
  verifyRipgrep.pipe(
    Effect.flatMap(({ rgPath, version }) => Console.log(`ripgrep: ${version}\npath: ${rgPath}`)),
  ),
).pipe(Command.withDescription("Verify packaged ACN runtime dependencies"))

const acn = Command.make("magnitude-acn", { parentBound, debug, dataDir }, launchServer).pipe(
  Command.withDescription("Magnitude Agent Control Node"),
  Command.withSubcommands([serve, server, version, coordinationRevision, doctor]),
)

const cli = Command.run(acn, {
  name: "Magnitude ACN",
  version: ACN_VERSION,
})

const program = cli(process.argv).pipe(
  Effect.provide(Layer.merge(BunContext.layer, FetchHttpClient.layer)),
)
BunRuntime.runMain(program)
