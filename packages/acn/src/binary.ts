import { Command, Options } from "@effect/cli"
import * as PlatformCommand from "@effect/platform/Command"
import { BunContext, BunRuntime } from "@effect/platform-bun"
import { Console, Data, Effect, Layer } from "effect"
import { AcnServerLayer } from "./server"
import { killAllAcns } from "./kill-all"
import { ACN_VERSION } from "./version"
import { resolveRgPath } from "@magnitudedev/ripgrep"

const register = Options.boolean("register")
const debug = Options.boolean("debug")

const launchServer = (options: { readonly register: boolean; readonly debug: boolean }) =>
  Layer.launch(AcnServerLayer(options))

const serve = Command.make(
  "serve",
  { register, debug },
  ({ register, debug }) => launchServer({ register, debug })
).pipe(
  Command.withDescription("Start the ACN server")
)

const server = Command.make(
  "server",
  { register, debug },
  ({ register, debug }) => launchServer({ register, debug })
).pipe(
  Command.withDescription("Alias for serve")
)

const version = Command.make(
  "version",
  {},
  () => Console.log(ACN_VERSION)
).pipe(
  Command.withDescription("Print the ACN version")
)

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
    Effect.mapError(toRipgrepVerificationError)
  )
  return { rgPath, version: stdout.split("\n")[0]?.trim() ?? "" }
})

const doctor = Command.make(
  "doctor",
  {},
  () =>
    verifyRipgrep.pipe(
      Effect.flatMap(({ rgPath, version }) =>
        Console.log(`ripgrep: ${version}\npath: ${rgPath}`)
      )
    )
).pipe(
  Command.withDescription("Verify packaged ACN runtime dependencies")
)

const killAll = Command.make(
  "kill-all",
  {},
  () => killAllAcns
).pipe(
  Command.withDescription("Terminate all registered ACN processes")
)

const acn = Command.make(
  "magnitude-acn",
  { register, debug },
  ({ register, debug }) => launchServer({ register, debug })
).pipe(
  Command.withDescription("Magnitude Agent Control Node"),
  Command.withSubcommands([serve, server, version, doctor, killAll])
)

const cli = Command.run(acn, {
  name: "Magnitude ACN",
  version: ACN_VERSION,
})

const program = cli(process.argv).pipe(Effect.provide(BunContext.layer))
BunRuntime.runMain(program)
