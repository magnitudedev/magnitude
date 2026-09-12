import * as Command from "@effect/platform/Command"
import * as NodeContext from "@effect/platform-node/NodeContext"
import { stripVTControlCharacters } from "node:util"
import { Effect, Exit, Fiber, Schema, Scope, Stream } from "effect"
import type { ExtensionAPI, ExtensionContext } from "@earendil-works/pi-coding-agent"

export class PiSetupFailed extends Schema.TaggedError<PiSetupFailed>()("PiSetupFailed", { message: Schema.String }) {}

/** Opening the desktop acknowledges only navigation, never completed onboarding or connection. */
export const openMagnitudeDesktop = (cwd: string) => Effect.scoped(Effect.gen(function* () {
  const executable = process.env.MAGNITUDE_CLI?.trim() || "magnitude"
  const child = yield* Command.make(executable, "app", "open").pipe(Command.workingDirectory(cwd), Command.start)
  const [code, , stderr] = yield* Effect.all([
    child.exitCode, child.stdout.pipe(Stream.runDrain),
    child.stderr.pipe(Stream.decodeText(), Stream.runFold("", (text, chunk) => (text + chunk).slice(-4096))),
  ], { concurrency: "unbounded" })
  if (code !== 0) return yield* new PiSetupFailed({ message: `Could not open Magnitude. ${stripVTControlCharacters(stderr).trim().slice(-1000) || `Command exited with status ${code}.`} Install the Magnitude desktop app and a matching CLI, then retry /magnitude-setup.` })
})).pipe(Effect.timeout("40 seconds"), Effect.mapError(error => error instanceof PiSetupFailed ? error : new PiSetupFailed({ message: `Could not open Magnitude: ${String(error)}. Install the Magnitude desktop app and make its CLI available, then retry /magnitude-setup.` })))

export const registerMagnitudeSetup = (pi: ExtensionAPI) => {
  const scope = Effect.runSync(Scope.make())
  const gate = Effect.runSync(Effect.makeSemaphore(1))
  const action = (ctx: ExtensionContext) => Effect.gen(function* () {
    if (ctx.mode !== "tui") return yield* new PiSetupFailed({ message: "Run /magnitude-setup in Pi's interactive terminal." })
    if (!ctx.isIdle() || ctx.hasPendingMessages()) return yield* new PiSetupFailed({ message: "Wait for the current task to finish, then run /magnitude-setup." })
    yield* openMagnitudeDesktop(ctx.cwd)
    yield* Effect.sync(() => ctx.ui.notify("Opened Magnitude. Choose a model and connect Pi in Connections, then return here and run /reload.", "info"))
    return true
  })
  const run = (ctx: ExtensionContext) => Effect.runPromise(Effect.forkIn(
    gate.withPermitsIfAvailable(1)(action(ctx)).pipe(
      Effect.flatMap(result => result._tag === "Some" ? Effect.succeed(result.value) : Effect.fail(new PiSetupFailed({ message: "Magnitude is already opening." }))),
      Effect.catchAll(error => Effect.sync(() => { ctx.ui.notify(error.message, "error"); return false })),
      Effect.provide(NodeContext.layer),
    ), scope,
  ).pipe(Effect.flatMap(Fiber.join)))
  pi.registerCommand("magnitude-setup", {
    description: "Open Magnitude desktop to discover models and connect Pi",
    handler: async (_args, ctx) => { await run(ctx) },
  })
  return { run, dispose: () => Effect.runPromise(Scope.close(scope, Exit.void)) }
}
