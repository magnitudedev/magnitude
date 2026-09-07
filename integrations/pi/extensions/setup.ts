import * as Command from "@effect/platform/Command"
import * as CommandExecutor from "@effect/platform/CommandExecutor"
import * as FileSystem from "@effect/platform/FileSystem"
import * as NodeContext from "@effect/platform-node/NodeContext"
import { FetchHttpClient } from "@effect/platform"
import { delimiter, join } from "node:path"
import { stripVTControlCharacters } from "node:util"
import { Context, Effect, Exit, Fiber, Layer, ManagedRuntime, Schema, Scope, Stream } from "effect"
import type { ExtensionAPI, ExtensionContext } from "@earendil-works/pi-coding-agent"
import { CancellableLoader, type TUI } from "@earendil-works/pi-tui"
import { runInteractiveProcess, type InteractiveProcessTermination } from "@magnitudedev/utils/process"
import { MagnitudeClient, formatConnectionError, type ProviderModelId } from "@magnitudedev/sdk"

export class PiSetupFailed extends Schema.TaggedError<PiSetupFailed>()("PiSetupFailed", {
  message: Schema.String,
}) {}
const failure = (message: string) => new PiSetupFailed({ message })

/** Only an absent ambient CLI authorizes installation. An override or a broken
 * installation is not permission to replace the user's chosen executable. */
export const prepareMagnitudeCli = (cwd: string, setMessage: (message: string) => Effect.Effect<void> = () => Effect.void) => Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const override = process.env.MAGNITUDE_CLI?.trim()
  const executable = override || "magnitude"
  const probe = Effect.scoped(Effect.gen(function* () {
    const child = yield* Command.make(executable, "--version").pipe(Command.start)
    const [code, , stderr] = yield* Effect.all([
      child.exitCode,
      Stream.runDrain(child.stdout),
      child.stderr.pipe(Stream.decodeText(), Stream.runFold("", (text, part) => (text + part).slice(-4_096))),
    ], { concurrency: "unbounded" })
    if (code !== 0) return yield* failure(`Magnitude could not run at ${executable} (exit ${code}). ${stripVTControlCharacters(stderr).trim().slice(-600)}${override ? " Check MAGNITUDE_CLI." : ""}`)
  }))
  yield* probe.pipe(Effect.catchIf(
    error => !override && error._tag === "SystemError" && error.reason === "NotFound",
    error => Effect.gen(function* () {
      // PATH lookup can report ENOENT for a non-executable file or a launcher
      // whose interpreter is missing. Neither is an absent installation.
      for (const directory of (process.env.PATH ?? "").split(delimiter)) {
        if (yield* fs.exists(join(directory || cwd, executable))) return yield* error
      }
      yield* setMessage("Installing Magnitude…")
      yield* Effect.scoped(Effect.gen(function* () {
        const child = yield* Command.make("npm", "install", "--global", "@magnitudedev/cli").pipe(Command.workingDirectory(cwd), Command.start)
        const [code, , stderr] = yield* Effect.all([
          child.exitCode,
          Stream.runDrain(child.stdout),
          child.stderr.pipe(Stream.decodeText(), Stream.runFold("", (text, part) => (text + part).slice(-4_096))),
        ], { concurrency: "unbounded" })
        if (code !== 0) return yield* failure(`Magnitude installation failed (exit ${code}). ${stripVTControlCharacters(stderr).trim().slice(-600)} Run /magnitude-setup to retry.`)
      }))
      return yield* probe
    }),
  ), Effect.timeout("10 minutes"), Effect.mapError(error => error instanceof PiSetupFailed ? error
    : failure(`Could not prepare Magnitude: ${String(error)}. Run /magnitude-setup to retry${override ? ", or check MAGNITUDE_CLI" : ""}.`)))
  return executable
})

/** The host keeps rendering during non-interactive acquisition. Only the actual
 * setup TUI takes terminal ownership after this scoped loader has closed. */
export const withPiPreparation = <A, E, R>(ctx: ExtensionContext, work: (setMessage: (message: string) => Effect.Effect<void>) => Effect.Effect<A, E, R>) =>
  Effect.acquireUseRelease(
    Effect.async<{ loader: CancellableLoader; finish: () => void; closed: () => Promise<void> }, PiSetupFailed>((resume, signal) => {
      let closed: Promise<void>
      closed = ctx.ui.custom<void>((tui, theme, _keys, done) => {
        const loader = new CancellableLoader(tui, text => theme.fg("accent", text), text => theme.fg("muted", text), "Installing Magnitude…")
        if (signal.aborted) { loader.dispose(); done() }
        else resume(Effect.succeed({ loader, finish: () => done(), closed: () => closed }))
        return loader
      })
      closed.catch(error => resume(Effect.fail(failure(`Could not open Pi setup: ${String(error)}`))))
    }),
    ({ loader }) => work(message => Effect.sync(() => loader.setMessage(message))).pipe(Effect.raceFirst(
      Effect.async<never, PiSetupFailed>(resume => {
        loader.onAbort = () => resume(Effect.fail(failure("Magnitude setup cancelled. Run /magnitude-setup to retry.")))
        if (loader.signal.aborted) loader.onAbort()
        return Effect.sync(() => { loader.onAbort = undefined })
      }),
    )),
    ({ loader, finish, closed }) => Effect.sync(() => { loader.dispose(); finish() }).pipe(Effect.zipRight(Effect.promise(closed))),
  )

interface TerminalLease {
  readonly tui: TUI
  readonly finish: () => void
  readonly closed: () => Promise<void>
}

// A killed OpenTUI child cannot restore its terminal modes. Pi has already
// stopped (including leaving its own alternate screen), and start() below
// re-establishes the selected Pi mode and keyboard protocol.
const restoreHostTerminal = (tui: TUI) => tui.terminal.write(
  "\x1b[?2026l\x1b[?1003l\x1b[?1002l\x1b[?1000l\x1b[?1006l\x1b[?1004l"
  + "\x1b[?2004l\x1b[>4;0m\x1b[=0u\x1b[?1049l\x1b[0m\x1b[0 q",
)

/** Pi's custom-UI callback is the host boundary; work remains in the caller's Effect scope. */
export const withPiTerminal = <A, E, R>(ctx: ExtensionContext, work: Effect.Effect<A, E, R>) =>
  Effect.acquireUseRelease(
    Effect.async<TerminalLease, PiSetupFailed>((resume, signal) => {
      let closed: Promise<void>
      closed = ctx.ui.custom<void>((tui, _theme, _keys, done) => {
        if (signal.aborted) done()
        else resume(Effect.succeed({ tui, finish: () => done(), closed: () => closed }))
        return { render: () => [], invalidate: () => {} }
      })
      closed.catch(error => resume(Effect.fail(failure(`Could not open Pi setup: ${String(error)}`))))
    }),
    ({ tui }) => Effect.sync(() => tui.stop()).pipe(Effect.zipRight(work)),
    ({ tui, finish, closed }) => Effect.sync(() => {
      try { restoreHostTerminal(tui); tui.start(); tui.requestRender(true) } finally { finish() }
    }).pipe(Effect.zipRight(Effect.promise(closed))),
  )

export const validateSetupTermination = (termination: InteractiveProcessTermination) => {
  if (termination._tag === "Signaled") {
    return termination.signal === "SIGINT" ? Effect.succeed(false)
      : Effect.fail(failure(`Magnitude setup stopped unexpectedly (${termination.signal}). Run /magnitude-setup to retry.`))
  }
  if (termination.code === 130) return Effect.succeed(false)
  return termination.code === 0 ? Effect.succeed(true) : Effect.fail(failure(
    `Magnitude setup failed (exit ${termination.code}). Check the terminal error above; if setup --host pi is unsupported, update @magnitudedev/cli. Run /magnitude-setup to retry.`,
  ))
}

export const readSetupModel = Effect.gen(function* () {
  const client = yield* MagnitudeClient
  const { slots } = yield* client.models.getSlots({}).pipe(Effect.mapError(error =>
    failure(`Magnitude setup completed, but the selected model could not be read: ${error._tag === "RpcClientError" ? error.message : formatConnectionError(error)}. Run /reload and select it with /model.`)))
  if (slots.primary._tag !== "ConfiguredLocal") return yield* failure("Magnitude setup completed, but no local primary model is configured. Run /reload and select a Magnitude model with /model.")
  return slots.primary.selection.providerModelId
})

type PiSetupResult = { readonly _tag: "Completed"; readonly modelId: ProviderModelId } | { readonly _tag: "Cancelled" }
export interface PiSetup {
  readonly run: (ctx: ExtensionContext) => Effect.Effect<PiSetupResult, PiSetupFailed>
}
export const PiSetup = Context.GenericTag<PiSetup>("pi/PiSetup")

export const PiSetupLive = Layer.effect(PiSetup, Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const executor = yield* CommandExecutor.CommandExecutor
  return {
    run: (ctx) => Effect.scoped(Effect.gen(function* () {
      const executable = yield* withPiPreparation(ctx, setMessage => prepareMagnitudeCli(ctx.cwd, setMessage))
      const termination = yield* withPiTerminal(ctx, runInteractiveProcess({
          executable,
          args: ["setup", "--host", "pi"],
          environment: process.env,
          workingDirectory: ctx.cwd,
      }))
      if (!(yield* validateSetupTermination(termination))) return { _tag: "Cancelled" } as const
      // Only a successful setup needs SDK access. No service startup or polling on
      // extension load, a declined offer, or a cancelled/failed child.
      const modelId = yield* readSetupModel.pipe(Effect.provide(
        MagnitudeClient.layer({ autoStart: false }).pipe(Layer.provide(FetchHttpClient.layer)),
      ))
      return { _tag: "Completed", modelId } as const
    })).pipe(
      Effect.provideService(CommandExecutor.CommandExecutor, executor),
      Effect.provideService(FileSystem.FileSystem, fs),
      Effect.mapError(error => error instanceof PiSetupFailed ? error : failure(`Magnitude setup could not finish: ${String(error)}`)),
    ),
  } satisfies PiSetup
}))

export const registerMagnitudeSetup = (
  pi: ExtensionAPI,
  layer: Layer.Layer<PiSetup> = PiSetupLive.pipe(Layer.provide(NodeContext.layer)),
) => {
  const runtime = ManagedRuntime.make(layer)
  const scope = Effect.runSync(Scope.make())
  const gate = Effect.runSync(Effect.makeSemaphore(1))
  const action = (ctx: ExtensionContext) => Effect.gen(function* () {
    if (ctx.mode !== "tui") return yield* failure("Run /magnitude-setup in Pi's interactive terminal.")
    if (!ctx.isIdle() || ctx.hasPendingMessages()) return yield* failure("Wait for the current task to finish, then run /magnitude-setup.")
    const setup = yield* PiSetup
    const result = yield* setup.run(ctx)
    if (result._tag !== "Completed") return false
    yield* Effect.tryPromise({
      try: async () => {
        await ctx.modelRegistry.refresh()
        const model = ctx.modelRegistry.find("magnitude", result.modelId)
        if (!model || !(await pi.setModel(model))) throw new Error("The selected model is not available in Pi")
      },
      catch: error => failure(`Magnitude setup completed, but Pi could not activate the model: ${String(error)}. Run /reload and select it with /model.`),
    })
    return true
  })
  const run = (ctx: ExtensionContext) => runtime.runPromise(Effect.forkIn(
    gate.withPermitsIfAvailable(1)(action(ctx)).pipe(
      Effect.flatMap(result => result._tag === "Some" ? Effect.succeed(result.value) : Effect.fail(failure("Magnitude setup is already open."))),
      Effect.catchAll(error => Effect.sync(() => { ctx.ui.notify(error.message, "error"); return false })),
    ), scope,
  ).pipe(Effect.flatMap(Fiber.join)))
  pi.registerCommand("magnitude-setup", {
    description: "Choose and set up a local Magnitude model",
    handler: async (_args, ctx) => {
      const reload = await run(ctx)
      // This disposes the old runtime. All owned work has already left it.
      if (reload) {
        try { await ctx.reload() }
        catch (error) {
          throw failure(`Magnitude setup completed and selected the model, but Pi could not reload its resources: ${String(error)}. Run /reload to retry.`)
        }
      }
    },
  })
  return {
    run,
    dispose: async () => {
      await Effect.runPromise(Scope.close(scope, Exit.void))
      await runtime.dispose()
    },
  }
}
