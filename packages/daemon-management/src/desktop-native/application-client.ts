import { spawn } from "node:child_process"
import { Deferred, Effect, Option, Schedule, Schema } from "effect"
import { ApplicationControlFailed, ApplicationControlUnavailable } from "./application-control"
import { type ApplicationIntent, type ApplicationSnapshot } from "@magnitudedev/sdk/desktop-host"

export class ApplicationLaunchFailed extends Schema.TaggedError<ApplicationLaunchFailed>()("ApplicationLaunchFailed", { message: Schema.String }) {}

export interface ApplicationLaunchCommand {
  readonly executable: string
  readonly arguments: readonly string[]
  readonly environment: Readonly<Record<string, string | undefined>>
  readonly installationGuarded?: boolean
}

/** Observe launch failure only until control admission; cancellation never kills the detached app. */
export const launchApplicationProcess = <A, E, R>(command: ApplicationLaunchCommand, observe: Effect.Effect<A, E, R>) => Effect.scoped(Effect.gen(function* () {
  const spawned = yield* Deferred.make<void, ApplicationLaunchFailed>()
  const failed = yield* Deferred.make<never, ApplicationLaunchFailed>()
  yield* Effect.acquireRelease(Effect.sync(() => {
    const child = spawn(command.executable, [...command.arguments], {
      env: command.environment, detached: true, stdio: "ignore", windowsHide: true,
    })
    const onError = (error: Error) => {
      const failure = Effect.fail(new ApplicationLaunchFailed({ message: `Could not start Magnitude: ${error.message}` }))
      Deferred.unsafeDone(spawned, failure)
      Deferred.unsafeDone(failed, failure)
    }
    const onSpawn = () => Deferred.unsafeDone(spawned, Effect.void)
    const onExit = (code: number | null, signal: NodeJS.Signals | null) => {
      // A successful platform dispatcher (macOS open) exits before the app responds.
      if (code === 0) return
      Deferred.unsafeDone(failed, Effect.fail(new ApplicationLaunchFailed({ message: command.installationGuarded && code === 75
        ? "Magnitude installation is in progress or needs package-manager repair. Retry after installation finishes."
        : `Magnitude launcher exited ${signal ? `after signal ${signal}` : `with code ${code}`}. Open the desktop app to inspect startup.` })))
    }
    child.once("error", onError)
    child.once("spawn", onSpawn)
    child.once("exit", onExit)
    return { child, onError, onSpawn, onExit }
  }), ({ child, onError, onSpawn, onExit }) => Effect.sync(() => {
    // Failed spawn can emit its error after cancellation; retain that one-shot listener.
    if (child.pid !== undefined) child.removeListener("error", onError)
    child.removeListener("spawn", onSpawn)
    child.removeListener("exit", onExit)
    child.unref()
  }))
  yield* Deferred.await(spawned)
  return yield* Effect.raceFirst(observe, Deferred.await(failed))
}))

export interface ApplicationClientOptions {
  readonly launch: (intent: "EnsureRunning" | "ShowWindow", observe: Effect.Effect<ApplicationSnapshot, ApplicationControlFailed | ApplicationControlUnavailable | ApplicationLaunchFailed>) => Effect.Effect<ApplicationSnapshot, ApplicationControlFailed | ApplicationControlUnavailable | ApplicationLaunchFailed>
  readonly request: (intent: ApplicationIntent) => Effect.Effect<ApplicationSnapshot, ApplicationControlFailed | ApplicationControlUnavailable>
}

/** A request may launch once. Waiting and recovery only observe that owner; they never relaunch it. */
export const makeApplicationClient = (options: ApplicationClientOptions) => {
  const request = options.request
  const ensure = (intent: "EnsureRunning" | "ShowWindow" = "EnsureRunning") => Effect.gen(function* () {
    const existing = yield* request(intent).pipe(Effect.map(Option.some), Effect.catchTag("ApplicationControlUnavailable", () => Effect.succeed(Option.none())))
    if (Option.isSome(existing)) return existing.value
    const observe = request(intent).pipe(
      Effect.retry({ while: error => error._tag === "ApplicationControlUnavailable", schedule: Schedule.spaced("100 millis") }),
      Effect.timeoutFail({ duration: "30 seconds", onTimeout: () => new ApplicationLaunchFailed({ message: "Magnitude did not respond after launch. Open the desktop app to inspect startup." }) }),
    )
    return yield* options.launch(intent, observe)
  }).pipe(Effect.flatMap(snapshot => snapshot.service._tag === "Stopping" || snapshot.service._tag === "Stopped"
    ? Effect.fail(new ApplicationLaunchFailed({ message: "Magnitude is quitting. This request will not restart it." }))
    : Effect.succeed(snapshot)))
  const awaitReady = (owner: ApplicationSnapshot, rpcVersion: number) => Effect.gen(function* () {
    let snapshot = owner
    for (;;) {
      if (snapshot.pid !== owner.pid) return yield* new ApplicationLaunchFailed({ message: "The Magnitude application changed while starting. Retry the command." })
      switch (snapshot.service._tag) {
        case "Ready":
          if (snapshot.service.health.rpcVersion !== rpcVersion) return yield* new ApplicationLaunchFailed({ message: "The Magnitude desktop app and CLI use different protocol versions. Update them together." })
          return snapshot
        case "Failed": case "CleanupFailed": return yield* new ApplicationLaunchFailed({ message: snapshot.service.message })
        case "Stopping": case "Stopped": return yield* new ApplicationLaunchFailed({ message: "Magnitude is quitting. This request will not restart it." })
        case "Starting": break
      }
      yield* Effect.sleep("100 millis")
      snapshot = yield* request("Observe")
    }
  }).pipe(Effect.timeoutFail({ duration: "5 minutes", onTimeout: () => new ApplicationLaunchFailed({ message: "Magnitude service startup timed out. Open Status in the desktop app." }) }))
  return { ensure, awaitReady, observe: request("Observe"), quit: request("Quit") }
}
