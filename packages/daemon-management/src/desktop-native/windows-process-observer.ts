import { createRequire } from "node:module"
import { Context, Effect, Layer, Option, Schedule, Schema, type Scope } from "effect"
import { WindowsProcessId } from "@magnitudedev/utils/windows-native"

export class WindowsProcessObserverUnavailable extends Schema.TaggedError<WindowsProcessObserverUnavailable>()("WindowsProcessObserverUnavailable", { message: Schema.String }) {}
export class WindowsProcessObservationFailed extends Schema.TaggedError<WindowsProcessObservationFailed>()("WindowsProcessObservationFailed", {
  pid: WindowsProcessId, message: Schema.String,
  win32Code: Schema.optionalWith(Schema.Int, { as: "Option", exact: true }),
}) {}
export interface WindowsProcessObservation {
  readonly exited: Effect.Effect<boolean, WindowsProcessObservationFailed>
  readonly awaitExit: Effect.Effect<void, WindowsProcessObservationFailed>
}
export interface WindowsProcessObserver {
  readonly observe: (pid: WindowsProcessId) => Effect.Effect<Option.Option<WindowsProcessObservation>, WindowsProcessObservationFailed, Scope.Scope>
}
export const WindowsProcessObserver = Context.GenericTag<WindowsProcessObserver>("@magnitudedev/daemon-management/WindowsProcessObserver")
export interface WindowsProcessObserverBindings {
  readonly observeProcess: (pid: number) => unknown
  readonly observedProcessExited: (handle: object) => unknown
  readonly releaseObservedProcess: (handle: object) => void
}

/** Observing an application acquires no process-termination or job capability. */
export const windowsProcessObserverLayer = (native: WindowsProcessObserverBindings) => Layer.succeed(WindowsProcessObserver, {
  observe: pid => Effect.gen(function* () {
    const failure = (message: string, error?: unknown) => new WindowsProcessObservationFailed({ pid, message,
      win32Code: Option.fromNullable(typeof error === "object" && error !== null && "win32Code" in error && typeof error.win32Code === "number" && Number.isInteger(error.win32Code) ? error.win32Code : undefined),
    })
    const handle = yield* Effect.acquireRelease(Effect.try({
      try: () => {
        const handle = native.observeProcess(pid)
        if (handle === null) return Option.none<object>()
        if (typeof handle !== "object") throw new Error("Invalid process observation handle")
        return Option.some(handle)
      }, catch: error => failure("Could not open the Windows application process for observation.", error),
    }), handle => Option.isNone(handle) ? Effect.void : Effect.try({
      try: () => native.releaseObservedProcess(handle.value), catch: error => failure("Could not release the Windows process observation.", error),
    }).pipe(Effect.orDie))
    if (Option.isNone(handle)) return Option.none()
    const exited = Effect.try({ try: () => native.observedProcessExited(handle.value), catch: error => failure("Could not observe the Windows application exit.", error) }).pipe(
      Effect.flatMap(Schema.decodeUnknown(Schema.Boolean)),
      Effect.mapError(error => error instanceof WindowsProcessObservationFailed ? error : failure("Windows returned an invalid process-exit observation.")),
    )
    return Option.some({ exited, awaitExit: exited.pipe(Effect.repeat({ until: exited => exited, schedule: Schedule.spaced("100 millis") }), Effect.asVoid) })
  }),
})
export const windowsProcessObserverLayerFromLoader = (load: () => unknown) => Layer.unwrapEffect(Effect.try({
  try: () => {
    const native = load() as WindowsProcessObserverBindings
    if (![native.observeProcess, native.observedProcessExited, native.releaseObservedProcess].every(value => typeof value === "function")) throw new Error("Process observation exports are absent")
    return windowsProcessObserverLayer(native)
  }, catch: () => new WindowsProcessObserverUnavailable({ message: "The installed native host does not provide Windows process observation." }),
}))

export const nativeWindowsProcessObserverLayer = (addonPath: string) => windowsProcessObserverLayerFromLoader(() => createRequire(import.meta.url)(addonPath))
