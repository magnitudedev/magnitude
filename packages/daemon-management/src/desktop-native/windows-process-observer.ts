import { createRequire } from "node:module"
import { Context, Effect, Layer, Option, Schedule, Schema, type Scope } from "effect"
import { WindowsProcessId, WindowsProcessIdentity } from "@magnitudedev/utils/windows-native"

export const WindowsObservedProcess = Schema.Struct({
  ...WindowsProcessIdentity.fields,
  executable: Schema.NonEmptyString.pipe(Schema.maxLength(32767)),
  userSid: Schema.String.pipe(Schema.pattern(/^S-1-(?:\d+-)+\d+$/), Schema.brand("WindowsUserSid")),
})
export type WindowsObservedProcess = typeof WindowsObservedProcess.Type
export const WindowsProcessParent = Schema.Struct({ pid: WindowsProcessId, parentPid: Schema.Int.pipe(Schema.between(0, 0xffffffff)) })
export class WindowsProcessSnapshotFailed extends Schema.TaggedError<WindowsProcessSnapshotFailed>()("WindowsProcessSnapshotFailed", { message: Schema.String }) {}

export class WindowsProcessObserverUnavailable extends Schema.TaggedError<WindowsProcessObserverUnavailable>()("WindowsProcessObserverUnavailable", { message: Schema.String }) {}
export class WindowsProcessObservationFailed extends Schema.TaggedError<WindowsProcessObservationFailed>()("WindowsProcessObservationFailed", {
  pid: WindowsProcessId, message: Schema.String,
  win32Code: Schema.optionalWith(Schema.Int, { as: "Option", exact: true }),
}) {}
export interface WindowsProcessObservation {
  readonly details: Effect.Effect<WindowsObservedProcess, WindowsProcessObservationFailed>
  readonly exited: Effect.Effect<boolean, WindowsProcessObservationFailed>
  readonly awaitExit: Effect.Effect<void, WindowsProcessObservationFailed>
}
export interface WindowsProcessObserver {
  readonly snapshotParents: Effect.Effect<readonly (typeof WindowsProcessParent.Type)[], WindowsProcessSnapshotFailed>
  readonly observe: (pid: WindowsProcessId) => Effect.Effect<Option.Option<WindowsProcessObservation>, WindowsProcessObservationFailed, Scope.Scope>
}
export const WindowsProcessObserver = Context.GenericTag<WindowsProcessObserver>("@magnitudedev/daemon-management/WindowsProcessObserver")
export interface WindowsProcessObserverBindings {
  readonly observeProcess: (pid: number) => unknown
  readonly observedProcessExited: (handle: object) => unknown
  readonly observedProcessDetails: (handle: object) => unknown
  readonly snapshotProcessParents: () => unknown
  readonly releaseObservedProcess: (handle: object) => void
}

/** Observing an application acquires no process-termination or job capability. */
export const windowsProcessObserverLayer = (native: WindowsProcessObserverBindings) => Layer.succeed(WindowsProcessObserver, {
  snapshotParents: Effect.try({ try: () => native.snapshotProcessParents(), catch: () => new WindowsProcessSnapshotFailed({ message: "Could not capture the Windows process-parent snapshot." }) }).pipe(
    Effect.flatMap(Schema.decodeUnknown(Schema.Array(WindowsProcessParent).pipe(Schema.maxItems(65536)))),
    Effect.mapError(() => new WindowsProcessSnapshotFailed({ message: "Windows process-parent snapshot is unavailable or invalid." })),
    Effect.flatMap(rows => new Set(rows.map(row => row.pid)).size !== rows.length || rows.some(row => row.pid === row.parentPid)
      ? Effect.fail(new WindowsProcessSnapshotFailed({ message: "Windows process-parent snapshot contains ambiguous identities." })) : Effect.succeed(rows)),
  ),
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
    const details = Effect.try({ try: () => native.observedProcessDetails(handle.value), catch: error => failure("Could not read the retained Windows process identity.", error) }).pipe(
      Effect.flatMap(Schema.decodeUnknown(WindowsObservedProcess)),
      Effect.mapError(error => error instanceof WindowsProcessObservationFailed ? error : failure("Windows returned invalid process details.")),
      Effect.flatMap(details => details.pid === pid ? Effect.succeed(details) : Effect.fail(failure("The retained Windows process does not match the requested PID."))),
    )
    return Option.some({ details, exited, awaitExit: exited.pipe(Effect.repeat({ until: exited => exited, schedule: Schedule.spaced("100 millis") }), Effect.asVoid) })
  }),
})
export const windowsProcessObserverLayerFromLoader = (load: () => unknown) => Layer.unwrapEffect(Effect.try({
  try: () => {
    const native = load() as WindowsProcessObserverBindings
    if (![native.observeProcess, native.observedProcessExited, native.observedProcessDetails, native.snapshotProcessParents, native.releaseObservedProcess].every(value => typeof value === "function")) throw new Error("Process observation exports are absent")
    return windowsProcessObserverLayer(native)
  }, catch: () => new WindowsProcessObserverUnavailable({ message: "The installed native host does not provide Windows process observation." }),
}))

export const nativeWindowsProcessObserverLayer = (addonPath: string) => windowsProcessObserverLayerFromLoader(() => createRequire(import.meta.url)(addonPath))
