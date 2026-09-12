import { Context, Duration, Effect, Layer, Option, Schedule, Schema, type Scope } from "effect"
import { WindowsProcessId } from "@magnitudedev/utils/windows-native"
import { WindowsObservedProcess } from "./windows-process-observer"

export class LegacyWindowsProcessFailed extends Schema.TaggedError<LegacyWindowsProcessFailed>()("LegacyWindowsProcessFailed", {
  pid: WindowsProcessId,
  message: Schema.String,
  win32Code: Schema.optionalWith(Schema.Int, { as: "Option", exact: true }),
}) {}

export interface LegacyWindowsProcess {
  readonly exited: Effect.Effect<boolean, LegacyWindowsProcessFailed>
  readonly retire: Effect.Effect<void, LegacyWindowsProcessFailed>
}
export interface LegacyWindowsProcesses {
  /** The caller has established ancestry and persisted this exact identity before acquisition. */
  readonly acquire: (identity: WindowsObservedProcess) => Effect.Effect<LegacyWindowsProcess, LegacyWindowsProcessFailed, Scope.Scope>
}
export const LegacyWindowsProcesses = Context.GenericTag<LegacyWindowsProcesses>("@magnitudedev/daemon-management/LegacyWindowsProcesses")
export interface LegacyWindowsProcessBindings {
  readonly acquireMigrationProcess: (pid: number, creationTime: string, executable: string, userSid: string) => unknown
  readonly startMigrationProcessRetirement: (handle: object) => void
  readonly migrationProcessExited: (handle: object) => unknown
  readonly releaseMigrationProcess: (handle: object) => void
}

/** Retires one fenced process. It cannot establish ancestry, clean a tree, or admit a service. */
export const legacyWindowsProcessesLayer = (
  native: LegacyWindowsProcessBindings,
  timeout: Duration.Duration = Duration.seconds(10),
) => Layer.succeed(LegacyWindowsProcesses, {
  acquire: identity => Effect.gen(function* () {
    const failed = (message: string, cause?: unknown) => new LegacyWindowsProcessFailed({
      pid: identity.pid, message,
      win32Code: Option.fromNullable(typeof cause === "object" && cause !== null && "win32Code" in cause
        && typeof cause.win32Code === "number" && Number.isInteger(cause.win32Code) ? cause.win32Code : undefined),
    })
    const handle = yield* Effect.acquireRelease(Effect.try({
      try: () => native.acquireMigrationProcess(identity.pid, identity.creationTime, identity.executable, identity.userSid),
      catch: cause => failed("Could not acquire the exact legacy Windows process; it has not been terminated.", cause),
    }).pipe(Effect.flatMap(value => typeof value === "object" && value !== null
      ? Effect.succeed(value)
      : Effect.fail(failed("Native migration returned an invalid process capability.")))),
    handle => Effect.try({ try: () => native.releaseMigrationProcess(handle),
      catch: cause => failed("Could not release the legacy Windows process handle.", cause),
    }).pipe(Effect.orDie))
    const exited = Effect.try({ try: () => native.migrationProcessExited(handle),
      catch: cause => failed("Could not observe legacy Windows process retirement.", cause),
    }).pipe(Effect.flatMap(value => Schema.decodeUnknown(Schema.Boolean)(value).pipe(
      Effect.mapError(() => failed("Native migration returned an invalid exit observation.")),
    )))
    const retire = Effect.try({ try: () => native.startMigrationProcessRetirement(handle),
      catch: cause => failed("Could not terminate the exact legacy Windows process.", cause),
    }).pipe(Effect.zipRight(exited.pipe(
      Effect.repeat({ until: complete => complete, schedule: Schedule.spaced("25 millis") }),
      Effect.timeoutFail({ duration: timeout, onTimeout: () => failed("Legacy Windows process retirement is unproven after its deadline.") }),
      Effect.asVoid,
    )))
    return { exited, retire }
  }),
})
