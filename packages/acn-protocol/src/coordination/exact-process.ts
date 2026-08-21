import { execFile } from "node:child_process"
import { readFile } from "node:fs/promises"
import { Context, Data, Effect, Option, Schema } from "effect"
import { ProcessStartIdentitySchema } from "../acn-identity"
import { ExactProcessSchema, ProcessGroupSchema } from "./schemas"
import {
  ExactProcessIdentityObservationFailed,
  ProcessGroupObservationFailed,
  ProcessGroupSignalFailed,
  ProcessGroupSignalPermissionDenied,
  type ProcessGroupSignalError,
} from "./errors"
import type { ExactProcess, ProcessGroup } from "./schemas"

export type ProcessGroupSignal = "term" | "kill"

export class ProcessGroupAbsent extends Schema.TaggedClass<ProcessGroupAbsent>()(
  "ProcessGroupAbsent",
  { group: ProcessGroupSchema },
) {}

export class ProcessGroupPresent extends Schema.TaggedClass<ProcessGroupPresent>()(
  "ProcessGroupPresent",
  { group: ProcessGroupSchema },
) {}

export type ProcessGroupObservation = ProcessGroupAbsent | ProcessGroupPresent

export class ProcessGroupSignaled extends Schema.TaggedClass<ProcessGroupSignaled>()(
  "ProcessGroupSignaled",
  { group: ProcessGroupSchema },
) {}

export class ProcessGroupAlreadyAbsent extends Schema.TaggedClass<ProcessGroupAlreadyAbsent>()(
  "ProcessGroupAlreadyAbsent",
  { group: ProcessGroupSchema },
) {}

export class ProcessGroupLeaderChanged extends Schema.TaggedClass<ProcessGroupLeaderChanged>()(
  "ProcessGroupLeaderChanged",
  { group: ProcessGroupSchema, observedLeader: ExactProcessSchema },
) {}

export type ProcessGroupSignalOutcome =
  | ProcessGroupSignaled
  | ProcessGroupAlreadyAbsent
  | ProcessGroupLeaderChanged

export interface ProcessGroupController {
  readonly inspect: (
    pid: number,
  ) => Effect.Effect<
    Option.Option<ExactProcess["processStartIdentity"]>,
    ExactProcessIdentityObservationFailed
  >
  readonly currentProcess: Effect.Effect<ExactProcess, ExactProcessIdentityObservationFailed>
  readonly observeGroup: (
    group: ProcessGroup,
  ) => Effect.Effect<ProcessGroupObservation, ProcessGroupObservationFailed>
  readonly signalGroup: (
    group: ProcessGroup,
    signal: ProcessGroupSignal,
  ) => Effect.Effect<ProcessGroupSignalOutcome, ProcessGroupSignalError>
}

export const ProcessGroupController = Context.GenericTag<ProcessGroupController>(
  "@magnitudedev/acn-protocol/coordination/ProcessGroupController",
)

class ProcessFacilityFailed extends Data.TaggedError("ProcessFacilityFailed")<{
  readonly message: string
}> {}
class ProcessCommandFailed extends Data.TaggedError("ProcessCommandFailed")<{
  readonly message: string
}> {}
class ProcessCommandPermissionDenied extends Data.TaggedError("ProcessCommandPermissionDenied")<{
  readonly message: string
}> {}
class ProcessAbsent extends Data.TaggedError("ProcessAbsent") {}
class ProcessPresent extends Data.TaggedError("ProcessPresent") {}

type ProcessCommandError =
  | ProcessCommandFailed
  | ProcessCommandPermissionDenied
  | ProcessFacilityFailed

const messageOf = (cause: unknown): string => cause instanceof Error ? cause.message : String(cause)
const isErrno = (cause: unknown, code: string): boolean =>
  cause instanceof Error && "code" in cause && cause.code === code

const commandFailure = (cause: unknown): ProcessCommandError => {
  if (isErrno(cause, "EPERM")) return new ProcessCommandPermissionDenied({ message: messageOf(cause) })
  return cause instanceof Error
    ? new ProcessCommandFailed({ message: cause.message })
    : new ProcessFacilityFailed({ message: messageOf(cause) })
}

const command = (
  executable: string,
  arguments_: readonly string[],
): Effect.Effect<string, ProcessCommandError> => Effect.async((resume) => {
  const child = execFile(executable, [...arguments_], { encoding: "utf8" }, (error, stdout) => {
    resume(error === null ? Effect.succeed(stdout) : Effect.fail(commandFailure(error)))
  })
  return Effect.sync(() => child.kill())
})

const commandOrAbsent = (
  executable: string,
  arguments_: readonly string[],
  absentExitCode: number,
): Effect.Effect<Option.Option<string>, ProcessCommandError> => Effect.async((resume) => {
  const child = execFile(executable, [...arguments_], { encoding: "utf8" }, (error, stdout) => {
    if (error === null) return resume(Effect.succeed(Option.some(stdout)))
    if (typeof error.code === "number" && error.code === absentExitCode) {
      return resume(Effect.succeed(Option.none()))
    }
    return resume(Effect.fail(commandFailure(error)))
  })
  return Effect.sync(() => child.kill())
})

const linuxIdentity = (pid: number): Effect.Effect<Option.Option<string>, ProcessFacilityFailed> =>
  Effect.tryPromise({
    try: async () => {
      const stat = await readFile(`/proc/${pid}/stat`, "utf8").catch((error: NodeJS.ErrnoException) => {
        if (error.code === "ENOENT") return null
        throw error
      })
      if (stat === null) return Option.none()
      const close = stat.lastIndexOf(")")
      if (close < 0) throw new Error("malformed proc stat")
      const startTicks = stat.slice(close + 2).trim().split(/\s+/)[19]
      if (startTicks === undefined) throw new Error("proc stat has no start time")
      const bootId = (await readFile("/proc/sys/kernel/random/boot_id", "utf8"))
        .trim()
        .toLowerCase()
      return Option.some(`linux:${bootId}:${startTicks}`)
    },
    catch: (cause) => new ProcessFacilityFailed({ message: messageOf(cause) }),
  })

const darwinIdentity = (
  pid: number,
): Effect.Effect<Option.Option<string>, ProcessCommandError> =>
  commandOrAbsent("/bin/ps", ["-o", "lstart=", "-p", String(pid)], 1).pipe(
    Effect.flatMap(Option.match({
      onNone: () => Effect.succeed(Option.none()),
      onSome: (started) => command("/usr/sbin/sysctl", ["-n", "kern.bootsessionuuid"]).pipe(
        Effect.map((boot) => Option.some(`darwin:${boot.trim().toLowerCase()}:${started.trim()}`)),
      ),
    })),
  )

const windowsIdentity = (
  pid: number,
): Effect.Effect<Option.Option<string>, ProcessCommandError> =>
  command("powershell.exe", [
    "-NoProfile",
    "-NonInteractive",
    "-Command",
    `$p = Get-Process -Id ${pid} -ErrorAction SilentlyContinue; if ($p) { $p.StartTime.ToUniversalTime().Ticks }`,
  ]).pipe(
    Effect.map((started) => started.trim()),
    Effect.map((started) => started.length === 0
      ? Option.none()
      : Option.some(`windows:${started}`)),
  )

const inspectIdentity = (
  pid: number,
): Effect.Effect<Option.Option<string>, ProcessCommandError | ProcessFacilityFailed> => {
  if (process.platform === "linux") return linuxIdentity(pid)
  if (process.platform === "darwin") return darwinIdentity(pid)
  if (process.platform === "win32") return windowsIdentity(pid)
  return Effect.fail(new ProcessFacilityFailed({
    message: `unsupported process platform ${process.platform}`,
  }))
}

const inspect = (
  pid: number,
): Effect.Effect<
  Option.Option<ExactProcess["processStartIdentity"]>,
  ExactProcessIdentityObservationFailed
> => inspectIdentity(pid).pipe(
  Effect.map(Option.map(ProcessStartIdentitySchema.make)),
  Effect.mapError((cause) => new ExactProcessIdentityObservationFailed({
    pid,
    message: cause.message,
  })),
)

const signalFailure = (
  group: ProcessGroup,
  cause: unknown,
): ProcessGroupSignalError => {
  const message = messageOf(cause)
  const permissionDenied = isErrno(cause, "EPERM") || cause instanceof ProcessCommandPermissionDenied
  return permissionDenied
    ? new ProcessGroupSignalPermissionDenied({ group, message })
    : new ProcessGroupSignalFailed({ group, message })
}

const signalGroup = (
  group: ProcessGroup,
  signal: ProcessGroupSignal,
): Effect.Effect<ProcessGroupSignalOutcome, ProcessGroupSignalError> => {
  const leader = group.leader
  if (process.platform === "win32") {
    return Effect.gen(function* () {
      const identity = yield* inspect(leader.pid)
      if (Option.isNone(identity)) return new ProcessGroupAlreadyAbsent({ group })
      if (identity.value !== leader.processStartIdentity) {
        return new ProcessGroupLeaderChanged({
          group,
          observedLeader: { pid: leader.pid, processStartIdentity: identity.value },
        })
      }
      const result = yield* command("taskkill.exe", [
        "/PID", String(leader.pid), "/T", ...(signal === "kill" ? ["/F"] : []),
      ]).pipe(Effect.either)
      if (result._tag === "Left") return yield* signalFailure(group, result.left)
      return new ProcessGroupSignaled({ group })
    })
  }
  return Effect.gen(function* () {
    const identity = yield* inspect(leader.pid)
    if (Option.isSome(identity) && identity.value !== leader.processStartIdentity) {
      return new ProcessGroupLeaderChanged({
        group,
        observedLeader: { pid: leader.pid, processStartIdentity: identity.value },
      })
    }
    const name = signal === "term" ? "SIGTERM" : "SIGKILL"
    return yield* Effect.try({
      try: () => {
        process.kill(-leader.pid, name)
        return new ProcessGroupSignaled({ group })
      },
      catch: (cause) => isErrno(cause, "ESRCH")
        ? new ProcessAbsent()
        : signalFailure(group, cause),
    }).pipe(Effect.catchTag("ProcessAbsent", () => Effect.succeed(new ProcessGroupAlreadyAbsent({ group }))))
  })
}

const observeUnixGroup = (
  group: ProcessGroup,
): Effect.Effect<ProcessGroupObservation, ProcessGroupObservationFailed> =>
  Effect.try({
    try: () => {
      process.kill(-group.leader.pid, 0)
    },
    catch: (cause) => isErrno(cause, "ESRCH")
      ? new ProcessAbsent()
      : isErrno(cause, "EPERM")
        ? new ProcessPresent()
        : new ProcessGroupObservationFailed({ group, message: messageOf(cause) }),
  }).pipe(
    Effect.as<ProcessGroupObservation>(new ProcessGroupPresent({ group })),
    Effect.catchTags({
      ProcessAbsent: () => Effect.succeed(new ProcessGroupAbsent({ group })),
      ProcessPresent: () => Effect.succeed(new ProcessGroupPresent({ group })),
    }),
  )

const observeWindowsGroup = (
  group: ProcessGroup,
): Effect.Effect<ProcessGroupObservation, ProcessGroupObservationFailed> => inspect(group.leader.pid).pipe(
  Effect.mapError((error) => new ProcessGroupObservationFailed({
    group,
    message: error.message,
  })),
  Effect.flatMap((identity) => Option.contains(identity, group.leader.processStartIdentity)
    ? Effect.succeed(new ProcessGroupPresent({ group }))
    : Effect.fail(new ProcessGroupObservationFailed({
        group,
        message: "native Windows cannot prove descendant-tree absence after the recorded root exits; use WSL",
      }))),
)

export const ProcessGroupControllerLive: ProcessGroupController = {
  inspect,
  currentProcess: inspect(process.pid).pipe(
    Effect.flatMap(Option.match({
      onNone: () => Effect.fail(new ExactProcessIdentityObservationFailed({
        pid: process.pid,
        message: "current process is absent",
      })),
      onSome: (processStartIdentity) => Effect.succeed({ pid: process.pid, processStartIdentity }),
    })),
  ),
  observeGroup: (group) => process.platform === "win32"
    ? observeWindowsGroup(group)
    : observeUnixGroup(group),
  signalGroup,
}
