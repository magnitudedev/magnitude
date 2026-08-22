import { execFile } from "node:child_process"
import { readFile } from "node:fs/promises"
import { Clock, Context, Data, Duration, Effect, Option, Schema } from "effect"
import { ProcessStartIdentitySchema } from "../acn-identity"
import { ExactProcessSchema, ProcessGroupSchema } from "./schemas"
import {
  ExactProcessIdentityObservationFailed,
  ProcessGroupAbsenceUnproven,
  ProcessGroupObservationFailed,
  ProcessGroupSignalFailed,
  ProcessGroupSignalPermissionDenied,
  type ProcessGroupSignalError,
  type ProcessGroupStopError,
} from "./errors"
import type { ExactProcess, ProcessGroup } from "./schemas"

/** Shared timing for process-group retirement. */
export const PROCESS_GROUP_EXIT_POLL_INTERVAL = Duration.millis(50)
export const PROCESS_GROUP_TERM_WAIT = Duration.seconds(2)
export const PROCESS_GROUP_KILL_WAIT = Duration.seconds(2)

/** The recorded leader occurrence is alive. */
export class ProcessGroupLeaderLive extends Schema.TaggedClass<ProcessGroupLeaderLive>()(
  "ProcessGroupLeaderLive",
  { group: ProcessGroupSchema },
) {}
/** The leader pid is occupied by a different process occurrence; the recorded group may still have members. */
export class ProcessGroupLeaderReplaced extends Schema.TaggedClass<ProcessGroupLeaderReplaced>()(
  "ProcessGroupLeaderReplaced",
  { group: ProcessGroupSchema, observedLeader: ExactProcessSchema },
) {}
/** The leader occurrence is gone but descendants of the group remain. */
export class ProcessGroupSurvivorsOnly extends Schema.TaggedClass<ProcessGroupSurvivorsOnly>()(
  "ProcessGroupSurvivorsOnly",
  { group: ProcessGroupSchema },
) {}
/** No member of the group remains. */
export class ProcessGroupAbsent extends Schema.TaggedClass<ProcessGroupAbsent>()(
  "ProcessGroupAbsent",
  { group: ProcessGroupSchema },
) {}
export type ProcessGroupObservation =
  | ProcessGroupLeaderLive
  | ProcessGroupLeaderReplaced
  | ProcessGroupSurvivorsOnly
  | ProcessGroupAbsent
export type ProcessGroupObservationError =
  | ExactProcessIdentityObservationFailed
  | ProcessGroupObservationFailed

export class ProcessGroupStopped extends Schema.TaggedClass<ProcessGroupStopped>()(
  "ProcessGroupStopped",
  { group: ProcessGroupSchema },
) {}
export type ProcessGroupStopOutcome = ProcessGroupStopped | ProcessGroupLeaderReplaced

export interface ProcessGroupController {
  /** Detects the exact process currently occupying `pid`, or none if the pid is free. */
  readonly inspect: (
    pid: number,
  ) => Effect.Effect<Option.Option<ExactProcess>, ExactProcessIdentityObservationFailed>
  /** This process's own exact identity. */
  readonly currentProcess: Effect.Effect<ExactProcess, ExactProcessIdentityObservationFailed>
  /** The state of the group led by that exact process occurrence. */
  readonly observe: (
    group: ProcessGroup,
  ) => Effect.Effect<ProcessGroupObservation, ProcessGroupObservationError>
  /** Polls until no member of the group remains or the deadline passes; `true` when it is gone. */
  readonly waitForGroupExit: (
    group: ProcessGroup,
    timeout: Duration.DurationInput,
  ) => Effect.Effect<boolean, ProcessGroupObservationFailed>
  /**
   * Retires the group: TERM, wait, KILL, wait, prove absence. The leader's exact identity is
   * checked before every signal; a replaced leader ends the retirement as
   * `ProcessGroupLeaderReplaced` because the recorded group can no longer be targeted safely.
   */
  readonly stop: (
    group: ProcessGroup,
  ) => Effect.Effect<ProcessGroupStopOutcome, ProcessGroupStopError>
}

export const ProcessGroupController = Context.GenericTag<ProcessGroupController>(
  "@magnitudedev/acn-protocol/coordination/ProcessGroupController",
)

type ProcessGroupSignal = "term" | "kill"

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

type SignalDelivery =
  | { readonly _tag: "Signaled" }
  | { readonly _tag: "AlreadyAbsent" }
  | ProcessGroupLeaderReplaced

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

const platformIdentity = (
  pid: number,
): Effect.Effect<Option.Option<string>, ProcessCommandError | ProcessFacilityFailed> => {
  if (process.platform === "linux") return linuxIdentity(pid)
  if (process.platform === "darwin") return darwinIdentity(pid)
  if (process.platform === "win32") return windowsIdentity(pid)
  return Effect.fail(new ProcessFacilityFailed({
    message: `unsupported process platform ${process.platform}`,
  }))
}

const inspect: ProcessGroupController["inspect"] = (pid) => platformIdentity(pid).pipe(
  Effect.map(Option.map((identity): ExactProcess => ({
    pid,
    processStartIdentity: ProcessStartIdentitySchema.make(identity),
  }))),
  Effect.mapError((cause) => new ExactProcessIdentityObservationFailed({
    pid,
    message: cause.message,
  })),
)

/** Whether anything still answers to the group id — without distinguishing whose processes they are. */
const unixMembersPresent = (
  group: ProcessGroup,
): Effect.Effect<boolean, ProcessGroupObservationFailed> =>
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
    Effect.as(true),
    Effect.catchTags({
      ProcessAbsent: () => Effect.succeed(false),
      ProcessPresent: () => Effect.succeed(true),
    }),
  )

const windowsMembersPresent = (
  group: ProcessGroup,
): Effect.Effect<boolean, ProcessGroupObservationFailed> => inspect(group.leader.pid).pipe(
  Effect.mapError((error) => new ProcessGroupObservationFailed({
    group,
    message: error.message,
  })),
  Effect.flatMap((occupant) =>
    Option.exists(occupant, (found) => found.processStartIdentity === group.leader.processStartIdentity)
      ? Effect.succeed(true)
      : Effect.fail(new ProcessGroupObservationFailed({
          group,
          message: "native Windows cannot prove descendant-tree absence after the recorded root exits; use WSL",
        }))),
)

const membersPresent = (group: ProcessGroup): Effect.Effect<boolean, ProcessGroupObservationFailed> =>
  process.platform === "win32" ? windowsMembersPresent(group) : unixMembersPresent(group)

const observe: ProcessGroupController["observe"] = (group) => Effect.gen(function* () {
  const occupant = yield* inspect(group.leader.pid)
  if (Option.isSome(occupant) && occupant.value.processStartIdentity === group.leader.processStartIdentity) {
    return new ProcessGroupLeaderLive({ group })
  }
  if (!(yield* membersPresent(group))) return new ProcessGroupAbsent({ group })
  return Option.match(occupant, {
    onNone: () => new ProcessGroupSurvivorsOnly({ group }),
    onSome: (observedLeader) => new ProcessGroupLeaderReplaced({ group, observedLeader }),
  })
})

const waitForGroupExit: ProcessGroupController["waitForGroupExit"] = (group, timeout) => Effect.gen(function* () {
  const deadline = (yield* Clock.currentTimeMillis) + Duration.toMillis(Duration.decode(timeout))
  while ((yield* Clock.currentTimeMillis) < deadline) {
    if (!(yield* membersPresent(group))) return true
    yield* Effect.sleep(PROCESS_GROUP_EXIT_POLL_INTERVAL)
  }
  return !(yield* membersPresent(group))
})

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

/** Delivers one signal to the group, refusing if the leader pid now names a different process occurrence. */
const signalGroup = (
  group: ProcessGroup,
  signal: ProcessGroupSignal,
): Effect.Effect<SignalDelivery, ProcessGroupSignalError> => Effect.gen(function* () {
  const leader = group.leader
  const occupant = yield* inspect(leader.pid)
  if (Option.isSome(occupant) && occupant.value.processStartIdentity !== leader.processStartIdentity) {
    return new ProcessGroupLeaderReplaced({ group, observedLeader: occupant.value })
  }
  if (process.platform === "win32") {
    if (Option.isNone(occupant)) return { _tag: "AlreadyAbsent" } as const
    const result = yield* command("taskkill.exe", [
      "/PID", String(leader.pid), "/T", ...(signal === "kill" ? ["/F"] : []),
    ]).pipe(Effect.either)
    if (result._tag === "Left") return yield* signalFailure(group, result.left)
    return { _tag: "Signaled" } as const
  }
  const name = signal === "term" ? "SIGTERM" : "SIGKILL"
  return yield* Effect.try({
    try: (): SignalDelivery => {
      process.kill(-leader.pid, name)
      return { _tag: "Signaled" }
    },
    catch: (cause) => isErrno(cause, "ESRCH")
      ? new ProcessAbsent()
      : signalFailure(group, cause),
  }).pipe(Effect.catchTag("ProcessAbsent", () => Effect.succeed<SignalDelivery>({ _tag: "AlreadyAbsent" })))
})

const stop: ProcessGroupController["stop"] = (group) => Effect.gen(function* () {
  const stopped = new ProcessGroupStopped({ group })
  const initial = yield* observe(group)
  if (initial._tag === "ProcessGroupAbsent") return stopped
  if (initial._tag === "ProcessGroupLeaderReplaced") return initial

  /** Signals once and waits; `none` means the group is still present after the wait. */
  const escalate = (
    signal: ProcessGroupSignal,
    wait: Duration.Duration,
  ): Effect.Effect<Option.Option<ProcessGroupStopOutcome>, ProcessGroupStopError> =>
    Effect.gen(function* () {
      const delivery = yield* signalGroup(group, signal)
      if (delivery._tag === "ProcessGroupLeaderReplaced") return Option.some(delivery)
      if (delivery._tag === "AlreadyAbsent") return Option.some(stopped)
      return (yield* waitForGroupExit(group, wait)) ? Option.some(stopped) : Option.none()
    })

  const afterTerm = yield* escalate("term", PROCESS_GROUP_TERM_WAIT)
  if (Option.isSome(afterTerm)) return afterTerm.value
  const afterKill = yield* escalate("kill", PROCESS_GROUP_KILL_WAIT)
  if (Option.isSome(afterKill)) return afterKill.value
  return yield* new ProcessGroupAbsenceUnproven({ group })
})

export const ProcessGroupControllerLive: ProcessGroupController = {
  inspect,
  currentProcess: inspect(process.pid).pipe(
    Effect.flatMap(Option.match({
      onNone: () => Effect.fail(new ExactProcessIdentityObservationFailed({
        pid: process.pid,
        message: "current process is absent",
      })),
      onSome: Effect.succeed,
    })),
  ),
  observe,
  waitForGroupExit,
  stop,
}
