import { execFile } from "node:child_process"
import { readFile } from "node:fs/promises"
import { Context, Data, Effect, Option } from "effect"
import { ProcessStartIdentitySchema } from "../acn-identity"
import { ExactProcessInspectionFailed } from "./errors"
import type { ExactProcess } from "./schemas"

export type ExactProcessSignal = "term" | "kill"

export interface ExactProcessController {
  readonly inspect: (
    pid: number,
  ) => Effect.Effect<Option.Option<ExactProcess["processStartIdentity"]>, ExactProcessInspectionFailed>
  readonly current: Effect.Effect<ExactProcess, ExactProcessInspectionFailed>
  readonly signal: (
    process: ExactProcess,
    signal: ExactProcessSignal,
  ) => Effect.Effect<boolean, ExactProcessInspectionFailed>
  readonly signalTree: (
    process: ExactProcess,
    signal: ExactProcessSignal,
  ) => Effect.Effect<boolean, ExactProcessInspectionFailed>
  readonly treeAbsent: (
    process: ExactProcess,
  ) => Effect.Effect<boolean, ExactProcessInspectionFailed>
}

export const ExactProcessController = Context.GenericTag<ExactProcessController>(
  "@magnitudedev/acn-protocol/coordination/ExactProcessController",
)

const failed = (pid: number, operation: string, cause: unknown) =>
  new ExactProcessInspectionFailed({ pid, operation, message: String(cause) })

class ProcessFacilityFailed extends Data.TaggedError("ProcessFacilityFailed")<{
  readonly message: string
  readonly code: string | undefined
}> {}

class ProcessAbsent extends Data.TaggedError("ProcessAbsent") {}

const facilityFailure = (cause: unknown): ProcessFacilityFailed =>
  new ProcessFacilityFailed({
    message: cause instanceof Error ? cause.message : String(cause),
    code: cause instanceof Error && "code" in cause &&
        (typeof cause.code === "string" || typeof cause.code === "number")
      ? String(cause.code)
      : undefined,
  })

const command = (
  executable: string,
  arguments_: readonly string[],
): Effect.Effect<string, ProcessFacilityFailed> => Effect.async((resume) => {
  execFile(executable, [...arguments_], { encoding: "utf8" }, (error, stdout) => {
    resume(error === null ? Effect.succeed(stdout) : Effect.fail(facilityFailure(error)))
  })
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
    catch: facilityFailure,
  })

const darwinIdentity = (pid: number): Effect.Effect<Option.Option<string>, ProcessFacilityFailed> =>
  command("/bin/ps", ["-o", "lstart=", "-p", String(pid)]).pipe(
    Effect.map((started) => started.trim()),
    Effect.flatMap((started) => started.length === 0
      ? Effect.succeed(Option.none())
      : command("/usr/sbin/sysctl", ["-n", "kern.bootsessionuuid"]).pipe(
          Effect.map((boot) => Option.some(`darwin:${boot.trim().toLowerCase()}:${started}`)),
        )),
    Effect.catchAll((error) => error.code === "1"
      ? Effect.succeed(Option.none())
      : Effect.fail(error)),
  )

const windowsIdentity = (pid: number): Effect.Effect<Option.Option<string>, ProcessFacilityFailed> =>
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

const inspectIdentity = (pid: number): Effect.Effect<Option.Option<string>, ProcessFacilityFailed> => {
  if (process.platform === "linux") return linuxIdentity(pid)
  if (process.platform === "darwin") return darwinIdentity(pid)
  if (process.platform === "win32") return windowsIdentity(pid)
  return Effect.fail(new ProcessFacilityFailed({
    message: `unsupported process platform ${process.platform}`,
    code: undefined,
  }))
}

const inspect = (
  pid: number,
): Effect.Effect<Option.Option<ExactProcess["processStartIdentity"]>, ExactProcessInspectionFailed> =>
  inspectIdentity(pid).pipe(
    Effect.map(Option.map(ProcessStartIdentitySchema.make)),
    Effect.mapError((cause) => failed(pid, "inspect", cause)),
  )

const isErrno = (cause: unknown, code: string): boolean =>
  cause instanceof Error && "code" in cause && cause.code === code

const send = (
  process_: ExactProcess,
  signal: ExactProcessSignal,
  tree: boolean,
): Effect.Effect<boolean, ExactProcessInspectionFailed> => Effect.gen(function* () {
  const identity = yield* inspect(process_.pid)
  if (!Option.contains(identity, process_.processStartIdentity)) return false
  const name = signal === "term" ? "SIGTERM" : "SIGKILL"
  if (process.platform === "win32" && tree) {
    yield* command("taskkill.exe", ["/PID", String(process_.pid), "/T", ...(signal === "kill" ? ["/F"] : [])])
      .pipe(Effect.catchAll((cause) => isErrno(cause, "ESRCH") ? Effect.void : Effect.fail(cause)))
    return true
  }
  return yield* Effect.try({
    try: () => {
      process.kill(tree ? -process_.pid : process_.pid, name)
      return true
    },
    catch: (cause) => isErrno(cause, "ESRCH")
      ? false
      : failed(process_.pid, tree ? `signal-tree-${signal}` : `signal-${signal}`, cause),
  })
}).pipe(Effect.mapError((cause) => cause instanceof ExactProcessInspectionFailed
  ? cause
  : failed(process_.pid, tree ? `signal-tree-${signal}` : `signal-${signal}`, cause)))

const unixTreeAbsent = (pid: number): Effect.Effect<boolean, ExactProcessInspectionFailed> =>
  Effect.try({
    try: () => {
      process.kill(-pid, 0)
    },
    catch: (cause) => isErrno(cause, "ESRCH")
      ? new ProcessAbsent()
      : failed(pid, "inspect-tree", cause),
  }).pipe(
    Effect.as(false),
    Effect.catchTag("ProcessAbsent", () => Effect.succeed(true)),
  )

export const ExactProcessControllerLive: ExactProcessController = {
  inspect,
  current: inspect(process.pid).pipe(
    Effect.flatMap(Option.match({
      onNone: () => Effect.fail(failed(process.pid, "inspect-current", "current process is absent")),
      onSome: (processStartIdentity) => Effect.succeed({
        pid: process.pid,
        processStartIdentity,
      }),
    })),
  ),
  signal: (process_, signal) => send(process_, signal, false),
  signalTree: (process_, signal) => send(process_, signal, true),
  treeAbsent: (process_) => process.platform === "win32"
    ? inspect(process_.pid).pipe(Effect.map((identity) =>
        !Option.contains(identity, process_.processStartIdentity)))
    : unixTreeAbsent(process_.pid),
}
