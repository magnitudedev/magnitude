import { createRequire } from "node:module"
import { Context, Duration, Effect, Layer, Option, Ref, Schema } from "effect"
import * as FSM from "../fsm/index"
import { WindowsPipeName } from "./windows-pipe"
import type { encodeWindowsCommand } from "./windows-command"
import { WindowsProcessIdentity } from "./windows-process-identity"

export class WindowsJobFailed extends Schema.TaggedError<WindowsJobFailed>()("WindowsJobFailed", {
  message: Schema.String,
  win32Code: Schema.optionalWith(Schema.Int, { as: "Option", exact: true }),
}) {}
const failure = (message: string, error?: unknown) => new WindowsJobFailed({ message, win32Code: Option.fromNullable(
  typeof error === "object" && error !== null && "win32Code" in error && typeof error.win32Code === "number" && Number.isInteger(error.win32Code) ? error.win32Code : undefined,
) })
const uint32 = Schema.Int.pipe(Schema.between(0, 0xffffffff))
export interface WindowsJobBindings {
  readonly spawnOwnedProcess: (executable: string, commandLine: string, environment: string, outputName: string) => object
  readonly spawnOwnedProcessWithPipes: (executable: string, commandLine: string, environment: string, inputName: string, outputName: string, errorName: string) => object
  readonly ownedProcessIdentity: (handle: object) => unknown
  readonly ownedProcessActiveCount: (handle: object) => unknown
  readonly ownedProcessExit: (handle: object) => unknown
  readonly terminateOwnedProcess: (handle: object) => void
  readonly closeOwnedProcess: (handle: object) => void
}
export interface WindowsOwnedJob {
  readonly identity: Effect.Effect<WindowsProcessIdentity, WindowsJobFailed>
  readonly exit: Effect.Effect<number, WindowsJobFailed>
  readonly retire: (timeout: Duration.DurationInput) => Effect.Effect<void, WindowsJobFailed>
}
export interface WindowsJobOwner {
  readonly spawn: (command: Effect.Effect.Success<ReturnType<typeof encodeWindowsCommand>>, streams: WindowsJobStreams) => Effect.Effect<WindowsOwnedJob, WindowsJobFailed>
}
export const WindowsJobStreams = Schema.Union(
  Schema.TaggedStruct("Diagnostics", { output: WindowsPipeName }),
  Schema.TaggedStruct("Separate", { input: WindowsPipeName, output: WindowsPipeName, error: WindowsPipeName }),
)
export type WindowsJobStreams = typeof WindowsJobStreams.Type
export const WindowsJobOwner = Context.GenericTag<WindowsJobOwner>("@magnitudedev/utils/WindowsJobOwner")

// Opaque native capabilities and Effect references are internal only, never serialized to IPC.
const handleSchema = Schema.declare<object>((value): value is object => typeof value === "object" && value !== null)
class Owned extends Schema.TaggedClass<Owned>()("Owned", { handle: handleSchema }) {}
class Retiring extends Schema.TaggedClass<Retiring>()("Retiring", { handle: handleSchema }) {}
class Retired extends Schema.TaggedClass<Retired>()("Retired", { exitCode: uint32 }) {}
class Released extends Schema.TaggedClass<Released>()("Released", {}) {}
const jobMachine = FSM.defineFSM({ Owned, Retiring, Retired, Released }, {
  Owned: ["Retiring", "Released"], Retiring: ["Retired", "Released"], Retired: [], Released: [],
} as const)
type JobState = Owned | Retiring | Retired | Released
class Idle extends Schema.TaggedClass<Idle>()("Idle", {}) {}
class Occupied extends Schema.TaggedClass<Occupied>()("Occupied", { job: Schema.declare<Ref.Ref<JobState>>((value): value is Ref.Ref<JobState> => typeof value === "object" && value !== null && Ref.RefTypeId in value) }) {}
class Closed extends Schema.TaggedClass<Closed>()("Closed", {}) {}
const ownerMachine = FSM.defineFSM({ Idle, Occupied, Closed }, { Idle: ["Occupied", "Closed"], Occupied: ["Idle", "Closed"], Closed: [] } as const)
type OwnerState = Idle | Occupied | Closed

/** The owning process scope retains handles across failed per-attempt cleanup. */
export const windowsJobOwnerLayer = (native: WindowsJobBindings) => Layer.scoped(WindowsJobOwner, Effect.gen(function* () {
  const state = yield* Ref.make<OwnerState>(new Idle({}))
  const gate = yield* Effect.makeSemaphore(1)
  const call = <A>(operation: () => A, message: string) => Effect.try({ try: operation, catch: error => failure(message, error) })
  const decode = <A, I>(schema: Schema.Schema<A, I>, value: unknown) => Schema.decodeUnknown(schema)(value).pipe(Effect.mapError(() => failure("Windows returned an invalid owned-process observation.")))
  yield* Effect.addFinalizer(() => gate.withPermits(1)(Effect.gen(function* () {
    const owner = yield* Ref.get(state)
    if (owner._tag === "Closed") return
    if (owner._tag === "Occupied") {
      const job = yield* Ref.get(owner.job)
      if (job._tag === "Owned" || job._tag === "Retiring") {
        yield* call(() => native.closeOwnedProcess(job.handle), "Could not release the Windows job.").pipe(Effect.orDie)
        yield* Ref.set(owner.job, jobMachine.transition(job, "Released", {}))
      }
    }
    yield* Ref.set(state, ownerMachine.transition(owner, "Closed", {}))
  })))
  return WindowsJobOwner.of({ spawn: (command, streams) => gate.withPermits(1)(Effect.gen(function* () {
    const owner = yield* Ref.get(state)
    if (owner._tag !== "Idle") return yield* failure(owner._tag === "Closed" ? "The Windows process owner is closed." : "The previous Windows job has not been retired.")
    const job = yield* Ref.make<JobState>(new Owned({ handle: yield* call(
      () => streams._tag === "Diagnostics"
        ? native.spawnOwnedProcess(command.executable, command.commandLine, command.environment, streams.output)
        : native.spawnOwnedProcessWithPipes(command.executable, command.commandLine, command.environment, streams.input, streams.output, streams.error), "Could not create the owned Windows process.",
    ) }))
    yield* Ref.set(state, ownerMachine.transition(owner, "Occupied", { job }))
    const identity = gate.withPermits(1)(Effect.gen(function* () {
      const current = yield* Ref.get(job)
      if (current._tag === "Retired" || current._tag === "Released") return yield* failure("The Windows process handle has been released.")
      return yield* decode(WindowsProcessIdentity, yield* call(() => native.ownedProcessIdentity(current.handle), "Could not observe the owned Windows process identity."))
    }))
    const observeExit = gate.withPermits(1)(Effect.gen(function* () {
      const current = yield* Ref.get(job)
      if (current._tag === "Retired") return current.exitCode
      if (current._tag === "Released") return yield* failure("The Windows process owner closed before exit was observed.")
      return yield* decode(Schema.NullOr(uint32), yield* call(() => native.ownedProcessExit(current.handle), "Could not observe the owned Windows process exit."))
    }))
    const exit = Effect.gen(function* () {
      for (;;) {
        const code = yield* observeExit
        if (code !== null) return code
        yield* Effect.sleep("25 millis")
      }
    })
    const retire = (timeout: Duration.DurationInput) => gate.withPermits(1)(Effect.gen(function* () {
      const current = yield* Ref.get(job)
      if (current._tag === "Retired") return
      if (current._tag === "Released") return yield* failure("The Windows job was released without proof of retirement.")
      if (current._tag === "Owned") yield* Ref.set(job, jobMachine.transition(current, "Retiring", { handle: current.handle }))
      yield* call(() => native.terminateOwnedProcess(current.handle), "Could not terminate the owned Windows job.")
      for (;;) {
        const count = yield* decode(uint32, yield* call(() => native.ownedProcessActiveCount(current.handle), "Could not prove Windows job retirement."))
        const code = yield* decode(Schema.NullOr(uint32), yield* call(() => native.ownedProcessExit(current.handle), "Could not observe the owned Windows process exit."))
        if (count === 0 && code !== null) {
          yield* Effect.gen(function* () {
            yield* call(() => native.closeOwnedProcess(current.handle), "Could not release the retired Windows job.")
            const retiring = yield* Ref.get(job)
            if (retiring._tag !== "Retiring") return yield* Effect.die("Windows job changed while retirement held the gate")
            yield* Ref.set(job, jobMachine.transition(retiring, "Retired", { exitCode: code }))
            const occupied = yield* Ref.get(state)
            if (occupied._tag !== "Occupied" || occupied.job !== job) return yield* Effect.die("Windows process ownership changed during retirement")
            yield* Ref.set(state, ownerMachine.transition(occupied, "Idle", {}))
          }).pipe(Effect.uninterruptible)
          return
        }
        yield* Effect.sleep("25 millis")
      }
    }).pipe(Effect.timeoutFail({ duration: timeout, onTimeout: () => failure("Windows job retirement remains unproven; its native handles are retained.") })))
    return { identity, exit, retire }
  }).pipe(Effect.uninterruptible)) })
}))

export const nativeWindowsJobOwnerLayer = (addonPath: string) => Layer.unwrapEffect(Effect.try({
  try: () => {
    const native = createRequire(import.meta.url)(addonPath) as WindowsJobBindings
    if (![native.spawnOwnedProcess, native.spawnOwnedProcessWithPipes, native.ownedProcessIdentity, native.ownedProcessActiveCount, native.ownedProcessExit, native.terminateOwnedProcess, native.closeOwnedProcess].every(value => typeof value === "function")) throw new Error("Windows job exports are absent")
    return windowsJobOwnerLayer(native)
  }, catch: () => failure("The installed native host does not provide Windows process ownership."),
}))
