import type { Duplex } from "node:stream"
import { DesktopChildEvent, DesktopOwnerCommand } from "@magnitudedev/acn-protocol/desktop-control"
import { receiveJsonLines, sendJsonLine, type JsonLineChannelFailed } from "@magnitudedev/utils/json-line-channel"
import { spawn, type ChildProcess } from "node:child_process"
import {
  ProcessGroupController,
  type ExactProcess,
  type ProcessGroupStopError,
  type ExactProcessIdentityObservationFailed,
} from "@magnitudedev/utils/process-groups"
import { Context, Deferred, Effect, Option, Ref, Runtime, Schema, type Scope, type Stream } from "effect"

export class OwnedChildSpawnFailed extends Schema.TaggedError<OwnedChildSpawnFailed>()("OwnedChildSpawnFailed", {
  executable: Schema.String,
  message: Schema.String,
}) {}
export class OwnedChildIdentityLost extends Schema.TaggedError<OwnedChildIdentityLost>()("OwnedChildIdentityLost", {
  pid: Schema.Number,
}) {}
export class OwnedChildReplaced extends Schema.TaggedError<OwnedChildReplaced>()("OwnedChildReplaced", {
  pid: Schema.Number,
}) {}
export class OwnedChildPlatformUnsupported extends Schema.TaggedError<OwnedChildPlatformUnsupported>()("OwnedChildPlatformUnsupported", {
  platform: Schema.String,
}) {}
export class OwnedChildObservationFailed extends Schema.TaggedError<OwnedChildObservationFailed>()("OwnedChildObservationFailed", {
  pid: Schema.Number, message: Schema.String,
}) {}
export class OwnedChildRetirementFailed extends Schema.TaggedError<OwnedChildRetirementFailed>()("OwnedChildRetirementFailed", {
  pid: Schema.Number, message: Schema.String,
}) {}
export type OwnedChildStopError = ProcessGroupStopError | OwnedChildReplaced | OwnedChildRetirementFailed

export interface OwnedChild {
  readonly events: Stream.Stream<DesktopChildEvent, JsonLineChannelFailed>
  readonly send: (command: DesktopOwnerCommand) => Effect.Effect<void, JsonLineChannelFailed>
  readonly identity: ExactProcess
  readonly exit: Effect.Effect<number, OwnedChildObservationFailed>
  readonly diagnosticTail: Effect.Effect<string>
  readonly stop: Effect.Effect<void, OwnedChildStopError>
}
export interface OwnedChildCommand {
  readonly executable: string
  readonly arguments: ReadonlyArray<string>
  readonly environment: Readonly<Record<string, string | undefined>>
}
export interface OwnedChildSpawner {
  readonly spawn: (command: OwnedChildCommand) => Effect.Effect<
    OwnedChild,
    OwnedChildSpawnFailed | OwnedChildIdentityLost | OwnedChildPlatformUnsupported | ExactProcessIdentityObservationFailed,
    Scope.Scope
  >
}
export const OwnedChildSpawner = Context.GenericTag<OwnedChildSpawner>("@magnitudedev/daemon-management/OwnedChildSpawner")

/** Native Windows spawning is supplied separately: ordinary spawn cannot atomically join a job. */
export const makeUnixOwnedChildSpawner = Effect.gen(function* () {
  const groups = yield* ProcessGroupController
  return OwnedChildSpawner.of({
    spawn: command => Effect.gen(function* () {
      if (process.platform === "win32") return yield* new OwnedChildPlatformUnsupported({ platform: process.platform })
      const tail = yield* Ref.make("")
      const exited = yield* Deferred.make<number>()
      const runtime = yield* Effect.runtime<never>()
      const publishExit = (code: number) => Runtime.runSync(runtime)(Deferred.succeed(exited, code))
      const append = (chunk: Buffer) => Runtime.runSync(runtime)(Ref.update(tail, text => {
        const bytes = Buffer.concat([Buffer.from(text), chunk])
        return bytes.subarray(Math.max(0, bytes.length - 16_384)).toString("utf8")
      }))
      // Own the raw handle before any interruptible identity observation.
      const child = yield* Effect.acquireRelease(
        Effect.async<ChildProcess, OwnedChildSpawnFailed>(resume => {
          const child = spawn(command.executable, [...command.arguments], {
            detached: true,
            stdio: ["pipe", "pipe", "pipe", "pipe"],
            env: command.environment,
          })
          child.stderr!.on("data", append)
          child.stdout!.on("data", append)
          child.once("exit", code => publishExit(code ?? 1))
          child.once("spawn", () => resume(Effect.succeed(child)))
          child.once("error", error => {
            publishExit(1)
            resume(Effect.fail(new OwnedChildSpawnFailed({ executable: command.executable, message: error.message })))
          })
          // A child may exit before any lifetime writes. Never turn EPIPE into an uncaught exception.
          child.stdin!.on("error", () => {})
          child.stdio[3]?.on("error", () => {})
        }),
        child => Effect.sync(() => {
          child.stdin?.destroy()
          child.stdio[3]?.destroy()
          if (child.exitCode === null && child.signalCode === null) child.kill("SIGKILL")
        }),
      )
      const pid = child.pid!
      const observed = yield* groups.inspect(pid)
      if (Option.isNone(observed)) return yield* new OwnedChildIdentityLost({ pid })
      const identity = observed.value
      const stopped = yield* Ref.make(false)
      const stopLock = yield* Effect.makeSemaphore(1)
      const stop = stopLock.withPermits(1)(Effect.gen(function* () {
        if (yield* Ref.get(stopped)) return
        const outcome = yield* groups.stop({ leader: identity })
        if (outcome._tag === "ProcessGroupLeaderReplaced") return yield* new OwnedChildReplaced({ pid })
        // Closing the channel only follows complete group retirement in the graceful path.
        child.stdin?.destroy()
        yield* Ref.set(stopped, true)
      }).pipe(Effect.uninterruptible))
      yield* Effect.addFinalizer(() => stop.pipe(Effect.catchAll(error => Effect.logError("Owned child cleanup failed", error))))
      const channel = child.stdio[3] as Duplex
      return {
        identity, stop, exit: Deferred.await(exited), diagnosticTail: Ref.get(tail),
        events: receiveJsonLines(channel, DesktopChildEvent),
        send: command => sendJsonLine(channel, DesktopOwnerCommand, command),
      }
    }),
  })
})
