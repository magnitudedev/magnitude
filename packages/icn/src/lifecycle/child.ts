import * as Command from "@effect/platform/Command"
import * as CommandExecutor from "@effect/platform/CommandExecutor"
import { ProcessGroupController } from "@magnitudedev/utils/process-groups"
import { Context, Duration, Effect, Layer, Option, Schema, Stream, type Scope } from "effect"
import { IcnProcessGroupReplaced, IcnProcessIdentityUnavailable, type IcnLifecycleError } from "./errors.js"

export class IcnChildLaunch extends Schema.Class<IcnChildLaunch>("IcnChildLaunch")({
  executable: Schema.NonEmptyString,
  arguments: Schema.Array(Schema.String),
  environment: Schema.Record({ key: Schema.String, value: Schema.String }),
  gracefulShutdownTimeout: Schema.DurationFromSelf.pipe(Schema.greaterThanDuration(Duration.zero)),
  forceShutdownTimeout: Schema.DurationFromSelf.pipe(Schema.greaterThanDuration(Duration.zero)),
}) {}

/** The platform owns exact process handles and retirement; the ICN lifecycle owns readiness. */
export interface IcnChild {
  readonly pid: number
  readonly stdout: Stream.Stream<Uint8Array, IcnLifecycleError>
  readonly stderr: Stream.Stream<Uint8Array, IcnLifecycleError>
  readonly exitCode: Effect.Effect<number, IcnLifecycleError>
  readonly terminate: Effect.Effect<void, IcnLifecycleError>
}
export interface IcnChildSpawner {
  readonly spawn: (launch: IcnChildLaunch) => Effect.Effect<IcnChild, IcnLifecycleError, Scope.Scope>
}
export const IcnChildSpawner = Context.GenericTag<IcnChildSpawner>("@magnitudedev/icn/IcnChildSpawner")

export const UnixIcnChildSpawner = Layer.effect(IcnChildSpawner, Effect.gen(function* () {
  const groups = yield* ProcessGroupController
  const executor = yield* CommandExecutor.CommandExecutor
  return IcnChildSpawner.of({ spawn: input => Effect.uninterruptible(Effect.gen(function* () {
    const launch = yield* Schema.validate(IcnChildLaunch)(input)
    const process = yield* Command.start(Command.make(launch.executable, ...launch.arguments).pipe(
      Command.env(launch.environment), Command.stdin(Stream.never),
    )).pipe(Effect.provideService(CommandExecutor.CommandExecutor, executor))
    const pid = Number(process.pid)
    const exact = yield* groups.inspect(pid)
    if (Option.isNone(exact)) return yield* new IcnProcessIdentityUnavailable({ pid })
    const terminate = yield* Effect.cached(Effect.gen(function* () {
      const outcome = yield* groups.stop({ leader: exact.value }, {
        termWait: launch.gracefulShutdownTimeout, killWait: launch.forceShutdownTimeout,
      })
      if (outcome._tag === "ProcessGroupLeaderReplaced") return yield* new IcnProcessGroupReplaced({ pid })
    }).pipe(Effect.uninterruptible))
    yield* Effect.addFinalizer(() => terminate.pipe(Effect.ignore))
    return { pid, stdout: process.stdout, stderr: process.stderr, exitCode: process.exitCode.pipe(Effect.map(Number)), terminate }
  })) })
}))
