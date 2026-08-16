import { Cause, Duration, Effect, Schedule, Stream } from "effect"
import type { AcnRpcClient, MirroredStateInvalidation } from "@magnitudedev/sdk"

const reconnectSchedule = Schedule.exponential("100 millis").pipe(
  Schedule.modifyDelay((_, delay) => Duration.min(delay, Duration.seconds(5))),
  Schedule.jittered,
)

/** Runs one retrying ACN invalidation subscription without choosing a client cache. */
export const runMirroredStateInvalidationWatch = (
  rpc: AcnRpcClient,
  connected: () => Effect.Effect<void>,
  invalidate: (event: MirroredStateInvalidation) => Effect.Effect<void>,
): Effect.Effect<void> => Stream.unwrap(Effect.gen(function* () {
  const stream = rpc("WatchMirroredStates", {})
  yield* Effect.logDebug("Mirrored state watch connected")
  yield* connected()
  return stream.pipe(Stream.tap(invalidate))
})).pipe(
  Stream.tapErrorCause((cause) => Cause.isInterruptedOnly(cause)
    ? Effect.void
    : Effect.logWarning("Mirrored state watch disconnected; retrying").pipe(
      Effect.annotateLogs({ cause: Cause.pretty(cause).slice(0, 1_000) }),
    )),
  Stream.retry(reconnectSchedule),
  Stream.runDrain,
  Effect.catchAllCause((cause) => Cause.isInterruptedOnly(cause)
    ? Effect.void
    : Effect.logError(Cause.pretty(cause))),
)
