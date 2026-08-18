import {
  sameAcnOwner,
  type AcnOwnerRecord,
  type AcnOwnerStore,
} from "@magnitudedev/acn-protocol/coordination"
import { Cause, Duration, Effect, Option, Scope } from "effect"
import type { AcnServiceLifecycleApi } from "./service-lifecycle"

const DEFAULT_OWNERSHIP_CHECK_INTERVAL = Duration.seconds(1)

export interface AcnOwnershipMonitorOptions {
  readonly pollInterval?: Duration.DurationInput
}

/** Installs the admitted ACN's mandatory lifetime ownership monitor. */
export const installAcnOwnershipMonitor = (
  owners: AcnOwnerStore,
  admittedOwner: AcnOwnerRecord,
  lifecycle: AcnServiceLifecycleApi,
  options: AcnOwnershipMonitorOptions = {},
): Effect.Effect<void, never, Scope.Scope> => {
  const pollInterval = options.pollInterval ?? DEFAULT_OWNERSHIP_CHECK_INTERVAL

  const monitor = Effect.gen(function* () {
    while (true) {
      const current = yield* owners.current
      if (!Option.exists(current, (owner) => sameAcnOwner(owner, admittedOwner))) {
        yield* lifecycle.beginStopping({
          reason: "ownership-lost",
          detail: "ACN owner record no longer identifies this process",
        })
        return yield* Effect.never
      }
      yield* Effect.sleep(pollInterval)
    }
  }).pipe(
    Effect.onError((cause) => Cause.isInterruptedOnly(cause)
      ? Effect.void
      : lifecycle.beginStopping({
          reason: "fatal",
          detail: "ACN could not continue monitoring ownership",
        }).pipe(
          Effect.zipRight(Effect.logError("ACN ownership monitor failed", cause)),
        )),
  )

  return monitor.pipe(Effect.forkScoped, Effect.asVoid)
}
