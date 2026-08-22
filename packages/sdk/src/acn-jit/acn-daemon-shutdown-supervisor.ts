import * as HttpClient from "@effect/platform/HttpClient"
import * as HttpClientRequest from "@effect/platform/HttpClientRequest"
import {
  AcnOwnerRecordSchema,
  sameAcnOwner,
  type AcnOwnerRecord,
  type AcnOwnerStore,
  type AcnOwnerStoreError,
  type ProcessGroup,
  type ProcessGroupController,
} from "@magnitudedev/acn-protocol/coordination"
import { Context, Duration, Effect, Option, Schema } from "effect"
import {
  AcnDaemonShutdownFailed,
  AcnDaemonShutdownReasonSchema,
  AcnOwnerRecordInvalid,
  AcnOwnerRecordReadUnavailable,
  type AcnDaemonShutdownControlFailure,
  type AcnDaemonShutdownReason,
} from "./errors"

const GRACEFUL_REQUEST_TIMEOUT = Duration.seconds(2)
const GRACEFUL_STOP_WAIT = Duration.seconds(5)

export class AcnDaemonStopped extends Schema.TaggedClass<AcnDaemonStopped>()(
  "AcnDaemonStopped",
  { owner: AcnOwnerRecordSchema, reason: AcnDaemonShutdownReasonSchema },
) {}

export class AcnDaemonSuperseded extends Schema.TaggedClass<AcnDaemonSuperseded>()(
  "AcnDaemonSuperseded",
  {
    owner: AcnOwnerRecordSchema,
    reason: AcnDaemonShutdownReasonSchema,
    cause: Schema.Literal("OwnerChanged", "IdentityChanged"),
  },
) {}

export type AcnDaemonShutdownOutcome = AcnDaemonStopped | AcnDaemonSuperseded

export interface AcnDaemonShutdownSupervisor {
  readonly shutdown: (
    expected: AcnOwnerRecord,
    reason: AcnDaemonShutdownReason,
  ) => Effect.Effect<AcnDaemonShutdownOutcome, AcnDaemonShutdownFailed>
}
export const AcnDaemonShutdownSupervisor = Context.GenericTag<AcnDaemonShutdownSupervisor>(
  "@magnitudedev/sdk/AcnDaemonShutdownSupervisor",
)

const groupFrom = (owner: AcnOwnerRecord): ProcessGroup => ({
  leader: { pid: owner.pid, processStartIdentity: owner.processStartIdentity },
})

const ownerStoreFailure = (error: AcnOwnerStoreError): AcnDaemonShutdownControlFailure =>
  error._tag === "AcnProcessStoreInvalid"
    ? new AcnOwnerRecordInvalid({ path: error.path, message: error.message })
    : new AcnOwnerRecordReadUnavailable({ path: error.path, message: error.message })

/** What a revalidation found: the daemon is done with, superseded, or still ours to retire. */
type Revalidation =
  | { readonly _tag: "Stopped" }
  | { readonly _tag: "Superseded"; readonly cause: "OwnerChanged" | "IdentityChanged" }
  | { readonly _tag: "LeaderLive" }
  | { readonly _tag: "SurvivorsOnly" }

export const makeAcnDaemonShutdownSupervisor = (
  owners: AcnOwnerStore,
  processes: ProcessGroupController,
  http: HttpClient.HttpClient,
): Effect.Effect<AcnDaemonShutdownSupervisor> => Effect.gen(function* () {
  const lock = yield* Effect.makeSemaphore(1)

  const shutdown: AcnDaemonShutdownSupervisor["shutdown"] = (expected, reason) => lock.withPermits(1)(
    Effect.gen(function* () {
      const group = groupFrom(expected)
      const stopped = new AcnDaemonStopped({ owner: expected, reason })
      const superseded = (cause: "OwnerChanged" | "IdentityChanged") =>
        new AcnDaemonSuperseded({ owner: expected, reason, cause })
      const fail = (failure: AcnDaemonShutdownControlFailure) =>
        new AcnDaemonShutdownFailed({ owner: expected, reason, failure })

      /**
       * Reconfirms that the owner row still names `expected` and observes its exact group. Between
       * signals, `stop` revalidates leader identity again before every signal it delivers, and the
       * row cannot lawfully change while the group persists (admission requires absence proof).
       */
      const revalidate: Effect.Effect<Revalidation, AcnDaemonShutdownFailed> = Effect.gen(function* () {
        const current = yield* owners.current.pipe(Effect.mapError((error) => fail(ownerStoreFailure(error))))
        if (!Option.exists(current, (owner) => sameAcnOwner(owner, expected))) {
          return { _tag: "Superseded", cause: "OwnerChanged" } as const
        }
        const observed = yield* processes.observe(group).pipe(Effect.mapError(fail))
        switch (observed._tag) {
          case "ProcessGroupAbsent": return { _tag: "Stopped" } as const
          case "ProcessGroupLeaderReplaced": return { _tag: "Superseded", cause: "IdentityChanged" } as const
          case "ProcessGroupLeaderLive": return { _tag: "LeaderLive" } as const
          case "ProcessGroupSurvivorsOnly": return { _tag: "SurvivorsOnly" } as const
        }
      })

      const beforeGraceful = yield* revalidate
      if (beforeGraceful._tag === "Stopped") return stopped
      if (beforeGraceful._tag === "Superseded") return superseded(beforeGraceful.cause)

      if (beforeGraceful._tag === "LeaderLive") {
        yield* http.execute(HttpClientRequest.post(`http://127.0.0.1:${expected.port}/shutdown`)).pipe(
          Effect.timeout(GRACEFUL_REQUEST_TIMEOUT),
          Effect.ignore,
        )
        if (yield* processes.waitForGroupExit(group, GRACEFUL_STOP_WAIT).pipe(Effect.mapError(fail))) return stopped
      }

      const beforeEscalation = yield* revalidate
      if (beforeEscalation._tag === "Stopped") return stopped
      if (beforeEscalation._tag === "Superseded") return superseded(beforeEscalation.cause)

      const outcome = yield* processes.stop(group).pipe(Effect.mapError(fail))
      return outcome._tag === "ProcessGroupLeaderReplaced" ? superseded("IdentityChanged") : stopped
    }),
  )

  return AcnDaemonShutdownSupervisor.of({ shutdown })
})

export type { AcnDaemonShutdownReason } from "./errors"
