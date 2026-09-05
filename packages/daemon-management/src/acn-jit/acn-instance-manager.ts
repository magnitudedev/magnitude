import {
  AcnTargetSchema,
  ServiceStartProgressSchema,
  AcnReady,
  AcnReadyInstanceSchema,
  type AcnInstance,
  type AcnTarget,
} from "@magnitudedev/acn-protocol"
import { Context, Duration, Effect, Option, Schema, Stream } from "effect"
import {
  AcnAdministrationFailed,
  AcnEnsuranceFailed,
  type AcnEnsuranceError as AcnEnsuranceErrorType,
} from "./errors"

type ReadyInstance = AcnInstance<AcnReady>

export const ACN_ENSURE_TIMEOUT = Duration.minutes(10)

export const AcnEnsureRequestSchema = Schema.Struct({ target: AcnTargetSchema })
export type AcnEnsureRequest = typeof AcnEnsureRequestSchema.Type

export const AcnEnsureEventSchema = Schema.Union(
  Schema.TaggedStruct("Observation", { observation: ServiceStartProgressSchema }),
  Schema.TaggedStruct("Ready", { instance: AcnReadyInstanceSchema }),
)
export type AcnEnsureEvent = typeof AcnEnsureEventSchema.Type

export interface AcnInstanceManager {
  readonly ensure: (
    request: AcnEnsureRequest,
  ) => Stream.Stream<AcnEnsureEvent, AcnEnsuranceErrorType>
  readonly stop: Effect.Effect<void, AcnAdministrationFailed>
}

export const AcnInstanceManager = Context.GenericTag<AcnInstanceManager>(
  "@magnitudedev/daemon-management/AcnInstanceManager",
)

export const runAcnEnsure = (
  events: Stream.Stream<AcnEnsureEvent, AcnEnsuranceErrorType>,
): Effect.Effect<ReadyInstance, AcnEnsuranceErrorType> =>
  events.pipe(
    Stream.runFoldEffect(Option.none<ReadyInstance>(), (ready, event) => {
      if (event._tag === "Observation") {
        return Option.isSome(ready)
          ? Effect.fail(new AcnEnsuranceFailed({ reason: "ACN ensure emitted progress after readiness" }))
          : Effect.succeed(ready)
      }
      return Option.isSome(ready)
        ? Effect.fail(new AcnEnsuranceFailed({ reason: "ACN ensure emitted readiness more than once" }))
        : Effect.succeed(Option.some(event.instance))
    }),
    Effect.flatMap(Option.match({
      onNone: () => Effect.fail(new AcnEnsuranceFailed({ reason: "ACN ensure ended without readiness" })),
      onSome: Effect.succeed,
    })),
  )

export type { AcnInstance, AcnTarget }
export { AcnReady, AcnReadyInstanceSchema }
