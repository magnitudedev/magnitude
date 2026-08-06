import {
  AcnIdentitySchema,
  AcnInstanceIdSchema,
  AcnReady,
  ProcessStartIdentitySchema,
} from "@magnitudedev/acn-protocol"
import { Context, Effect, Option, Schema, Stream } from "effect"
import { AcnLifecycleObservationSchema } from "./lifecycle"
import { AcnEnsuranceError, AcnEnsuranceFailed, type AcnEnsuranceError as AcnEnsuranceErrorType } from "./errors"

const PositiveSafeInteger = Schema.Number.pipe(
  Schema.int(),
  Schema.positive(),
  Schema.lessThanOrEqualTo(Number.MAX_SAFE_INTEGER),
)

export const ReadyAcnSchema = Schema.Struct({
  id: AcnInstanceIdSchema,
  identity: AcnIdentitySchema,
  url: Schema.NonEmptyString,
  pid: PositiveSafeInteger,
  processStartIdentity: ProcessStartIdentitySchema,
  lifecycle: AcnReady,
})
export type ReadyAcn = typeof ReadyAcnSchema.Type

export const AcnEnsureRequestSchema = Schema.Struct({
  minimumIdentity: AcnIdentitySchema,
})
export type AcnEnsureRequest = typeof AcnEnsureRequestSchema.Type

export const AcnEnsureEventSchema = Schema.Union(
  Schema.TaggedStruct("Observation", {
    observation: AcnLifecycleObservationSchema,
  }),
  Schema.TaggedStruct("Ready", {
    instance: ReadyAcnSchema,
  }),
)
export type AcnEnsureEvent = typeof AcnEnsureEventSchema.Type

export interface AcnEnsurer {
  readonly ensure: (
    request: AcnEnsureRequest,
  ) => Stream.Stream<AcnEnsureEvent, AcnEnsuranceErrorType>
}

export const AcnEnsurer = Context.GenericTag<AcnEnsurer>(
  "@magnitudedev/sdk/AcnEnsurer",
)

export const runAcnEnsure = (
  events: Stream.Stream<AcnEnsureEvent, AcnEnsuranceErrorType>,
): Effect.Effect<ReadyAcn, AcnEnsuranceErrorType> =>
  events.pipe(
    Stream.runFoldEffect(Option.none<ReadyAcn>(), (ready, event) => {
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

export const RemoteAcnEnsureMessageSchema = Schema.Union(
  AcnEnsureEventSchema,
  Schema.TaggedStruct("Failed", { error: AcnEnsuranceError }),
)
export type RemoteAcnEnsureMessage = typeof RemoteAcnEnsureMessageSchema.Type
