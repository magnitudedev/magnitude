import * as HttpClient from "@effect/platform/HttpClient"
import * as HttpClientError from "@effect/platform/HttpClientError"
import * as HttpClientRequest from "@effect/platform/HttpClientRequest"
import {
  AcnHealthResponseSchema,
  AcnReady,
  type AcnInstance,
} from "@magnitudedev/acn-protocol"
import {
  AcnOwnerRecordSchema,
  sameAcnOwner,
  type ExactProcess,
  type ExactProcessIdentityObservationFailed,
  type ProcessGroupObservation,
  type ProcessGroupObservationFailed,
  type AcnOwnerRecord,
  type AcnOwnerStore,
  type ProcessGroup,
  type ProcessGroupController,
} from "@magnitudedev/acn-protocol/coordination"
import { Context, Duration, Effect, Either, Option, Schedule, Schema } from "effect"
import {
  AcnHealthAttemptFailureSchema,
  type AcnHealthAttemptFailure,
  AcnHealthAttemptTimedOut,
  AcnHealthRequestFailed,
  AcnHealthResponseInvalid,
  AcnOwnerRecordInvalid,
  AcnOwnerRecordReadUnavailable,
  AcnProcessIdentityObservationTimedOut,
} from "./errors"

const HEALTH_TIMEOUT = Duration.seconds(2)
const PROCESS_INSPECTION_RETRY_INTERVAL = Duration.seconds(1)
const PROCESS_OPERATION_TIMEOUT = Duration.seconds(30)

export const AcnHealthObservationSchema = Schema.Struct({
  status: Schema.Number,
  health: AcnHealthResponseSchema,
})
export type AcnHealthObservation = typeof AcnHealthObservationSchema.Type

export class AcnRecordedOwnerAbsent extends Schema.TaggedClass<AcnRecordedOwnerAbsent>()(
  "AcnRecordedOwnerAbsent",
  { expectedOwner: Schema.OptionFromSelf(AcnOwnerRecordSchema) },
) {}

export class AcnRecordedOwnerProcessGroupSurvives extends Schema.TaggedClass<AcnRecordedOwnerProcessGroupSurvives>()(
  "AcnRecordedOwnerProcessGroupSurvives",
  { owner: AcnOwnerRecordSchema },
) {}

export class AcnRecordedOwnerLiveWithoutHealth extends Schema.TaggedClass<AcnRecordedOwnerLiveWithoutHealth>()(
  "AcnRecordedOwnerLiveWithoutHealth",
  {
    owner: AcnOwnerRecordSchema,
    attempts: Schema.Tuple(AcnHealthAttemptFailureSchema, AcnHealthAttemptFailureSchema),
  },
) {}

export class AcnRecordedOwnerLiveWithHealth extends Schema.TaggedClass<AcnRecordedOwnerLiveWithHealth>()(
  "AcnRecordedOwnerLiveWithHealth",
  {
    owner: AcnOwnerRecordSchema,
    health: AcnHealthObservationSchema,
  },
) {}

export const AcnOwnerObservationSchema = Schema.Union(
  AcnRecordedOwnerAbsent,
  AcnRecordedOwnerProcessGroupSurvives,
  AcnRecordedOwnerLiveWithoutHealth,
  AcnRecordedOwnerLiveWithHealth,
)
export type AcnOwnerObservation = typeof AcnOwnerObservationSchema.Type

const groupFrom = (owner: AcnOwnerRecord): ProcessGroup => ({
  leader: { pid: owner.pid, processStartIdentity: owner.processStartIdentity },
})

/** Retries transient process-observation failures for a bounded time before failing typed. */
const boundedObservation = <A, E>(
  observation: Effect.Effect<A, E>,
  pid: number,
): Effect.Effect<A, E | AcnProcessIdentityObservationTimedOut> =>
  observation.pipe(
    Effect.retry(Schedule.spaced(PROCESS_INSPECTION_RETRY_INTERVAL)),
    Effect.timeoutFail({
      duration: PROCESS_OPERATION_TIMEOUT,
      onTimeout: () => new AcnProcessIdentityObservationTimedOut({ pid }),
    }),
  )

export const inspectExactProcess = (
  processes: ProcessGroupController,
  pid: number,
): Effect.Effect<
  Option.Option<ExactProcess>,
  ExactProcessIdentityObservationFailed | AcnProcessIdentityObservationTimedOut
> => boundedObservation(processes.inspect(pid), pid)

const observeOwnerGroup = (
  processes: ProcessGroupController,
  owner: AcnOwnerRecord,
): Effect.Effect<
  ProcessGroupObservation,
  ExactProcessIdentityObservationFailed | ProcessGroupObservationFailed | AcnProcessIdentityObservationTimedOut
> => boundedObservation(processes.observe(groupFrom(owner)), owner.pid)

export type AcnOwnerObservationError =
  | AcnOwnerRecordReadUnavailable
  | AcnOwnerRecordInvalid
  | ExactProcessIdentityObservationFailed
  | AcnProcessIdentityObservationTimedOut
  | ProcessGroupObservationFailed

export interface AcnOwnerObserver {
  readonly observe: Effect.Effect<AcnOwnerObservation, AcnOwnerObservationError>
  readonly confirmReady: (
    owner: AcnOwnerRecord,
    observed: AcnHealthObservation,
  ) => Effect.Effect<Option.Option<AcnInstance<AcnReady>>, AcnOwnerObservationError>
}

export const AcnOwnerObserver = Context.GenericTag<AcnOwnerObserver>(
  "@magnitudedev/sdk/AcnOwnerObserver",
)

export const makeAcnOwnerObserver = (
  owners: AcnOwnerStore,
  processes: ProcessGroupController,
  http: HttpClient.HttpClient,
): AcnOwnerObserver => {
  const readCurrentOwner = owners.current.pipe(
    Effect.mapError((error) => error._tag === "AcnProcessStoreInvalid"
      ? new AcnOwnerRecordInvalid({ path: error.path, message: error.message })
      : new AcnOwnerRecordReadUnavailable({ path: error.path, message: error.message })),
  )

  const diagnosticMessage = (
    error: { readonly message: string; readonly cause?: unknown },
  ): string => {
    const causeMessage = error.cause instanceof Error ? error.cause.message.trim() : ""
    return causeMessage.length > 0 && !error.message.includes(causeMessage)
      ? `${error.message}: ${causeMessage}`
      : error.message
  }

  const mapHttpFailure = (
    error: HttpClientError.HttpClientError,
  ): AcnHealthAttemptFailure => error._tag === "RequestError"
    ? new AcnHealthRequestFailed({ message: diagnosticMessage(error) })
    : new AcnHealthResponseInvalid({ message: diagnosticMessage(error) })

  const healthAttempt = (
    owner: AcnOwnerRecord,
  ): Effect.Effect<AcnHealthObservation, AcnHealthAttemptFailure> =>
    http.execute(HttpClientRequest.get(`http://127.0.0.1:${owner.port}/health`)).pipe(
      Effect.mapError(mapHttpFailure),
      Effect.flatMap((response) => response.json.pipe(
        Effect.mapError(mapHttpFailure),
        Effect.flatMap((body) => Schema.decodeUnknown(AcnHealthResponseSchema)(body).pipe(
          Effect.mapError((error) => new AcnHealthResponseInvalid({ message: diagnosticMessage(error) })),
        )),
        Effect.map((health) => ({ status: response.status, health })),
      )),
      Effect.timeoutFail({
        duration: HEALTH_TIMEOUT,
        onTimeout: () => new AcnHealthAttemptTimedOut({}),
      }),
    )

  const probeHealth = (
    owner: AcnOwnerRecord,
  ): Effect.Effect<Either.Either<
    AcnHealthObservation,
    readonly [AcnHealthAttemptFailure, AcnHealthAttemptFailure]
  >> =>
    Effect.gen(function* () {
      const first = yield* Effect.either(healthAttempt(owner))
      if (Either.isRight(first)) return Either.right(first.right)
      const second = yield* Effect.either(healthAttempt(owner))
      return Either.isRight(second)
        ? Either.right(second.right)
        : Either.left([first.left, second.left] as const)
    })

  const observe = Effect.gen(function* () {
    const current = yield* readCurrentOwner
    if (Option.isNone(current)) {
      return new AcnRecordedOwnerAbsent({ expectedOwner: Option.none() })
    }
    const owner = current.value
    const group = yield* observeOwnerGroup(processes, owner)
    switch (group._tag) {
      case "ProcessGroupAbsent":
        return new AcnRecordedOwnerAbsent({ expectedOwner: Option.some(owner) })
      case "ProcessGroupSurvivorsOnly":
      case "ProcessGroupLeaderReplaced":
        return new AcnRecordedOwnerProcessGroupSurvives({ owner })
      case "ProcessGroupLeaderLive":
        break
    }
    return Either.match(yield* probeHealth(owner), {
      onLeft: (attempts) => new AcnRecordedOwnerLiveWithoutHealth({ owner, attempts }),
      onRight: (health) => new AcnRecordedOwnerLiveWithHealth({ owner, health }),
    })
  })

  const confirmReady: AcnOwnerObserver["confirmReady"] = (owner, observed) => Effect.gen(function* () {
    const { health, status } = observed
    if (status !== 200 || health.state._tag !== "Ready") return Option.none()
    const confirmedOwner = yield* readCurrentOwner
    if (!Option.exists(confirmedOwner, (current) => sameAcnOwner(current, owner))) return Option.none()
    if ((yield* observeOwnerGroup(processes, owner))._tag !== "ProcessGroupLeaderLive") return Option.none()
    return Option.some({
      revision: health.revision,
      id: health.id,
      identity: health.version,
      url: `http://127.0.0.1:${owner.port}`,
      pid: owner.pid,
      processStartIdentity: owner.processStartIdentity,
      lifecycle: new AcnReady({}),
    })
  })

  return AcnOwnerObserver.of({ observe, confirmReady })
}
