import * as HttpClient from "@effect/platform/HttpClient"
import * as HttpClientRequest from "@effect/platform/HttpClientRequest"
import {
  AcnHealthResponseSchema,
  AcnReady,
  type AcnInstance,
} from "@magnitudedev/acn-protocol"
import {
  AcnOwnerRecordSchema,
  sameAcnOwner,
  type ExactProcessIdentityObservationFailed,
  type ProcessGroupObservationFailed,
  type AcnOwnerRecord,
  type AcnOwnerStore,
  type ProcessGroup,
  type ProcessGroupController,
} from "@magnitudedev/acn-protocol/coordination"
import { Context, Duration, Effect, Option, Schedule, Schema } from "effect"
import {
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
  { owner: AcnOwnerRecordSchema },
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

export const inspectExactProcess = (
  processes: ProcessGroupController,
  pid: number,
): Effect.Effect<
  Option.Option<AcnOwnerRecord["processStartIdentity"]>,
  ExactProcessIdentityObservationFailed | AcnProcessIdentityObservationTimedOut
> =>
  processes.inspect(pid).pipe(
    Effect.retry(Schedule.spaced(PROCESS_INSPECTION_RETRY_INTERVAL)),
    Effect.timeoutFail({
      duration: PROCESS_OPERATION_TIMEOUT,
      onTimeout: () => new AcnProcessIdentityObservationTimedOut({ pid }),
    }),
  )

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

  const probeHealth = (owner: AcnOwnerRecord): Effect.Effect<Option.Option<AcnHealthObservation>> =>
    http.execute(HttpClientRequest.get(`http://127.0.0.1:${owner.port}/health`)).pipe(
      Effect.timeoutOption(HEALTH_TIMEOUT),
      Effect.option,
      Effect.flatMap(Option.match({
        onNone: () => Effect.succeed(Option.none()),
        onSome: Option.match({
          onNone: () => Effect.succeed(Option.none()),
          onSome: (response) => response.json.pipe(
            Effect.flatMap(Schema.decodeUnknown(AcnHealthResponseSchema)),
            Effect.map((health) => Option.some({ status: response.status, health })),
            Effect.catchAll(() => Effect.succeed(Option.none())),
          ),
        }),
      })),
    )

  const observe = Effect.gen(function* () {
    const current = yield* readCurrentOwner
    if (Option.isNone(current)) {
      return new AcnRecordedOwnerAbsent({ expectedOwner: Option.none() })
    }
    const owner = current.value
    const identity = yield* inspectExactProcess(processes, owner.pid)
    if (!Option.contains(identity, owner.processStartIdentity)) {
      const group = yield* processes.observeGroup(groupFrom(owner))
      return group._tag === "ProcessGroupAbsent"
        ? new AcnRecordedOwnerAbsent({ expectedOwner: Option.some(owner) })
        : new AcnRecordedOwnerProcessGroupSurvives({ owner })
    }
    return Option.match(yield* probeHealth(owner), {
      onNone: () => new AcnRecordedOwnerLiveWithoutHealth({ owner }),
      onSome: (health) => new AcnRecordedOwnerLiveWithHealth({ owner, health }),
    })
  })

  const confirmReady: AcnOwnerObserver["confirmReady"] = (owner, observed) => Effect.gen(function* () {
    const { health, status } = observed
    if (status !== 200 || health.state._tag !== "Ready") return Option.none()
    const confirmedOwner = yield* readCurrentOwner
    if (!Option.exists(confirmedOwner, (current) => sameAcnOwner(current, owner))) return Option.none()
    const identity = yield* inspectExactProcess(processes, owner.pid)
    if (!Option.contains(identity, owner.processStartIdentity)) return Option.none()
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
