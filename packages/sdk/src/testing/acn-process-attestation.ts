import { BunContext } from "@effect/platform-bun"
import {
  ExactProcessControllerLive,
  makeAcnOwnerStore,
  waitForTreeAbsence,
  type AcnOwnerRecord,
  type AcnOwnerStore,
  type ExactProcess,
  type ExactProcessController,
} from "@magnitudedev/acn-protocol/coordination"
import { BunSqliteDriverLayer } from "@magnitudedev/acn-protocol/coordination/bun"
import { Data, Duration, Effect, Layer, Option } from "effect"

export interface AcnProcessAttestation {
  readonly owner: AcnOwnerRecord
}

export interface AcnProcessAttestationServices {
  readonly owners: AcnOwnerStore
  readonly processes: ExactProcessController
}

export class AcnProcessAttestationFailed extends Data.TaggedError(
  "AcnProcessAttestationFailed",
)<{
  readonly operation: "capture" | "verify-exit"
  readonly message: string
}> {}

const failureMessage = (cause: unknown): string =>
  cause instanceof Error ? cause.message : String(cause)

const failed = (
  operation: AcnProcessAttestationFailed["operation"],
  message: string,
): AcnProcessAttestationFailed => new AcnProcessAttestationFailed({ operation, message })

const exactFrom = (owner: AcnOwnerRecord): ExactProcess => ({
  pid: owner.pid,
  processStartIdentity: owner.processStartIdentity,
})

const readOwner = (
  services: AcnProcessAttestationServices,
  operation: AcnProcessAttestationFailed["operation"],
) => services.owners.current.pipe(
  Effect.mapError((error) => failed(operation, failureMessage(error))),
)

export const captureAcnProcessAttestationWith = (
  services: AcnProcessAttestationServices,
): Effect.Effect<AcnProcessAttestation, AcnProcessAttestationFailed> =>
  Effect.gen(function* () {
    const owner = yield* readOwner(services, "capture")
    if (Option.isNone(owner)) {
      return yield* new AcnProcessAttestationFailed({
        operation: "capture",
        message: "ACN coordination has no current owner",
      })
    }
    const identity = yield* services.processes.inspect(owner.value.pid).pipe(
      Effect.mapError((error) => failed("capture", failureMessage(error))),
    )
    if (!Option.contains(identity, owner.value.processStartIdentity)) {
      return yield* new AcnProcessAttestationFailed({
        operation: "capture",
        message: `ACN owner ${owner.value.pid} does not identify its recorded process occurrence`,
      })
    }
    return { owner: owner.value }
  })

export const attestAcnProcessTreeExitWith = (
  services: AcnProcessAttestationServices,
  attestation: AcnProcessAttestation,
  timeoutMs: number,
): Effect.Effect<void, AcnProcessAttestationFailed> => Effect.gen(function* () {
  const original = exactFrom(attestation.owner)
  const originalAbsent = yield* waitForTreeAbsence(
    services.processes,
    original,
    Duration.millis(Math.max(0, timeoutMs)),
  ).pipe(Effect.mapError((error) => failed("verify-exit", failureMessage(error))))
  if (!originalAbsent) {
    return yield* new AcnProcessAttestationFailed({
      operation: "verify-exit",
      message: `ACN process tree ${attestation.owner.pid} remained present`,
    })
  }

  const current = yield* readOwner(services, "verify-exit")
  if (Option.isNone(current)) return
  const currentExact = exactFrom(current.value)
  const identity = yield* services.processes.inspect(current.value.pid).pipe(
    Effect.mapError((error) => failed("verify-exit", failureMessage(error))),
  )
  if (Option.contains(identity, current.value.processStartIdentity)) {
    return yield* new AcnProcessAttestationFailed({
      operation: "verify-exit",
      message: `ACN coordination identifies live owner ${current.value.pid}`,
    })
  }
  const successorTreeAbsent = yield* services.processes.treeAbsent(currentExact).pipe(
    Effect.mapError((error) => failed("verify-exit", failureMessage(error))),
  )
  if (!successorTreeAbsent) {
    return yield* new AcnProcessAttestationFailed({
      operation: "verify-exit",
      message: `ACN coordination owner ${current.value.pid} has surviving process-group members`,
    })
  }
})

const liveLayer = Layer.merge(BunContext.layer, BunSqliteDriverLayer)

const withLiveServices = <A>(
  dataDirectory: string,
  use: (services: AcnProcessAttestationServices) => Effect.Effect<A, AcnProcessAttestationFailed>,
): Effect.Effect<A, AcnProcessAttestationFailed> => Effect.gen(function* () {
  const owners = yield* makeAcnOwnerStore(dataDirectory)
  return yield* use({ owners, processes: ExactProcessControllerLive })
}).pipe(Effect.provide(liveLayer))

export const waitForAcnProcessAttestation = (
  dataDirectory: string,
  timeoutMs: number,
): Effect.Effect<AcnProcessAttestation, AcnProcessAttestationFailed> =>
  withLiveServices(dataDirectory, (services) => Effect.gen(function* () {
    const startedAt = yield* Effect.clockWith((clock) => clock.currentTimeMillis)
    const deadline = startedAt + Math.max(0, timeoutMs)
    while (true) {
      const observed = yield* captureAcnProcessAttestationWith(services).pipe(Effect.either)
      if (observed._tag === "Right") return observed.right
      const now = yield* Effect.clockWith((clock) => clock.currentTimeMillis)
      if (now >= deadline) return yield* observed.left
      yield* Effect.sleep(25)
    }
  }))

export const attestAcnProcessTreeExit = (
  dataDirectory: string,
  attestation: AcnProcessAttestation,
  timeoutMs: number,
): Effect.Effect<void, AcnProcessAttestationFailed> =>
  withLiveServices(dataDirectory, (services) =>
    attestAcnProcessTreeExitWith(services, attestation, timeoutMs))
