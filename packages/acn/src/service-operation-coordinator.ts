import {
  Deferred,
  Effect,
  Exit,
  Option,
  Ref,
  Scope,
  type Fiber,
} from "effect"
import {
  AcnActivityTracker,
} from "./activity-tracker"
import type { ResourceRetired } from "./resource-use-gate"

export type ServiceOperationAdmission<E> =
  | { readonly _tag: "Satisfied" }
  | { readonly _tag: "Current"; readonly outcome: Effect.Effect<Exit.Exit<void, E>> }
  | { readonly _tag: "Conflicting"; readonly outcome: Effect.Effect<Exit.Exit<void, E>> }

export interface ServiceOperationDefinition<E> {
  readonly activityLabel: string
  /** Infallible, bounded publication of the domain's nonterminal state. */
  readonly commit: Effect.Effect<void>
  readonly operation: Effect.Effect<void, E>
  readonly terminalize: (exit: Exit.Exit<void, E>) => Effect.Effect<void>
}

export interface ServiceOperationRequest<K, E, AdmissionError> {
  readonly key: K
  readonly whenIdle: Effect.Effect<
    Option.Option<ServiceOperationDefinition<E>>,
    AdmissionError
  >
}

export interface ServiceOperationSupersession<K, E, AdmissionError>
  extends ServiceOperationRequest<K, E, AdmissionError> {
  readonly whenConflicting: (
    currentKey: K,
    currentOutcome: Effect.Effect<Exit.Exit<void, E>>,
  ) => Effect.Effect<Option.Option<ServiceOperationDefinition<E>>, AdmissionError>
}

export interface ServiceOperationCoordinator<K, E> {
  readonly admit: <AdmissionError>(
    request: Effect.Effect<ServiceOperationRequest<K, E, AdmissionError>, AdmissionError>,
  ) => Effect.Effect<ServiceOperationAdmission<E>, AdmissionError | ResourceRetired>
  readonly supersede: <AdmissionError>(
    request: Effect.Effect<
      ServiceOperationSupersession<K, E, AdmissionError>,
      AdmissionError
    >,
  ) => Effect.Effect<ServiceOperationAdmission<E>, AdmissionError | ResourceRetired>
}

interface ActiveOperation<K, E> {
  readonly key: K
  readonly outcome: Deferred.Deferred<Exit.Exit<void, E>>
}

export const makeServiceOperationCoordinator = <K, E>(
  equivalent: (left: K, right: K) => boolean,
): Effect.Effect<
  ServiceOperationCoordinator<K, E>,
  never,
  AcnActivityTracker | Scope.Scope
> => Effect.gen(function* () {
  const activity = yield* AcnActivityTracker
  const scope = yield* Scope.Scope
  const admissionLock = yield* Effect.makeSemaphore(1)
  const active = yield* Ref.make<Option.Option<ActiveOperation<K, E>>>(Option.none())

  const completeUnlocked = (
    operation: ActiveOperation<K, E>,
    exit: Exit.Exit<void, E>,
  ) => Effect.gen(function* () {
    const current = yield* Ref.get(active)
    if (Option.isSome(current) && current.value.outcome === operation.outcome) {
      yield* Ref.set(active, Option.none())
    }
    yield* Deferred.succeed(operation.outcome, exit)
  })

  const complete = (
    operation: ActiveOperation<K, E>,
    exit: Exit.Exit<void, E>,
  ) => Effect.uninterruptible(
    admissionLock.withPermits(1)(completeUnlocked(operation, exit)),
  )

  const launch = (
    current: ActiveOperation<K, E>,
    definition: ServiceOperationDefinition<E>,
    releaseActivity: Effect.Effect<void>,
  ): Effect.Effect<Fiber.RuntimeFiber<void, never>> =>
    Effect.forkIn(
      Effect.exit(definition.operation).pipe(
        Effect.flatMap((exit) =>
          Effect.uninterruptible(
            Effect.exit(definition.terminalize(exit)).pipe(
              Effect.flatMap((terminalization) =>
                complete(current, exit).pipe(
                  Effect.zipRight(Exit.isFailure(terminalization)
                    ? Effect.failCause(terminalization.cause)
                    : Effect.void),
                )),
            ),
          )),
        Effect.ensuring(releaseActivity),
      ),
      scope,
    )

  const start = (
    key: K,
    definition: ServiceOperationDefinition<E>,
  ): Effect.Effect<ServiceOperationAdmission<E>, ResourceRetired> =>
    Effect.uninterruptible(Effect.gen(function* () {
      const outcome = yield* Deferred.make<Exit.Exit<void, E>>()
      const releaseActivity = yield* activity.acquire(definition.activityLabel)
      const current = { key, outcome } satisfies ActiveOperation<K, E>
      yield* Ref.set(active, Option.some(current))
      const committed = yield* Effect.exit(definition.commit)
      if (Exit.isFailure(committed)) {
        const operationExit: Exit.Exit<void, E> = Exit.failCause(committed.cause)
        const terminalized = yield* Effect.exit(definition.terminalize(operationExit))
        yield* completeUnlocked(current, operationExit)
        yield* releaseActivity
        if (Exit.isFailure(terminalized)) return yield* Effect.failCause(terminalized.cause)
        return yield* Effect.failCause(committed.cause)
      }
      yield* launch(current, definition, releaseActivity)
      return {
        _tag: "Current" as const,
        outcome: Deferred.await(outcome),
      }
    }))

  const admit: ServiceOperationCoordinator<K, E>["admit"] = (request) =>
    admissionLock.withPermits(1)(Effect.gen(function* () {
      const prepared = yield* request
      const current = yield* Ref.get(active)
      if (Option.isSome(current)) {
        return {
          _tag: equivalent(current.value.key, prepared.key)
            ? "Current" as const
            : "Conflicting" as const,
          outcome: Deferred.await(current.value.outcome),
        }
      }
      const definition = yield* prepared.whenIdle
      if (Option.isNone(definition)) return { _tag: "Satisfied" as const }
      return yield* start(prepared.key, definition.value)
    }))

  const supersede: ServiceOperationCoordinator<K, E>["supersede"] = (request) =>
    admissionLock.withPermits(1)(Effect.gen(function* () {
      const prepared = yield* request
      const current = yield* Ref.get(active)
      if (Option.isNone(current)) {
        const definition = yield* prepared.whenIdle
        if (Option.isNone(definition)) return { _tag: "Satisfied" as const }
        return yield* start(prepared.key, definition.value)
      }
      if (equivalent(current.value.key, prepared.key)) {
        return {
          _tag: "Current" as const,
          outcome: Deferred.await(current.value.outcome),
        }
      }
      const definition = yield* prepared.whenConflicting(
        current.value.key,
        Deferred.await(current.value.outcome),
      )
      if (Option.isNone(definition)) {
        return {
          _tag: "Conflicting" as const,
          outcome: Deferred.await(current.value.outcome),
        }
      }
      return yield* start(prepared.key, definition.value)
    }))

  return { admit, supersede }
})
