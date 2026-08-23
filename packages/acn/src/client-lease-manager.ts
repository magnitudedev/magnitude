import { type ClientId, type ClientLeaseMutationResult } from "@magnitudedev/acn-protocol"
import { Clock, Context, Deferred, Duration, Effect, Fiber, Layer, Option, Ref, Scope } from "effect"
import {
  dueClientLeases,
  emptyClientLeaseSet,
  nextClientLeaseDeadline,
  removeClientLease,
  renewClientLease,
  type ClientLeaseSet,
} from "./client-lease-state"
import { ModelResidencyPolicy } from "./model-residency-policy"
import { AcnServiceLifecycle } from "./service-lifecycle"

interface ClientLeaseManagerState {
  readonly leaseSet: ClientLeaseSet
  readonly changed: Deferred.Deferred<void>
}

export interface ClientLeaseManager {
  readonly renew: (clientId: ClientId) => Effect.Effect<ClientLeaseMutationResult>
  readonly release: (clientId: ClientId) => Effect.Effect<ClientLeaseMutationResult>
}

export const ClientLeaseManager = Context.GenericTag<ClientLeaseManager>("ClientLeaseManager")

const DEFAULT_CLIENT_LEASE_TIMEOUT = Duration.seconds(35)

export const makeClientLeaseManager = (
  leaseTimeout: Duration.DurationInput = DEFAULT_CLIENT_LEASE_TIMEOUT
): Effect.Effect<ClientLeaseManager, never, AcnServiceLifecycle | ModelResidencyPolicy | Scope.Scope> =>
  Effect.gen(function* () {
    const lifecycle = yield* AcnServiceLifecycle
    const residencyPolicy = yield* ModelResidencyPolicy
    const scope = yield* Scope.Scope
    const timeout = Duration.decode(leaseTimeout)
    const timeoutNanos = yield* Option.match(Duration.toNanos(timeout), {
      onNone: () => Effect.dieMessage("Client lease timeout must be finite"),
      onSome: Effect.succeed,
    })
    if (timeoutNanos <= 0n) {
      return yield* Effect.dieMessage("Client lease timeout must be positive")
    }

    const initialChanged = yield* Deferred.make<void>()
    const state = yield* Ref.make<ClientLeaseManagerState>({
      leaseSet: emptyClientLeaseSet(),
      changed: initialChanged,
    })
    const mutationLock = yield* Effect.makeSemaphore(1)

    const asResult = (leaseSet: ClientLeaseSet): ClientLeaseMutationResult => ({
      connectedClientCount: leaseSet.leases.size,
    })

    const failClosedPolicyUpdate = (connected: boolean) =>
      residencyPolicy.setConnected(connected).pipe(
        Effect.catchTag("ModelResidencyPolicyUnavailable", (error) =>
          Effect.logError("Failed to establish model residency policy").pipe(
            Effect.annotateLogs({
              connected,
              operation: error.operation,
              message: error.message,
            }),
            Effect.zipRight(
              lifecycle.beginStopping({
                reason: "fatal",
                detail: "Magnitude could not establish the local-model residency policy",
              })
            ),
            Effect.zipRight(Effect.interrupt)
          )
        )
      )

    // Caller interruption is deferred through the matching ACN commit. Run
    // the bounded policy operation in an interruptible child so its own
    // timeout remains effective, then join it from the atomic mutation.
    const completePolicyUpdate = (connected: boolean) =>
      Effect.gen(function* () {
        const update = yield* failClosedPolicyUpdate(connected).pipe(
          Effect.interruptible,
          Effect.fork
        )
        yield* Fiber.join(update)
      })

    const commit = (
      previous: ClientLeaseManagerState,
      leaseSet: ClientLeaseSet,
    ) =>
      Effect.uninterruptible(
        Effect.gen(function* () {
          const changed = yield* Deferred.make<void>()
          yield* Ref.set(state, { leaseSet, changed })
          yield* Deferred.succeed(previous.changed, undefined)
        })
      )

    const renew: ClientLeaseManager["renew"] = (clientId) =>
      mutationLock.withPermits(1)(
        Effect.uninterruptible(
          Effect.gen(function* () {
            const previous = yield* Ref.get(state)
            const nowNanos = yield* Clock.currentTimeNanos
            const transition = renewClientLease(previous.leaseSet, clientId, nowNanos, timeoutNanos)

            if (!transition.connectionChanged) {
              yield* commit(previous, transition.state)
              return asResult(transition.state)
            }

            if (!(yield* lifecycle.setClientPresence(true))) return yield* Effect.interrupt
            yield* completePolicyUpdate(true)
            yield* commit(previous, transition.state)
            return asResult(transition.state)
          })
        )
      )

    const remove = (clientId: ClientId, expectedRenewalGeneration?: number): Effect.Effect<ClientLeaseMutationResult> =>
      mutationLock.withPermits(1)(
        Effect.uninterruptible(
          Effect.gen(function* () {
            const previous = yield* Ref.get(state)
            const transition = removeClientLease(previous.leaseSet, clientId, expectedRenewalGeneration)
            if (!transition.removed) return asResult(previous.leaseSet)

            if (!transition.connectionChanged) {
              yield* commit(previous, transition.state)
              return asResult(transition.state)
            }

            yield* completePolicyUpdate(false)
            yield* commit(previous, transition.state)
            yield* lifecycle.setClientPresence(false)
            return asResult(transition.state)
          })
        )
      )

    const supervise = Effect.forever(
      Effect.gen(function* () {
        const observed = yield* Ref.get(state)
        const deadline = nextClientLeaseDeadline(observed.leaseSet)
        if (deadline === undefined) {
          yield* Deferred.await(observed.changed)
          return
        }

        const nowNanos = yield* Clock.currentTimeNanos
        const remaining = deadline > nowNanos ? Duration.nanos(deadline - nowNanos) : Duration.zero
        const wake = yield* Effect.race(
          Deferred.await(observed.changed).pipe(Effect.as("changed" as const)),
          Effect.sleep(remaining).pipe(Effect.as("deadline" as const))
        )
        if (wake === "changed") return

        const dueAt = yield* Clock.currentTimeNanos
        const current = yield* Ref.get(state)
        for (const [clientId, lease] of dueClientLeases(current.leaseSet, dueAt)) {
          yield* remove(clientId, lease.renewalGeneration)
        }
      })
    )
    yield* Effect.forkIn(supervise, scope)

    return {
      renew,
      release: (clientId) => remove(clientId),
    }
  })

export const ClientLeaseManagerLive: Layer.Layer<
  ClientLeaseManager,
  never,
  AcnServiceLifecycle | ModelResidencyPolicy
> = Layer.scoped(ClientLeaseManager, makeClientLeaseManager())
