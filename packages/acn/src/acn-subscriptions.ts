import {
  AcnSubscriptionKeepalive,
  AcnSubscriptionSuspended,
  AcnSubscriptionTerminated,
  ACN_SUBSCRIPTION_KEEPALIVE_INTERVAL_MS,
  type AcnSubscriptionControl,
} from "@magnitudedev/protocol"
import { Context, Effect, Fiber, Layer, Ref, Scope } from "effect"

export interface AcnSubscriptionRegistration {
  readonly clientId: number
  readonly requestId: string
  readonly sessionId?: string
  readonly emit: (control: AcnSubscriptionControl) => Effect.Effect<void>
}

export interface AcnSubscriptionHandle {
  readonly unregister: Effect.Effect<void>
}

export interface AcnSubscriptionsApi {
  /** Registers one encoded subscription transport and returns its exact ownership handle. */
  readonly register: (
    registration: AcnSubscriptionRegistration,
  ) => Effect.Effect<AcnSubscriptionHandle>
  /** Notifies only display subscriptions attached to the retired session. */
  readonly suspendSession: (sessionId: string) => Effect.Effect<void>
  /** Stops admission and emits the authoritative terminal control to every subscription. */
  readonly terminate: Effect.Effect<void>
}

export class AcnSubscriptions extends Context.Tag("AcnSubscriptions")<
  AcnSubscriptions,
  AcnSubscriptionsApi
>() {}

interface ActiveSubscription extends AcnSubscriptionRegistration {
  readonly keepalive: Fiber.RuntimeFiber<void>
}

interface SubscriptionState {
  readonly terminated: boolean
  readonly active: ReadonlyMap<number, ReadonlyMap<string, ActiveSubscription>>
}

const emptyState: SubscriptionState = {
  terminated: false,
  active: new Map(),
}

const values = (state: SubscriptionState): ReadonlyArray<ActiveSubscription> =>
  Array.from(state.active.values()).flatMap((requests) => Array.from(requests.values()))

export const AcnSubscriptionsLive: Layer.Layer<AcnSubscriptions> = Layer.scoped(
  AcnSubscriptions,
  Effect.gen(function* () {
    const scope = yield* Scope.Scope
    const state = yield* Ref.make(emptyState)
    const semaphore = yield* Effect.makeSemaphore(1)

    const unregister = (clientId: number, requestId: string) =>
      semaphore.withPermits(1)(
        Effect.gen(function* () {
          const current = yield* Ref.get(state)
          const client = current.active.get(clientId)
          const subscription = client?.get(requestId)
          if (!subscription) return

          const nextClient = new Map(client)
          nextClient.delete(requestId)
          const active = new Map(current.active)
          if (nextClient.size === 0) active.delete(clientId)
          else active.set(clientId, nextClient)
          yield* Ref.set(state, { ...current, active })
          yield* Fiber.interrupt(subscription.keepalive)
        }),
      )

    const register = (registration: AcnSubscriptionRegistration) =>
      semaphore.withPermits(1)(
        Effect.gen(function* () {
          const current = yield* Ref.get(state)
          if (current.terminated) {
            yield* registration.emit(
              AcnSubscriptionTerminated.make({ reason: "acn-shutdown" }),
            )
            return { unregister: Effect.void }
          }

          const keepalive = yield* Effect.sleep(
            `${ACN_SUBSCRIPTION_KEEPALIVE_INTERVAL_MS} millis`,
          ).pipe(
            Effect.zipRight(
              registration.emit(AcnSubscriptionKeepalive.make({})),
            ),
            Effect.forever,
            Effect.forkIn(scope),
          )
          const client = new Map(current.active.get(registration.clientId) ?? [])
          client.set(registration.requestId, { ...registration, keepalive })
          const active = new Map(current.active)
          active.set(registration.clientId, client)
          yield* Ref.set(state, { ...current, active })
          return {
            unregister: unregister(registration.clientId, registration.requestId),
          }
        }),
      )

    const suspendSession = (sessionId: string) =>
      Ref.get(state).pipe(
        Effect.flatMap((current) =>
          Effect.forEach(
            values(current).filter(
              (subscription) => subscription.sessionId === sessionId,
            ),
            (subscription) =>
              subscription.emit(
                AcnSubscriptionSuspended.make({ reason: "session-offloaded" }),
              ),
            { discard: true },
          ),
        ),
      )

    const terminate = semaphore.withPermits(1)(
      Effect.gen(function* () {
        const current = yield* Ref.get(state)
        if (current.terminated) return
        const active = values(current)
        yield* Ref.set(state, { ...current, terminated: true })
        yield* Effect.forEach(
          active,
          (subscription) => Fiber.interrupt(subscription.keepalive),
          { discard: true },
        )
        yield* Effect.forEach(
          active,
          (subscription) =>
            subscription.emit(
              AcnSubscriptionTerminated.make({ reason: "acn-shutdown" }),
          ),
          { discard: true },
        )
      }),
    )

    return { register, suspendSession, terminate }
  }),
)
