import {
  AcnSubscriptionKeepalive,
  AcnSubscriptionTerminated,
  ACN_SUBSCRIPTION_KEEPALIVE_INTERVAL_MS,
  type AcnSubscriptionControl,
} from "@magnitudedev/acn-protocol"
import { Context, Effect, Fiber, Layer, Option, Ref, Scope } from "effect"

export interface AcnSubscriptionRegistration {
  readonly clientId: number
  readonly requestId: string
  readonly emit: (control: AcnSubscriptionControl) => Effect.Effect<void>
  readonly close: Effect.Effect<void>
}

export interface AcnSubscriptionHandle {
  readonly unregister: Effect.Effect<void>
}

export interface AcnSubscriptionsApi {
  /** Registers one encoded subscription transport and returns its exact ownership handle. */
  readonly register: (
    registration: AcnSubscriptionRegistration,
  ) => Effect.Effect<AcnSubscriptionHandle>
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

/** Observe an operation for a bounded interval without waiting for its interruption. */
const runBounded = (effect: Effect.Effect<unknown>): Effect.Effect<void> =>
  Effect.gen(function* () {
    const fiber = yield* Effect.forkDaemon(effect)
    yield* Effect.raceFirst(
      Fiber.await(fiber),
      Effect.sleep("250 millis"),
    ).pipe(Effect.asVoid)
    yield* Fiber.interruptFork(fiber)
  })

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
          yield* Fiber.interruptFork(subscription.keepalive)
        }),
      )

    const register = (registration: AcnSubscriptionRegistration) =>
      semaphore.withPermits(1)(
        Effect.gen(function* () {
          const current = yield* Ref.get(state)
          if (current.terminated) {
            return Option.none<AcnSubscriptionHandle>()
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
          return Option.some({
            unregister: unregister(registration.clientId, registration.requestId),
          })
        }),
      ).pipe(
        Effect.flatMap(
          Option.match({
            onSome: Effect.succeed,
            onNone: () => runBounded(registration.close.pipe(
              Effect.ignore,
            )).pipe(
              Effect.as({ unregister: Effect.void }),
            ),
          }),
        ),
      )

    const detach = semaphore.withPermits(1)(
      Effect.gen(function* () {
        const current = yield* Ref.get(state)
        if (current.terminated) return []
        const active = values(current)
        yield* Ref.set(state, { terminated: true, active: new Map() })
        return active
      }),
    )

    const terminate = Effect.gen(function* () {
      const active = yield* detach
      if (active.length === 0) return
      yield* Effect.forEach(
        active,
        (subscription) => Fiber.interruptFork(subscription.keepalive),
        { discard: true, concurrency: "unbounded" },
      )
      yield* Effect.forEach(
        active,
        (subscription) =>
          subscription.emit(
            AcnSubscriptionTerminated.make({ reason: "acn-shutdown" }),
          ),
        { discard: true, concurrency: "unbounded" },
      ).pipe(Effect.ignore, runBounded)
      const clients = new Map(active.map((subscription) => [
        subscription.clientId,
        subscription.close,
      ]))
      yield* Effect.forEach(
        clients.values(),
        (close) => close,
        { discard: true, concurrency: "unbounded" },
      ).pipe(Effect.ignore, runBounded)
    })

    return { register, terminate }
  }),
)
