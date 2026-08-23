import { createHash } from "node:crypto"
import { Context, Deferred, Effect, Fiber, Layer, Option, PubSub, Ref, Stream } from "effect"
import { Key } from "@magnitudedev/effect-query"
import {
  SessionOperationFailed,
  type DisplayViewShape,
  type DisplayViewStateEvent,
  type SessionError,
  type StreamEvent,
} from "@magnitudedev/acn-protocol"
import { AgentRuntime } from "./agent-runtime"
import { formatUnknownCause } from "./session-errors"
import type { RuntimeEntry } from "./session-types"

export interface DisplayViewStreamsApi {
  /**
   * The accepted display view of `sessionId` for `shape`. Subscribing
   * materializes the view (loading the session runtime if needed) and emits a
   * complete `state` event before live `patch` events; reopening rereads a
   * complete snapshot. Holding the stream open is observation and does not
   * retain the runtime.
   */
  readonly stream: (
    sessionId: string,
    shape: DisplayViewShape,
  ) => Stream.Stream<StreamEvent, SessionError>
}

export class DisplayViewStreams extends Context.Tag("DisplayViewStreams")<
  DisplayViewStreams,
  DisplayViewStreamsApi
>() {}

/** The ACN-internal view identity of one shape: equal shapes share one view. */
export const displayViewId = (shape: DisplayViewShape): string =>
  `shape:${createHash("sha256").update(Key.canonical(shape)).digest("hex").slice(0, 16)}`

interface Attachment {
  readonly token: string
  readonly generation: number
  readonly fiber: Fiber.RuntimeFiber<void, unknown>
}

interface RegistrationState {
  readonly attachment: Attachment | null
  readonly subscribers: number
}

interface Registration {
  readonly sessionId: string
  readonly viewId: string
  readonly shape: DisplayViewShape
  readonly events: PubSub.PubSub<StreamEvent>
  readonly state: Ref.Ref<RegistrationState>
  readonly serialize: Effect.Semaphore
}

const operationFailed =
  (sessionId: string, viewId: string, operation: string) =>
  (cause: unknown): SessionError =>
    new SessionOperationFailed({
      operation: `DisplayView.${operation}`,
      reason: `${sessionId}/${viewId}: ${formatUnknownCause(cause)}`,
    })

const toStateEvent = (snapshot: {
  readonly shape: DisplayViewShape
  readonly state: DisplayViewStateEvent["state"]
}): DisplayViewStateEvent => ({
  _tag: "state",
  shape: snapshot.shape,
  state: snapshot.state,
})

export const DisplayViewStreamsLive: Layer.Layer<DisplayViewStreams, never, AgentRuntime> =
  Layer.scoped(
    DisplayViewStreams,
    Effect.gen(function* () {
      const runtime = yield* AgentRuntime
      const layerScope = yield* Effect.scope
      const registrations = yield* Ref.make(new Map<string, Registration>())

      const keyFor = (sessionId: string, viewId: string) => JSON.stringify([sessionId, viewId])

      const makeRegistration = (sessionId: string, viewId: string, shape: DisplayViewShape) =>
        Effect.gen(function* () {
          const events = yield* PubSub.unbounded<StreamEvent>()
          const state = yield* Ref.make<RegistrationState>({ attachment: null, subscribers: 0 })
          const serialize = yield* Effect.makeSemaphore(1)
          return { sessionId, viewId, shape, events, state, serialize } satisfies Registration
        })

      const getOrCreate = (sessionId: string, viewId: string, shape: DisplayViewShape) =>
        Effect.gen(function* () {
          const key = keyFor(sessionId, viewId)
          const current = (yield* Ref.get(registrations)).get(key)
          if (current) return current
          const candidate = yield* makeRegistration(sessionId, viewId, shape)
          return yield* Ref.modify(registrations, (all) => {
            const winner = all.get(key)
            if (winner) return [winner, all] as const
            return [candidate, new Map(all).set(key, candidate)] as const
          })
        })

      const detachUnlocked = (registration: Registration, generation?: number) =>
        Ref.modify(registration.state, (state) => {
          const attachment = state.attachment
          if (
            attachment === null ||
            (generation !== undefined && attachment.generation !== generation)
          ) {
            return [null, state] as const
          }
          return [attachment, { ...state, attachment: null }] as const
        }).pipe(
          Effect.flatMap((attachment) => {
            if (!attachment) return Effect.void
            // Clearing the attachment is the authoritative detach. Forwarding
            // already checks the token before publishing, so termination can
            // happen asynchronously without allowing stale events through.
            // Waiting here would make session retirement depend on every
            // downstream stream finalizer completing promptly.
            return Fiber.interruptFork(attachment.fiber)
          }),
        )

      const detach = (registration: Registration, generation?: number) =>
        registration.serialize.withPermits(1)(detachUnlocked(registration, generation))

      /**
       * Attaches the registration to the loaded runtime generation (or
       * refreshes an existing attachment): sets the agent view's shape, takes
       * a complete snapshot, publishes it, and returns it.
       */
      const attachUnlocked = (
        registration: Registration,
        entry: RuntimeEntry,
        generation: number,
      ): Effect.Effect<DisplayViewStateEvent, SessionError> =>
        Effect.gen(function* () {
          const { sessionId, viewId, shape } = registration
          const current = yield* Ref.get(registration.state)
          yield* entry.session.displayView
            .setShape(viewId, shape)
            .pipe(Effect.mapError(operationFailed(sessionId, viewId, "setShape")))
          const snapshot = yield* entry.session.displayView
            .snapshot(viewId)
            .pipe(Effect.mapError(operationFailed(sessionId, viewId, "snapshot")))
          const event = toStateEvent(snapshot)

          if (current.attachment?.generation === generation) {
            yield* PubSub.publish(registration.events, event)
            return event
          }
          if (current.attachment) {
            yield* Fiber.interrupt(current.attachment.fiber)
          }

          const ready = yield* Deferred.make<void>()
          const token = crypto.randomUUID()
          const display = entry.session.displayView
            .stream(viewId)
            .pipe(
              Stream.map(toStateEvent),
              Stream.mapError(operationFailed(sessionId, viewId, "stream")),
            )
          const queued = entry.session.on.restoreQueuedMessages.pipe(
            Stream.map(
              (value): StreamEvent => ({
                _tag: "restore_queued_messages",
                forkId: value.forkId,
                messages: [...value.messages],
              }),
            ),
          )
          const forward = Stream.merge(display, queued).pipe(
            Stream.runForEach((forwarded) =>
              Deferred.await(ready).pipe(
                Effect.zipRight(
                  registration.serialize.withPermits(1)(
                    Effect.gen(function* () {
                      const observed = (yield* Ref.get(registration.state)).attachment
                      if (observed?.token !== token) return
                      yield* PubSub.publish(registration.events, forwarded)
                    }),
                  ),
                ),
              ),
            ),
            Effect.ensuring(
              Ref.update(registration.state, (state) =>
                state.attachment?.token === token
                  ? { ...state, attachment: null }
                  : state,
              ),
            ),
          )
          const fiber = yield* Effect.forkIn(forward, layerScope)
          yield* Ref.set(registration.state, {
            ...current,
            attachment: { token, generation, fiber },
          })
          yield* Deferred.succeed(ready, undefined)
          yield* PubSub.publish(registration.events, event)
          return event
        })

      const attach = (registration: Registration, entry: RuntimeEntry, generation: number) =>
        registration.serialize.withPermits(1)(attachUnlocked(registration, entry, generation))

      const attachIfBusy = (registration: Registration) =>
        Effect.scoped(Effect.gen(function* () {
          const acquired = yield* runtime.tryAcquireActiveSession(
            registration.sessionId,
            `display-attach:${registration.viewId}`,
          )
          if (Option.isNone(acquired)) return
          yield* attach(registration, acquired.value.entry, acquired.value.generation)
        }))

      const attachBusyRegistrations = Effect.gen(function* () {
        for (const registration of (yield* Ref.get(registrations)).values()) {
          const state = yield* Ref.get(registration.state)
          if (state.attachment === null) yield* attachIfBusy(registration)
        }
      })

      const unregisterRetirementObserver = yield* runtime.registerRetirementObserver({
        retire: ({ sessionId, generation }) =>
          Effect.gen(function* () {
            const all = yield* Ref.get(registrations)
            for (const registration of [...all.values()].filter(
              (registration) => registration.sessionId === sessionId,
            )) {
              const state = yield* Ref.get(registration.state)
              if (state.attachment?.generation !== generation) continue
              yield* detach(registration, generation)
            }
          }),
      })
      yield* Effect.addFinalizer(() => unregisterRetirementObserver)
      yield* runtime.changes.pipe(
        Stream.runForEach(() => attachBusyRegistrations),
        Effect.forkScoped,
      )

      /** Display materialization is session work: it loads the runtime and takes a complete snapshot. */
      const materialize = (registration: Registration): Effect.Effect<void, SessionError> =>
        registration.serialize.withPermits(1)(
          Effect.gen(function* () {
            const key = keyFor(registration.sessionId, registration.viewId)
            if ((yield* Ref.get(registrations)).get(key) !== registration) return
            const acquired = yield* runtime.acquireSession(
              registration.sessionId,
              `display-materialize:${registration.viewId}`,
            )
            yield* attachUnlocked(registration, acquired.entry, acquired.generation)
          }),
        ).pipe(Effect.scoped)

      const stream = (
        sessionId: string,
        shape: DisplayViewShape,
      ): Stream.Stream<StreamEvent, SessionError> =>
        Stream.unwrapScoped(
          Effect.gen(function* () {
            const viewId = displayViewId(shape)
            const registration = yield* getOrCreate(sessionId, viewId, shape)
            const queue = yield* PubSub.subscribe(registration.events)
            const admitted = yield* registration.serialize.withPermits(1)(
              Effect.gen(function* () {
                const exact = (yield* Ref.get(registrations)).get(keyFor(sessionId, viewId))
                if (exact !== registration) return false
                yield* Ref.update(registration.state, (state) => ({
                  ...state,
                  subscribers: state.subscribers + 1,
                }))
                return true
              }),
            )
            if (!admitted) return stream(sessionId, shape)
            yield* Effect.addFinalizer(() =>
              registration.serialize.withPermits(1)(
                Effect.gen(function* () {
                  const state = yield* Ref.get(registration.state)
                  const subscribers = Math.max(0, state.subscribers - 1)
                  yield* Ref.set(registration.state, { ...state, subscribers })
                  if (subscribers !== 0) return
                  if ((yield* Ref.get(registrations)).get(keyFor(sessionId, viewId)) !== registration) {
                    return
                  }
                  yield* Effect.scoped(Effect.gen(function* () {
                    const acquired = yield* runtime.tryAcquireActiveSession(
                      sessionId,
                      `display-close:${viewId}`,
                    )
                    if (Option.isSome(acquired)) {
                      yield* acquired.value.entry.session.displayView.close(viewId)
                    }
                  }))
                  yield* detachUnlocked(registration)
                  yield* Ref.update(registrations, (all) => {
                    if (all.get(keyFor(sessionId, viewId)) !== registration) return all
                    const next = new Map(all)
                    next.delete(keyFor(sessionId, viewId))
                    return next
                  })
                }),
              ).pipe(Effect.catchAll(() => Effect.void)),
            )
            yield* materialize(registration)
            // The materialized snapshot is the first event this observer needs;
            // anything published to the shared view before it belongs to an
            // earlier snapshot.
            return Stream.fromQueue(queue).pipe(
              Stream.dropWhile((event) => event._tag !== "state"),
            )
          }),
        )

      yield* Effect.addFinalizer(() =>
        Effect.gen(function* () {
          const all = yield* Ref.get(registrations)
          yield* Effect.forEach(all.values(), (registration) => detach(registration), {
            discard: true,
          })
        }),
      )

      return { stream }
    }),
  )
