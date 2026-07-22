import { Context, Deferred, Effect, Fiber, Layer, Option, PubSub, Ref, Scope, Stream } from "effect"
import {
  DisplayViewNotOpen,
  SessionOperationFailed,
  type DisplayViewShape,
  type DisplayViewStateEvent,
  type SessionError,
  type StreamEvent,
} from "@magnitudedev/protocol"
import { AcnSubscriptions } from "./acn-subscriptions"
import { AgentRuntime } from "./agent-runtime"
import { formatUnknownCause } from "./session-errors"
import type { RuntimeEntry } from "./session-types"

export interface DisplayViewStreamsApi {
  readonly getDisplayViewStream: (
    sessionId: string,
    viewId: string,
    shape: DisplayViewShape,
  ) => Stream.Stream<StreamEvent, SessionError>
  readonly requestDisplayViewSnapshot: (
    sessionId: string,
    viewId: string,
  ) => Effect.Effect<DisplayViewStateEvent, SessionError>
  readonly setDisplayViewShape: (
    sessionId: string,
    viewId: string,
    shape: DisplayViewShape,
  ) => Effect.Effect<DisplayViewStateEvent, SessionError>
}

export class DisplayViewStreams extends Context.Tag("DisplayViewStreams")<
  DisplayViewStreams,
  DisplayViewStreamsApi
>() {}

interface Attachment {
  readonly token: string
  readonly generation: number
  readonly fiber: Fiber.RuntimeFiber<void, unknown>
  readonly latest: Ref.Ref<DisplayViewStateEvent>
}

interface RegistrationState {
  readonly shape: DisplayViewShape
  readonly attachment: Attachment | null
  readonly subscribers: number
}

interface Registration {
  readonly sessionId: string
  readonly viewId: string
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

export const DisplayViewStreamsLive: Layer.Layer<
  DisplayViewStreams,
  never,
  AgentRuntime | AcnSubscriptions
> =
  Layer.scoped(
    DisplayViewStreams,
    Effect.gen(function* () {
      const runtime = yield* AgentRuntime
      const subscriptions = yield* AcnSubscriptions
      const layerScope = yield* Effect.scope
      const registrations = yield* Ref.make(new Map<string, Registration>())

      const keyFor = (sessionId: string, viewId: string) => JSON.stringify([sessionId, viewId])

      const makeRegistration = (sessionId: string, viewId: string, shape: DisplayViewShape) =>
        Effect.gen(function* () {
          const events = yield* PubSub.unbounded<StreamEvent>()
          const state = yield* Ref.make<RegistrationState>({
            shape,
            attachment: null,
            subscribers: 0,
          })
          const serialize = yield* Effect.makeSemaphore(1)
          return { sessionId, viewId, events, state, serialize } satisfies Registration
        })

      const getOrCreate = (sessionId: string, viewId: string, shape: DisplayViewShape) =>
        Effect.gen(function* () {
          const key = keyFor(sessionId, viewId)
          const current = (yield* Ref.get(registrations)).get(key)
          // A stream subscription is observation. In particular, reconnecting
          // an old stream must not mutate the materialized shape. Shape changes
          // are admitted only by setDisplayViewShape below.
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
          Effect.flatMap((attachment) =>
            attachment ? Fiber.interrupt(attachment.fiber).pipe(Effect.asVoid) : Effect.void,
          ),
        )

      const detach = (registration: Registration, generation?: number) =>
        registration.serialize.withPermits(1)(detachUnlocked(registration, generation))

      const attachUnlocked = (
        registration: Registration,
        entry: RuntimeEntry,
        generation: number,
        refresh = false,
      ): Effect.Effect<DisplayViewStateEvent, SessionError> =>
        Effect.gen(function* () {
            const current = yield* Ref.get(registration.state)
            if (current.attachment?.generation === generation) {
              if (!refresh) return yield* Ref.get(current.attachment.latest)
              yield* entry.session.displayView
                .setShape(registration.viewId, current.shape)
                .pipe(
                  Effect.mapError(
                    operationFailed(registration.sessionId, registration.viewId, "setShape"),
                  ),
                )
              const snapshot = yield* entry.session.displayView
                .snapshot(registration.viewId)
                .pipe(
                  Effect.mapError(
                    operationFailed(registration.sessionId, registration.viewId, "snapshot"),
                  ),
                )
              const event = toStateEvent(snapshot)
              yield* Ref.set(current.attachment.latest, event)
              yield* PubSub.publish(registration.events, event)
              return event
            }

            if (current.attachment) {
              yield* Fiber.interrupt(current.attachment.fiber)
            }
            yield* entry.session.displayView
              .setShape(registration.viewId, current.shape)
              .pipe(
                Effect.mapError(
                  operationFailed(registration.sessionId, registration.viewId, "setShape"),
                ),
              )
            const snapshot = yield* entry.session.displayView
              .snapshot(registration.viewId)
              .pipe(
                Effect.mapError(
                  operationFailed(registration.sessionId, registration.viewId, "snapshot"),
                ),
              )
            const initial = toStateEvent(snapshot)
            const latest = yield* Ref.make(initial)
            const ready = yield* Deferred.make<void>()
            const token = crypto.randomUUID()

            const display = entry.session.displayView
              .stream(registration.viewId)
              .pipe(
                Stream.map(toStateEvent),
                Stream.mapError(
                  operationFailed(registration.sessionId, registration.viewId, "stream"),
                ),
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
              Stream.runForEach((event) =>
                Deferred.await(ready).pipe(
                  Effect.zipRight(
                    registration.serialize.withPermits(1)(
                      Effect.gen(function* () {
                        const observed = (yield* Ref.get(registration.state)).attachment
                        if (observed?.token !== token) return
                        if (event._tag === "state") yield* Ref.set(latest, event)
                        yield* PubSub.publish(registration.events, event)
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
              attachment: { token, generation, fiber, latest },
            })
            yield* Deferred.succeed(ready, undefined)
            yield* PubSub.publish(registration.events, initial)
            return initial
        })

      const attach = (
        registration: Registration,
        entry: RuntimeEntry,
        generation: number,
        refresh = false,
      ) =>
        registration.serialize.withPermits(1)(
          attachUnlocked(registration, entry, generation, refresh),
        )

      const attachIfBusy = (registration: Registration) =>
        runtime
          .tryWithBusyResident(
            registration.sessionId,
            `display-attach:${registration.viewId}`,
            (entry, generation) => attach(registration, entry, generation),
          )
          .pipe(Effect.asVoid)

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
            let suspended = false
            for (const registration of [...all.values()].filter(
              (registration) => registration.sessionId === sessionId,
            )) {
              const state = yield* Ref.get(registration.state)
              if (state.attachment?.generation !== generation) continue
              yield* detach(registration, generation)
              suspended = true
            }
            if (suspended) yield* subscriptions.suspendSession(sessionId)
          }),
      })
      yield* Effect.addFinalizer(() => unregisterRetirementObserver)
      yield* runtime.changes.pipe(
        Stream.runForEach(() => attachBusyRegistrations),
        Effect.forkScoped,
      )

      const getDisplayViewStream = (
        sessionId: string,
        viewId: string,
        shape: DisplayViewShape,
      ): Stream.Stream<StreamEvent, SessionError> =>
        Stream.unwrapScoped(
          Effect.gen(function* () {
            const registration = yield* getOrCreate(sessionId, viewId, shape)
            const queue = yield* PubSub.subscribe(registration.events)
            const admission = yield* registration.serialize.withPermits(1)(
              Effect.gen(function* () {
                const exact = (yield* Ref.get(registrations)).get(keyFor(sessionId, viewId))
                if (exact !== registration) return { admitted: false as const }
                const state = yield* Ref.get(registration.state)
                yield* Ref.update(registration.state, (state) => ({
                  ...state,
                  subscribers: state.subscribers + 1,
                }))
                return {
                  admitted: true as const,
                  initial: state.attachment
                    ? Option.some(yield* Ref.get(state.attachment.latest))
                    : Option.none<DisplayViewStateEvent>(),
                }
              }),
            )
            if (!admission.admitted) return getDisplayViewStream(sessionId, viewId, shape)
            yield* attachIfBusy(registration)
            yield* Effect.addFinalizer(() =>
              Effect.gen(function* () {
                const shouldRemove = yield* registration.serialize.withPermits(1)(
                  Effect.gen(function* () {
                    const state = yield* Ref.get(registration.state)
                    const subscribers = Math.max(0, state.subscribers - 1)
                    yield* Ref.set(registration.state, { ...state, subscribers })
                    if (subscribers !== 0) return false
                    if ((yield* Ref.get(registrations)).get(keyFor(sessionId, viewId)) !== registration) {
                      return false
                    }
                    yield* runtime.tryWithBusyResident(
                      sessionId,
                      `display-close:${viewId}`,
                      (entry) => entry.session.displayView.close(viewId),
                    )
                    yield* detachUnlocked(registration)
                    yield* Ref.update(registrations, (all) => {
                      if (all.get(keyFor(sessionId, viewId)) !== registration) return all
                      const next = new Map(all)
                      next.delete(keyFor(sessionId, viewId))
                      return next
                    })
                    return true
                  }),
                )
                if (!shouldRemove) return
              }).pipe(Effect.catchAll(() => Effect.void)),
            )
            const initial = Option.toArray(admission.initial)
            return Stream.concat(Stream.fromIterable(initial), Stream.fromQueue(queue))
          }),
        )

      const materialize = (
        sessionId: string,
        viewId: string,
        shape?: DisplayViewShape,
      ): Effect.Effect<DisplayViewStateEvent, SessionError> =>
        Effect.suspend(() =>
          Effect.gen(function* () {
            const key = keyFor(sessionId, viewId)
            const existing = (yield* Ref.get(registrations)).get(key)
            if (!existing && !shape) {
              return yield* new DisplayViewNotOpen({ sessionId, viewId })
            }
            const registration = existing ?? (yield* getOrCreate(sessionId, viewId, shape!))
            const result = yield* registration.serialize.withPermits(1)(
              Effect.gen(function* () {
                // The final subscriber may have removed this registration
                // while materialization waited for its lock.
                if ((yield* Ref.get(registrations)).get(key) !== registration) {
                  return Option.none<DisplayViewStateEvent>()
                }
                if (shape) {
                  yield* Ref.update(registration.state, (state) => ({ ...state, shape }))
                }
                return Option.some(
                  yield* runtime.withSession(
                    sessionId,
                    `display-materialize:${viewId}`,
                    (entry, generation) => attachUnlocked(registration, entry, generation, true),
                  ),
                )
              }),
            )
            return yield* Option.match(result, {
              onNone: () => materialize(sessionId, viewId, shape),
              onSome: Effect.succeed,
            })
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

      return {
        getDisplayViewStream,
        requestDisplayViewSnapshot: (sessionId, viewId) => materialize(sessionId, viewId),
        setDisplayViewShape: (sessionId, viewId, shape) => materialize(sessionId, viewId, shape),
      }
    }),
  )
