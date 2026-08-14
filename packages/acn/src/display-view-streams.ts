import { Context, Data, Deferred, Effect, Exit, Fiber, Layer, Option, PubSub, Queue, Ref, Scope, Stream } from "effect"
import {
  DisplayViewNotOpen,
  sameDisplayViewShape,
  SessionOperationFailed,
  type DisplayViewShape,
  type DisplayViewStateEvent,
  type SessionError,
  type StreamEvent,
} from "@magnitudedev/acn-protocol"
import { AcnSubscriptions } from "./acn-subscriptions"
import { AgentRuntime } from "./agent-runtime"
import { formatUnknownCause } from "./session-errors"
import type { RuntimeEntry } from "./session-types"

export interface DisplayViewStreamsApi {
  readonly getDisplayViewStream: (
    sessionId: string,
    viewId: string,
    shape: DisplayViewShape,
    materialize?: boolean,
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

interface SequencedEvent<Event extends StreamEvent = StreamEvent> {
  readonly sequence: number
  readonly event: Event
}

interface Attachment {
  readonly token: string
  readonly generation: number
  readonly shapeRevision: number
  readonly fiber: Fiber.RuntimeFiber<void, unknown>
  readonly latest: Ref.Ref<SequencedEvent<DisplayViewStateEvent>>
}

interface RegistrationState {
  readonly shape: DisplayViewShape
  readonly shapeRevision: number
  readonly attachment: Attachment | null
  readonly subscribers: number
  readonly closing: Deferred.Deferred<void> | null
}

interface ShapeIntent {
  readonly revision: number
  readonly previousShape: DisplayViewShape
  readonly previousRevision: number
}

interface Registration {
  readonly sessionId: string
  readonly viewId: string
  readonly events: PubSub.PubSub<SequencedEvent>
  readonly nextSequence: Ref.Ref<number>
  readonly state: Ref.Ref<RegistrationState>
  readonly serialize: Effect.Semaphore
  readonly shapeAdmissions: Effect.Semaphore
}

class RegistrationRetry extends Data.TaggedError("RegistrationRetry")<{
  readonly wait: Effect.Effect<void>
}> {}

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
          const events = yield* PubSub.unbounded<SequencedEvent>()
          const nextSequence = yield* Ref.make(0)
          const state = yield* Ref.make<RegistrationState>({
            shape,
            shapeRevision: 0,
            attachment: null,
            subscribers: 0,
            closing: null,
          })
          const serialize = yield* Effect.makeSemaphore(1)
          const shapeAdmissions = yield* Effect.makeSemaphore(1)
          return {
            sessionId,
            viewId,
            events,
            nextSequence,
            state,
            serialize,
            shapeAdmissions,
          } satisfies Registration
        })

      const publishEvent = <Event extends StreamEvent>(registration: Registration, event: Event) =>
        Effect.gen(function* () {
          const sequence = yield* Ref.getAndUpdate(registration.nextSequence, (value) => value + 1)
          const sequenced = { sequence, event } satisfies SequencedEvent<Event>
          yield* PubSub.publish(registration.events, sequenced)
          return sequenced
        })

      const getOrCreate = (sessionId: string, viewId: string, shape: DisplayViewShape) =>
        Effect.gen(function* () {
          const key = keyFor(sessionId, viewId)
          const current = (yield* Ref.get(registrations)).get(key)
          // A passive reconnect must not mutate an existing materialized shape.
          // A new stream registration owns its initial requested shape; explicit
          // materialization is handled only after subscriber cleanup is installed.
          if (current) return current
          const candidate = yield* makeRegistration(sessionId, viewId, shape)
          return yield* Ref.modify(registrations, (all) => {
            const winner = all.get(key)
            if (winner) return [winner, all] as const
            return [candidate, new Map(all).set(key, candidate)] as const
          })
        })

      const setShapeIntentUnlocked = (
        registration: Registration,
        shape: DisplayViewShape,
      ): Effect.Effect<ShapeIntent> =>
        Ref.modify(registration.state, (state) => {
          const intent = {
            revision: state.shapeRevision + 1,
            previousShape: state.shape,
            previousRevision: state.shapeRevision,
          } satisfies ShapeIntent
          return [intent, { ...state, shape, shapeRevision: intent.revision }] as const
        })

      const rollbackShapeIntent = (registration: Registration, intent: ShapeIntent) =>
        registration.serialize.withPermits(1)(
          Effect.gen(function* () {
            const key = keyFor(registration.sessionId, registration.viewId)
            if ((yield* Ref.get(registrations)).get(key) !== registration) return
            yield* Ref.update(registration.state, (state) => {
              if (
                state.shapeRevision !== intent.revision
                || state.attachment?.shapeRevision === intent.revision
              ) return state
              return {
                ...state,
                shape: intent.previousShape,
                shapeRevision: intent.previousRevision,
              }
            })
          }),
        )

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

      const removeIfUnused = (registration: Registration) =>
        Effect.gen(function* () {
          const completion = yield* registration.serialize.withPermits(1)(
            Effect.gen(function* () {
              const key = keyFor(registration.sessionId, registration.viewId)
              const state = yield* Ref.get(registration.state)
              const exact = (yield* Ref.get(registrations)).get(key)
              if (
                exact !== registration
                || state.subscribers !== 0
                || state.closing !== null
              ) return null
              const completion = yield* Deferred.make<void>()
              yield* Ref.update(registration.state, (state) => ({
                ...state,
                closing: completion,
              }))
              yield* detachUnlocked(registration).pipe(Effect.catchAllCause(() => Effect.void))
              return completion
            }),
          )
          if (completion === null) return false

          const finalize = registration.serialize.withPermits(1)(
            Effect.gen(function* () {
              const key = keyFor(registration.sessionId, registration.viewId)
              yield* Ref.update(registrations, (current) => {
                if (current.get(key) !== registration) return current
                const next = new Map(current)
                next.delete(key)
                return next
              })
              yield* Deferred.succeed(completion, undefined)
            }),
          )

          yield* runtime
            .tryWithBusyResident(
              registration.sessionId,
              `display-close:${registration.viewId}`,
              ({ session }) => session.displayView.close(registration.viewId),
            )
            .pipe(
              Effect.catchAllCause(() => Effect.void),
              Effect.ensuring(finalize),
            )
          return true
        })

      const releaseSubscriber = (registration: Registration) =>
        registration.serialize.withPermits(1)(
          Ref.update(registration.state, (state) => ({
            ...state,
            subscribers: Math.max(0, state.subscribers - 1),
          })),
        ).pipe(Effect.zipRight(removeIfUnused(registration)))

      const attachUnlocked = (
        registration: Registration,
        entry: RuntimeEntry,
        generation: number,
        refresh = false,
      ): Effect.Effect<SequencedEvent<DisplayViewStateEvent>, SessionError> =>
        Effect.uninterruptible(Effect.gen(function* () {
            const current = yield* Ref.get(registration.state)
            if (current.attachment?.generation === generation) {
              if (!refresh) return yield* Ref.get(current.attachment.latest)
              const attachment = current.attachment
              const snapshot = yield* entry.session.displayView
                .setShapeAndSnapshot(registration.viewId, current.shape)
                .pipe(
                  Effect.mapError(
                    operationFailed(registration.sessionId, registration.viewId, "setShapeAndSnapshot"),
                  ),
                )
              const event = toStateEvent(snapshot)
              yield* Ref.update(registration.state, (state) =>
                state.attachment?.token === attachment.token
                  ? {
                      ...state,
                      attachment: { ...state.attachment, shapeRevision: current.shapeRevision },
                    }
                  : state,
              )
              const sequenced = yield* publishEvent(registration, event)
              yield* Ref.set(attachment.latest, sequenced)
              return sequenced
            }

            if (current.attachment) {
              yield* Fiber.interrupt(current.attachment.fiber)
            }
            const snapshot = yield* entry.session.displayView
              .setShapeAndSnapshot(registration.viewId, current.shape)
              .pipe(
                Effect.mapError(
                  operationFailed(registration.sessionId, registration.viewId, "setShapeAndSnapshot"),
                ),
              )
            const initial = toStateEvent(snapshot)
            const latest = yield* Ref.make<SequencedEvent<DisplayViewStateEvent>>({
              sequence: -1,
              event: initial,
            })
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
                        const state = yield* Ref.get(registration.state)
                        const observed = state.attachment
                        if (observed?.token !== token) return
                        // A runtime stream may already have produced the old shape
                        // while a new admission was being materialized. Never let
                        // that stale state cross the committed shape fence.
                        if (event._tag === "state" && !sameDisplayViewShape(event.shape, state.shape)) return
                        if (event._tag === "state") {
                          const sequenced = yield* publishEvent(registration, event)
                          yield* Ref.set(latest, sequenced)
                        } else {
                          yield* publishEvent(registration, event)
                        }
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
            const fiber = yield* Effect.forkIn(Effect.interruptible(forward), layerScope)
            yield* Ref.set(registration.state, {
              ...current,
              attachment: {
                token,
                generation,
                shapeRevision: current.shapeRevision,
                fiber,
                latest,
              },
            })
            const sequenced = yield* publishEvent(registration, initial)
            yield* Ref.set(latest, sequenced)
            yield* Deferred.succeed(ready, undefined)
            return sequenced
        }))

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
        Effect.gen(function* () {
          const eligible = yield* registration.serialize.withPermits(1)(
            Effect.gen(function* () {
              const exact = (yield* Ref.get(registrations)).get(
                keyFor(registration.sessionId, registration.viewId),
              )
              const state = yield* Ref.get(registration.state)
              return exact === registration
                && state.subscribers > 0
                && state.attachment === null
                && state.closing === null
            }),
          )
          if (!eligible) return
          yield* runtime.tryWithBusyResident(
            registration.sessionId,
            `display-attach:${registration.viewId}`,
            (entry, generation) =>
              registration.serialize.withPermits(1)(
                Effect.gen(function* () {
                  const exact = (yield* Ref.get(registrations)).get(
                    keyFor(registration.sessionId, registration.viewId),
                  )
                  const state = yield* Ref.get(registration.state)
                  if (
                    exact !== registration
                    || state.subscribers === 0
                    || state.attachment !== null
                    || state.closing !== null
                  ) return Option.none<SequencedEvent<DisplayViewStateEvent>>()
                  return Option.some(
                    yield* attachUnlocked(registration, entry, generation),
                  )
                }),
              ),
          )
        })

      const attachBusyRegistrations = Effect.gen(function* () {
        for (const registration of (yield* Ref.get(registrations)).values()) {
          yield* attachIfBusy(registration)
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

      const materialize = (
        sessionId: string,
        viewId: string,
        shape?: DisplayViewShape,
      ): Effect.Effect<SequencedEvent<DisplayViewStateEvent>, SessionError> =>
        Effect.suspend(() =>
          Effect.gen(function* () {
            const key = keyFor(sessionId, viewId)
            const existing = (yield* Ref.get(registrations)).get(key)
            if (!existing && !shape) {
              return yield* new DisplayViewNotOpen({ sessionId, viewId })
            }
            const registration = existing ?? (yield* getOrCreate(sessionId, viewId, shape!))
            const outcome = yield* registration.shapeAdmissions.withPermits(1)(
              Effect.gen(function* () {
                const prepared = yield* registration.serialize.withPermits(1)(
                  Effect.gen(function* () {
                    if ((yield* Ref.get(registrations)).get(key) !== registration) {
                      return { _tag: "Retry" as const, wait: Effect.void }
                    }
                    const state = yield* Ref.get(registration.state)
                    if (state.closing !== null) {
                      return {
                        _tag: "Retry" as const,
                        wait: Deferred.await(state.closing),
                      }
                    }
                    return {
                      _tag: "Ready" as const,
                      shapeIntent: shape
                        ? yield* setShapeIntentUnlocked(registration, shape)
                        : undefined,
                    }
                  }),
                )
                if (prepared._tag === "Retry") return prepared

                const result = yield* runtime.withSession(
                  sessionId,
                  `display-materialize:${viewId}`,
                  (entry, generation) => registration.serialize.withPermits(1)(
                    Effect.gen(function* () {
                      // Runtime acquisition must not hold display serialization:
                      // retirement needs the same lock to detach the old generation.
                      const state = yield* Ref.get(registration.state)
                      if (
                        (yield* Ref.get(registrations)).get(key) !== registration
                        || state.closing !== null
                      ) {
                        return Option.none<SequencedEvent<DisplayViewStateEvent>>()
                      }
                      return Option.some(
                        yield* attachUnlocked(registration, entry, generation, true),
                      )
                    }),
                  ),
                ).pipe(
                  Effect.onExit((exit) =>
                    Exit.isFailure(exit) && prepared.shapeIntent !== undefined
                      ? rollbackShapeIntent(registration, prepared.shapeIntent)
                      : Effect.void,
                  ),
                )
                return Option.match(result, {
                  onNone: () => ({ _tag: "Retry" as const, wait: Effect.void }),
                  onSome: (event) => ({ _tag: "Done" as const, event }),
                })
              }),
            )
            if (outcome._tag === "Retry") {
              yield* outcome.wait
              return yield* materialize(sessionId, viewId, shape)
            }
            return outcome.event
          }),
        )

      const getDisplayViewStream = (
        sessionId: string,
        viewId: string,
        shape: DisplayViewShape,
        materializeShape = false,
      ): Stream.Stream<StreamEvent, SessionError> => {
        const attempt = Stream.unwrapScoped(
          Effect.gen(function* () {
            const registration = yield* Effect.acquireRelease(
              getOrCreate(sessionId, viewId, shape),
              removeIfUnused,
            )
            const admission = yield* Effect.acquireRelease(
              registration.serialize.withPermits(1)(
                Effect.gen(function* () {
                  const exact = (yield* Ref.get(registrations)).get(keyFor(sessionId, viewId))
                  if (exact !== registration) {
                    return { admitted: false as const, wait: Effect.void }
                  }
                  const state = yield* Ref.get(registration.state)
                  if (state.closing !== null) {
                    return {
                      admitted: false as const,
                      wait: Deferred.await(state.closing),
                    }
                  }
                  const queue = yield* PubSub.subscribe(registration.events)
                  yield* Ref.update(registration.state, (state) => ({
                    ...state,
                    subscribers: state.subscribers + 1,
                  }))
                  return {
                    admitted: true as const,
                    initial: state.attachment
                      ? Option.some((yield* Ref.get(state.attachment.latest)).event)
                      : Option.none<DisplayViewStateEvent>(),
                    queue,
                  }
                }),
              ),
              (admitted) => admitted.admitted ? releaseSubscriber(registration) : Effect.void,
            )
            if (!admission.admitted) {
              return yield* new RegistrationRetry({ wait: admission.wait })
            }
            if (materializeShape) {
              const committed = yield* materialize(sessionId, viewId, shape)
              // The queue was subscribed before materialization so no live event
              // is lost. Preserve records above the commit sequence while dropping
              // only pre-commit data below the subscriber admission fence.
              const buffered = yield* registration.serialize.withPermits(1)(
                Queue.takeAll(admission.queue),
              )
              const postFence = Array.from(buffered).filter(
                (item) => item.sequence > committed.sequence,
              )
              return Stream.concat(
                Stream.succeed(committed.event),
                Stream.concat(
                  Stream.fromIterable(postFence).pipe(Stream.map((item) => item.event)),
                  Stream.fromQueue(admission.queue).pipe(Stream.map((item) => item.event)),
                ),
              )
            }

            yield* attachIfBusy(registration)
            return Stream.concat(
              Stream.fromIterable(Option.toArray(admission.initial)),
              Stream.fromQueue(admission.queue).pipe(Stream.map((item) => item.event)),
            )
          }),
        )
        return attempt.pipe(
          Stream.catchTag("RegistrationRetry", ({ wait }) =>
            Stream.unwrap(
              wait.pipe(Effect.as(getDisplayViewStream(sessionId, viewId, shape, materializeShape))),
            ),
          ),
        )
      }

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
        requestDisplayViewSnapshot: (sessionId, viewId) =>
          materialize(sessionId, viewId).pipe(Effect.map((item) => item.event)),
        setDisplayViewShape: (sessionId, viewId, shape) =>
          materialize(sessionId, viewId, shape).pipe(Effect.map((item) => item.event)),
      }
    }),
  )
