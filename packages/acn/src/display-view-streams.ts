import { Context, Effect, Exit, Layer, Stream, SynchronizedRef } from "effect"
import {
  DisplayViewNotOpen,
  SessionOperationFailed,
  sameDisplayViewShape,
  type SessionError,
  type DisplayViewShape,
  type StreamEvent as ProtocolStreamEvent,
} from "@magnitudedev/protocol"
import { AgentRuntime } from "./agent-runtime"
import { makeDisplayViewStream, type DisplayViewSource, type DisplayViewStreamHandle } from "./display-view-stream"
import { formatUnknownCause } from "./session-errors"
import type { RuntimeEntry } from "./session-types"

export interface DisplayViewStreamsApi {
  readonly getDisplayViewStream: (
    sessionId: string,
    viewId: string,
    shape: DisplayViewShape,
  ) => Stream.Stream<ProtocolStreamEvent, SessionError>
  readonly requestDisplayViewSnapshot: (sessionId: string, viewId: string) => Effect.Effect<"ok", SessionError>
  readonly setDisplayViewShape: (sessionId: string, viewId: string, shape: DisplayViewShape) => Effect.Effect<"ok", SessionError>
  readonly closeDisplayView: (sessionId: string, viewId: string) => Effect.Effect<"ok", SessionError>
}

export class DisplayViewStreams extends Context.Tag("DisplayViewStreams")<
  DisplayViewStreams,
  DisplayViewStreamsApi
>() {}

interface DisplayViewStreamRegistration {
  readonly sessionId: string
  readonly viewId: string
  readonly shape: DisplayViewShape
  readonly handle: DisplayViewStreamHandle
  readonly refCount: number
}

type ReleaseHandleDecision =
  | { readonly _tag: "ignore" }
  | { readonly _tag: "decremented" }
  | { readonly _tag: "close"; readonly registration: DisplayViewStreamRegistration }

const displayViewOperationFailed = (
  sessionId: string,
  viewId: string,
  operation: string
) =>
  (cause: unknown): SessionError =>
    new SessionOperationFailed({
      operation: `DisplayView.${operation}`,
      reason: `${sessionId}/${viewId}: ${formatUnknownCause(cause)}`,
    })

const displayViewNotOpen = (sessionId: string, viewId: string): SessionError =>
  new DisplayViewNotOpen({ sessionId, viewId })

const displayViewSourceFor = (sessionId: string, session: RuntimeEntry["session"]): DisplayViewSource => ({
  on: {
    restoreQueuedMessages: session.on.restoreQueuedMessages,
  },
  displayView: {
    stream: (viewId) =>
      session.displayView.stream(viewId).pipe(
        Stream.mapError(displayViewOperationFailed(sessionId, viewId, "stream"))
      ),
    snapshot: (viewId) =>
      session.displayView.snapshot(viewId).pipe(
        Effect.mapError(displayViewOperationFailed(sessionId, viewId, "snapshot"))
      ),
    setShape: (viewId, shape) =>
      session.displayView.setShape(viewId, shape).pipe(
        Effect.mapError(displayViewOperationFailed(sessionId, viewId, "setShape"))
      ),
    close: (viewId) =>
      session.displayView.close(viewId).pipe(
        Effect.mapError(displayViewOperationFailed(sessionId, viewId, "close"))
      ),
  },
})

export const DisplayViewStreamsLive: Layer.Layer<DisplayViewStreams, never, AgentRuntime> =
  Layer.scoped(
    DisplayViewStreams,
    Effect.gen(function* () {
      const runtime = yield* AgentRuntime
      const registrationsRef = yield* SynchronizedRef.make<ReadonlyMap<string, DisplayViewStreamRegistration>>(
        new Map()
      )

      const viewKey = (sessionId: string, viewId: string) => JSON.stringify([sessionId, viewId])

      const closeHandle = (
        handle: DisplayViewStreamHandle,
      ) => handle.close

      const closeAndRelease = (registration: DisplayViewStreamRegistration) =>
        closeHandle(registration.handle).pipe(
          Effect.ensuring(runtime.releaseEntry(registration.sessionId)),
        )

      const finalizeCloseAndRelease = (registration: DisplayViewStreamRegistration) =>
        closeAndRelease(registration).pipe(
          Effect.catchAll((error) =>
            Effect.logWarning("Failed to close display view during release").pipe(
              Effect.annotateLogs({
                sessionId: registration.sessionId,
                viewId: registration.viewId,
                error: formatUnknownCause(error),
              })
            )
          ),
        )

      const openHandle = Effect.fn("acn.display-view-streams.open-handle")(function* (
        sessionId: string,
        viewId: string,
        shape: DisplayViewShape,
      ) {
        const key = viewKey(sessionId, viewId)
        return yield* SynchronizedRef.modifyEffect(
          registrationsRef,
          (registrations) => {
            const existing = registrations.get(key)
            if (existing) {
              if (sameDisplayViewShape(existing.shape, shape)) {
                return Effect.as(runtime.touchEntry(sessionId), [existing.handle, registrations] as const)
              }

              const next = new Map(registrations)
              return Effect.as(
                existing.handle.setShape(shape).pipe(
                  Effect.zipRight(runtime.touchEntry(sessionId)),
                ),
                [existing.handle, next.set(key, { ...existing, shape })] as const
              )
            }

            return Effect.gen(function* () {
              const entry = yield* runtime.requireOrStart(sessionId)
              const displaySource = displayViewSourceFor(sessionId, entry.session)
              yield* runtime.retainEntry(sessionId)

              return yield* Effect.gen(function* () {
                yield* displaySource.displayView.setShape(viewId, shape)

                const handle = yield* makeDisplayViewStream({
                  source: displaySource,
                  viewId,
                })

                return [
                  handle,
                  new Map(registrations).set(key, {
                    sessionId,
                    viewId,
                    shape,
                    handle,
                    refCount: 0,
                  }),
                ] as const
              }).pipe(
                Effect.onExit((exit) =>
                  Exit.isFailure(exit)
                    ? runtime.releaseEntry(sessionId)
                    : Effect.void
                )
              )
            })
          },
        )
      })

      const getHandleForStream = Effect.fn("acn.display-view-streams.get-stream-handle")(function* (
        sessionId: string,
        viewId: string,
        shape: DisplayViewShape,
      ) {
        yield* openHandle(sessionId, viewId, shape)
        const key = viewKey(sessionId, viewId)
        return yield* SynchronizedRef.modifyEffect(
          registrationsRef,
          (registrations) => {
            const existing = registrations.get(key)
            if (!existing) {
              return Effect.fail(displayViewNotOpen(sessionId, viewId))
            }
            const next = new Map(registrations)
            next.set(key, { ...existing, refCount: existing.refCount + 1 })
            return Effect.succeed([existing.handle, next] as const)
          },
        )
      })

      const releaseHandle = Effect.fn("acn.display-view-streams.release-handle")(function* (
        sessionId: string,
        viewId: string,
        handle: DisplayViewStreamHandle,
      ) {
        const key = viewKey(sessionId, viewId)
        const decision = yield* SynchronizedRef.modify(registrationsRef, (registrations): readonly [
          ReleaseHandleDecision,
          ReadonlyMap<string, DisplayViewStreamRegistration>,
        ] => {
          const registration = registrations.get(key)
          if (!registration || registration.handle !== handle) {
            return [{ _tag: "ignore" }, registrations] as const
          }

          if (registration.refCount > 1) {
            const next = new Map(registrations)
            next.set(key, { ...registration, refCount: registration.refCount - 1 })
            return [{ _tag: "decremented" }, next] as const
          }

          const next = new Map(registrations)
          next.delete(key)
          return [{ _tag: "close", registration }, next] as const
        })

        if (decision._tag === "close") {
          yield* finalizeCloseAndRelease(decision.registration)
        }
      })

      yield* Effect.addFinalizer(() =>
        Effect.gen(function* () {
          const registrations = yield* SynchronizedRef.get(registrationsRef)
          yield* SynchronizedRef.set(registrationsRef, new Map())
          yield* Effect.forEach(
            registrations.values(),
            finalizeCloseAndRelease,
            { discard: true }
          )
        }),
      )

      return {
        getDisplayViewStream: (sessionId, viewId, shape) =>
          Stream.unwrap(
            Effect.gen(function* () {
              const handle = yield* getHandleForStream(sessionId, viewId, shape)
              return handle.stream.pipe(
                Stream.ensuring(releaseHandle(sessionId, viewId, handle)),
              )
            }),
          ),
        requestDisplayViewSnapshot: Effect.fn("acn.display-view-streams.request-snapshot")(function* (sessionId, viewId) {
          const key = viewKey(sessionId, viewId)
          const registration = (yield* SynchronizedRef.get(registrationsRef)).get(key)
          if (!registration) {
            return yield* displayViewNotOpen(sessionId, viewId)
          }
          yield* registration.handle.takeSnapshot
          return "ok" as const
        }),
        setDisplayViewShape: Effect.fn("acn.display-view-streams.set-shape")(function* (sessionId, viewId, shape) {
          yield* openHandle(sessionId, viewId, shape)
          return "ok" as const
        }),
        closeDisplayView: Effect.fn("acn.display-view-streams.close")(function* (sessionId, viewId) {
          const key = viewKey(sessionId, viewId)
          const registration = yield* SynchronizedRef.modify(registrationsRef, (registrations) => {
            const current = registrations.get(key)
            if (!current) return [null, registrations] as const

            const next = new Map(registrations)
            next.delete(key)
            return [current, next] as const
          })
          if (registration) {
            yield* closeAndRelease(registration)
          }
          return "ok" as const
        }),
      }
    }),
  )
