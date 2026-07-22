import { Context, Effect, Layer, PubSub, Ref, Stream } from "effect"
import type { DisplayViewShape } from "@magnitudedev/protocol"

export interface AcnDisplayViewIntrospection {
  readonly sessionId: string
  readonly viewId: string
  readonly subscriberCount: number
  readonly openedAt: number | null
  readonly lastActivityAt: number
  readonly shape: DisplayViewShape | null
  readonly shapeUpdatedAt: number | null
  readonly resyncCount: number
}

export interface AcnDisplayViewIntrospectorApi {
  readonly openStream: (sessionId: string, viewId: string) => Effect.Effect<void>
  readonly closeStream: (sessionId: string, viewId: string) => Effect.Effect<void>
  readonly setShape: (sessionId: string, viewId: string, shape: DisplayViewShape) => Effect.Effect<void>
  readonly resync: (sessionId: string, viewId: string) => Effect.Effect<void>
  readonly current: (sessionId?: string | null) => Effect.Effect<readonly AcnDisplayViewIntrospection[]>
  readonly changes: (sessionId?: string | null) => Stream.Stream<void>
}

export class AcnDisplayViewIntrospector extends Context.Tag("AcnDisplayViewIntrospector")<
  AcnDisplayViewIntrospector,
  AcnDisplayViewIntrospectorApi
>() {}

const now = () => Date.now()
const viewKey = (sessionId: string, viewId: string) => `${sessionId}\u0000${viewId}`

const emptyView = (
  sessionId: string,
  viewId: string,
  timestamp: number,
): AcnDisplayViewIntrospection => ({
  sessionId,
  viewId,
  subscriberCount: 0,
  openedAt: null,
  lastActivityAt: timestamp,
  shape: null,
  shapeUpdatedAt: null,
  resyncCount: 0,
})

const shouldKeep = (view: AcnDisplayViewIntrospection): boolean =>
  view.subscriberCount > 0 || view.shape !== null

export const AcnDisplayViewIntrospectorLive: Layer.Layer<AcnDisplayViewIntrospector> =
  Layer.effect(
    AcnDisplayViewIntrospector,
    Effect.gen(function* () {
      const state = yield* Ref.make<ReadonlyMap<string, AcnDisplayViewIntrospection>>(new Map())
      const changes = yield* PubSub.unbounded<string>()

      const modifyView = (
        sessionId: string,
        viewId: string,
        f: (
          view: AcnDisplayViewIntrospection | null,
          timestamp: number,
        ) => AcnDisplayViewIntrospection | null,
      ) =>
        Ref.update(state, (current) => {
          const timestamp = now()
          const key = viewKey(sessionId, viewId)
          const previous = current.get(key) ?? null
          const next = f(previous, timestamp)
          const updated = new Map(current)
          if (next && shouldKeep(next)) {
            updated.set(key, next)
          } else {
            updated.delete(key)
          }
          return updated
        }).pipe(
          Effect.zipRight(PubSub.publish(changes, sessionId)),
          Effect.asVoid,
        )

      return {
        openStream: (sessionId, viewId) =>
          modifyView(sessionId, viewId, (view, timestamp) => ({
            ...(view ?? emptyView(sessionId, viewId, timestamp)),
            subscriberCount: (view?.subscriberCount ?? 0) + 1,
            openedAt: view?.openedAt ?? timestamp,
            lastActivityAt: timestamp,
          })),
        closeStream: (sessionId, viewId) =>
          modifyView(sessionId, viewId, (view, timestamp) => {
            if (!view) return null
            const subscriberCount = Math.max(0, view.subscriberCount - 1)
            if (subscriberCount === 0) return null
            return {
              ...view,
              subscriberCount,
              lastActivityAt: timestamp,
            }
          }),
        setShape: (sessionId, viewId, shape) =>
          modifyView(sessionId, viewId, (view, timestamp) => ({
            ...(view ?? emptyView(sessionId, viewId, timestamp)),
            shape,
            shapeUpdatedAt: timestamp,
            lastActivityAt: timestamp,
          })),
        resync: (sessionId, viewId) =>
          modifyView(sessionId, viewId, (view, timestamp) =>
            view
              ? {
                  ...view,
                  resyncCount: view.resyncCount + 1,
                  lastActivityAt: timestamp,
                }
              : null
          ),
        current: (sessionId) =>
          Ref.get(state).pipe(
            Effect.map((current) =>
              [...current.values()]
                .filter((view) => !sessionId || view.sessionId === sessionId)
                .sort((left, right) =>
                  left.sessionId === right.sessionId
                    ? left.viewId.localeCompare(right.viewId)
                    : left.sessionId.localeCompare(right.sessionId)
                ),
            ),
          ),
        changes: (sessionId) =>
          Stream.fromPubSub(changes).pipe(
            Stream.filter((changedSessionId) => !sessionId || changedSessionId === sessionId),
            Stream.map(() => undefined),
          ),
      } satisfies AcnDisplayViewIntrospectorApi
    }),
  )
