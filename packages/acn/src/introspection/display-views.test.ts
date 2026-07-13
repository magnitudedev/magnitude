import { Effect, Fiber, Stream } from "effect"
import { describe, expect, it } from "vitest"
import type { DisplayViewShape } from "@magnitudedev/protocol"
import { AcnDisplayViewIntrospector, AcnDisplayViewIntrospectorLive } from "./display-views"

const rootShape: DisplayViewShape = {
  timelines: {
    root: {
      kind: "tail",
      limit: 40,
      live: true,
      presentation: "default",
    },
  },
}

describe("AcnDisplayViewIntrospector", () => {
  it("tracks display stream subscribers by session and view", async () => {
    const program = Effect.gen(function* () {
      const introspector = yield* AcnDisplayViewIntrospector

      yield* introspector.setShape("session-a", "view-main", rootShape)
      yield* introspector.openStream("session-a", "view-main")
      yield* introspector.openStream("session-a", "view-main")
      const during = yield* introspector.current("session-a")

      yield* introspector.closeStream("session-a", "view-main")
      const afterOneClose = yield* introspector.current("session-a")

      yield* introspector.closeStream("session-a", "view-main")
      const afterAllClosed = yield* introspector.current("session-a")

      return { during, afterOneClose, afterAllClosed }
    }).pipe(Effect.provide(AcnDisplayViewIntrospectorLive))

    const result = await Effect.runPromise(program)

    expect(result.during).toHaveLength(1)
    expect(result.during[0].subscriberCount).toBe(2)
    expect(result.during[0].shape).toEqual(rootShape)
    expect(result.during[0].openedAt).toBeGreaterThan(0)
    expect(result.afterOneClose[0].subscriberCount).toBe(1)
    expect(result.afterAllClosed).toHaveLength(1)
    expect(result.afterAllClosed[0].subscriberCount).toBe(0)
    expect(result.afterAllClosed[0].shape).toEqual(rootShape)
  })

  it("keeps shape state until the view is closed", async () => {
    const program = Effect.gen(function* () {
      const introspector = yield* AcnDisplayViewIntrospector

      yield* introspector.setShape("session-a", "view-main", rootShape)
      yield* introspector.resync("session-a", "view-main")
      const beforeClose = yield* introspector.current("session-a")

      yield* introspector.closeView("session-a", "view-main")
      const afterClose = yield* introspector.current("session-a")

      return { beforeClose, afterClose }
    }).pipe(Effect.provide(AcnDisplayViewIntrospectorLive))

    const result = await Effect.runPromise(program)

    expect(result.beforeClose).toHaveLength(1)
    expect(result.beforeClose[0].shape).toEqual(rootShape)
    expect(result.beforeClose[0].resyncCount).toBe(1)
    expect(result.beforeClose[0].shapeUpdatedAt).toBeGreaterThan(0)
    expect(result.afterClose).toHaveLength(0)
  })

  it("filters current views by session", async () => {
    const program = Effect.gen(function* () {
      const introspector = yield* AcnDisplayViewIntrospector

      yield* introspector.setShape("session-a", "view-a", rootShape)
      yield* introspector.setShape("session-b", "view-b", rootShape)

      return yield* introspector.current("session-a")
    }).pipe(Effect.provide(AcnDisplayViewIntrospectorLive))

    const result = await Effect.runPromise(program)

    expect(result.map((view) => view.sessionId)).toEqual(["session-a"])
    expect(result.map((view) => view.viewId)).toEqual(["view-a"])
  })

  it("emits changes for the changed session only", async () => {
    const program = Effect.gen(function* () {
      const introspector = yield* AcnDisplayViewIntrospector
      const sessionAFiber = yield* introspector.changes("session-a").pipe(
        Stream.take(1),
        Stream.runCollect,
        Effect.fork,
      )
      const sessionBFiber = yield* introspector.changes("session-b").pipe(
        Stream.take(1),
        Stream.runCollect,
        Effect.fork,
      )

      yield* Effect.sleep("10 millis")
      yield* introspector.setShape("session-a", "view-a", rootShape)

      const sessionAChanges = yield* Fiber.join(sessionAFiber)
      const sessionBDone = yield* Fiber.poll(sessionBFiber)
      yield* Fiber.interrupt(sessionBFiber)

      return { sessionAChanges, sessionBDone }
    }).pipe(Effect.provide(AcnDisplayViewIntrospectorLive))

    const result = await Effect.runPromise(program)

    expect(result.sessionAChanges.length).toBe(1)
    expect(result.sessionBDone._tag).toBe("None")
  })
})
