import { Chunk, Context, Effect, Fiber, Layer, PubSub, Stream } from "effect"
import { describe, expect, it } from "vitest"
import { AcnChanges, AcnChangesLive, AcnStorageChangesLive } from "./changes"
import { ProjectStore } from "./project-store"
import { SessionInspector } from "./session-inspector"

describe("ACN changes", () => {
  it("serves published pokes to subscribers on one stream", async () => {
    const values = await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const changes = yield* AcnChanges
      const collected = yield* changes.stream.pipe(Stream.take(2), Stream.runCollect, Effect.fork)
      yield* Effect.sleep("5 millis")
      yield* changes.publish({ query: "GetModelSlots", revision: 4 })
      yield* changes.publish({ query: "GetSession", key: { sessionId: "s1" } })
      return Chunk.toReadonlyArray(yield* Fiber.join(collected))
    }).pipe(Effect.provide(AcnChangesLive))))

    expect(values).toEqual([
      { query: "GetModelSlots", revision: 4 },
      { query: "GetSession", key: { sessionId: "s1" } },
    ])
  })

  it("names the affected queries of every storage change source", async () => {
    const values = await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const projects = yield* PubSub.unbounded<void>()
      const sessions = yield* PubSub.unbounded<void>()
      const projectStore: Pick<Context.Tag.Service<typeof ProjectStore>, "changes"> = {
        changes: Stream.fromPubSub(projects),
      }
      const sessionInspector: Pick<Context.Tag.Service<typeof SessionInspector>, "changes"> = {
        changes: Stream.fromPubSub(sessions),
      }
      const layer = AcnStorageChangesLive.pipe(Layer.provideMerge(Layer.mergeAll(
        AcnChangesLive,
        Layer.succeed(ProjectStore, projectStore as Context.Tag.Service<typeof ProjectStore>),
        Layer.succeed(SessionInspector, sessionInspector as Context.Tag.Service<typeof SessionInspector>),
      )))
      const context = yield* Layer.build(layer)
      const changes = Context.get(context, AcnChanges)
      const collected = yield* changes.stream.pipe(Stream.take(5), Stream.runCollect, Effect.fork)
      yield* Effect.sleep("5 millis")
      yield* PubSub.publish(projects, undefined)
      yield* PubSub.publish(sessions, undefined)
      return Chunk.toReadonlyArray(yield* Fiber.join(collected))
    })))

    expect(values).toEqual(expect.arrayContaining([
      { query: "ListProjects" },
      { query: "InspectProject" },
      { query: "ListSessions" },
      { query: "ListRecentSessionDirectories" },
      { query: "GetSession" },
    ]))
  })
})
