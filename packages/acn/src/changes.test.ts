import { Chunk, Context, Deferred, Effect, Fiber, Layer, PubSub, Stream } from "effect"
import { describe, expect, it } from "vitest"
import { AcnChanges, AcnChangesLive, AcnStorageChangesLive } from "./changes"
import { ProjectStore } from "./project-store"

describe("ACN changes", () => {
  it("serves published pokes to subscribers on one stream", async () => {
    const values = await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const changes = yield* AcnChanges
      const collected = yield* changes.stream.pipe(Stream.take(2), Stream.runCollect, Effect.fork)
      yield* Effect.sleep("5 millis")
      yield* changes.publish({ operation: "GetModelSlots" })
      yield* changes.publish({ operation: "InspectProject", key: { projectId: "p1" } })
      return Chunk.toReadonlyArray(yield* Fiber.join(collected))
    }).pipe(Effect.provide(AcnChangesLive))))

    expect(values).toEqual([
      { operation: "GetModelSlots" },
      { operation: "InspectProject", key: { projectId: "p1" } },
    ])
  })

  it("broadens excessive keyed invalidations for a slow subscriber", async () => {
    const values = await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const changes = yield* AcnChanges
      const firstObserved = yield* Deferred.make<void>()
      const releaseFirst = yield* Deferred.make<void>()
      const collected = yield* changes.stream.pipe(
        Stream.tap((change) => change.key === undefined
          ? Effect.void
          : Deferred.succeed(firstObserved, undefined).pipe(
              Effect.zipRight(Deferred.await(releaseFirst)),
            )),
        Stream.take(2),
        Stream.runCollect,
        Effect.fork,
      )
      yield* Effect.sleep("5 millis")
      yield* changes.publish({ operation: "InspectProject", key: { projectId: "first" } })
      yield* Deferred.await(firstObserved)
      yield* Effect.forEach(
        Array.from({ length: 100 }, (_, index) => index),
        (index) => changes.publish({ operation: "InspectProject", key: { projectId: `pending-${index}` } }),
        { discard: true },
      )
      yield* Deferred.succeed(releaseFirst, undefined)
      return Chunk.toReadonlyArray(yield* Fiber.join(collected))
    }).pipe(Effect.provide(AcnChangesLive))))

    expect(values).toEqual([
      { operation: "InspectProject", key: { projectId: "first" } },
      { operation: "InspectProject" },
    ])
  })

  it("names the affected queries of every storage change source", async () => {
    const values = await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const projects = yield* PubSub.unbounded<void>()
      const projectStore: Pick<Context.Tag.Service<typeof ProjectStore>, "changes"> = {
        changes: Stream.fromPubSub(projects),
      }
      const layer = AcnStorageChangesLive.pipe(Layer.provideMerge(Layer.mergeAll(
        AcnChangesLive,
        Layer.succeed(ProjectStore, projectStore as Context.Tag.Service<typeof ProjectStore>),
      )))
      const context = yield* Layer.build(layer)
      const changes = Context.get(context, AcnChanges)
      const collected = yield* changes.stream.pipe(Stream.take(2), Stream.runCollect, Effect.fork)
      yield* Effect.sleep("5 millis")
      yield* PubSub.publish(projects, undefined)
      return Chunk.toReadonlyArray(yield* Fiber.join(collected))
    })))

    expect(values).toEqual(expect.arrayContaining([
      { operation: "ListProjects" },
      { operation: "InspectProject" },
    ]))
  })
})
