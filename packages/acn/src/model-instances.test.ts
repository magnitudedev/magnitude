import { Chunk, Effect, Fiber, Layer, PubSub, Ref, Schema, Stream } from "effect"
import { describe, expect, it } from "vitest"
import { Models } from "@magnitudedev/acn-protocol"
import { IcnInstances } from "@magnitudedev/icn"
import {
  ModelInstancesSnapshot as ModelInstancesSnapshotSchema,
  type ModelInstancesSnapshot,
} from "@magnitudedev/icn-protocol/schemas"
import { AcnChanges, AcnChangesLive } from "./changes"
import { ModelInstances, ModelInstancesLive } from "./model-instances"

const snapshot = (revision: number, state: "Ready" | "Stopped"): ModelInstancesSnapshot =>
  Schema.decodeUnknownSync(ModelInstancesSnapshotSchema)({
  revision,
  instances: [{
    id: "instance-test",
    modelId: "model-test",
    lifecycle: state === "Stopped"
      ? {
          _tag: "Stopped",
          reason: "user_stop",
        }
      : {
          _tag: "Ready",
          allocation: {
            contextWindowTokens: 8_192,
            parallelSequences: 1,
            physicalContextTokens: 8_192,
            memoryDomains: [],
          },
        },
  }],
  })

describe("ACN model instances", () => {
  it("projects native changes and publishes the ACN query identity", async () => {
    const result = await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const current = yield* Ref.make(snapshot(1, "Ready"))
      const nativeChanges = yield* PubSub.unbounded<ModelInstancesSnapshot>()
      const dependencies = Layer.mergeAll(
        AcnChangesLive,
        Layer.succeed(IcnInstances, IcnInstances.of({
          get: Ref.get(current),
          changes: Stream.fromPubSub(nativeChanges),
          initialized: Effect.succeed(true),
          refresh: Effect.void,
        })),
      )
      const layer = ModelInstancesLive.pipe(Layer.provideMerge(dependencies))
      return yield* Effect.gen(function* () {
        const instances = yield* ModelInstances
        const changes = yield* AcnChanges
        const projected = yield* instances.changes.pipe(Stream.take(1), Stream.runCollect, Effect.fork)
        const pokes = yield* changes.stream.pipe(Stream.take(1), Stream.runCollect, Effect.fork)

        yield* Effect.sleep("10 millis")
        yield* PubSub.publish(nativeChanges, snapshot(2, "Stopped"))

        return {
          initial: yield* instances.state,
          projected: Chunk.toReadonlyArray(yield* Fiber.join(projected)),
          pokes: Chunk.toReadonlyArray(yield* Fiber.join(pokes)),
        }
      }).pipe(Effect.provide(layer))
    })))

    expect(result.initial.instances[0]?.residency._tag).toBe("Ready")
    expect(result.projected[0]?.instances[0]?.residency._tag).toBe("Unloaded")
    expect(result.pokes).toEqual([{ query: Models.GetInstances.name }])
  })
})
