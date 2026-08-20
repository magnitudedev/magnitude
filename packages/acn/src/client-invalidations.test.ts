import { Chunk, Effect, Stream } from "effect"
import { describe, expect, it } from "vitest"
import { mergeClientInvalidations } from "./client-invalidations"

describe("client invalidation transport", () => {
  it("tags each independent invalidation domain on one stream", async () => {
    const values = await Effect.runPromise(mergeClientInvalidations({
      mirrors: Stream.make({ _tag: "changed" as const, id: "ModelSlots", revision: 4 }),
      projects: Stream.make(undefined),
      sessions: Stream.make(undefined),
    }).pipe(Stream.runCollect))

    expect(Chunk.toReadonlyArray(values)).toEqual(expect.arrayContaining([
      { _tag: "MirroredState", invalidation: { _tag: "changed", id: "ModelSlots", revision: 4 } },
      { _tag: "Projects" },
      { _tag: "Sessions" },
    ]))
  })
})
