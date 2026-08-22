import { Chunk, Effect, Stream } from "effect"
import { describe, expect, it } from "vitest"
import { mergeChanges } from "./changes"

describe("change transport", () => {
  it("names the affected queries of every change source on one stream", async () => {
    const values = await Effect.runPromise(mergeChanges({
      mirrors: Stream.make({ _tag: "changed" as const, id: "GetModelSlots", revision: 4 }),
      projects: Stream.make(undefined),
      sessions: Stream.make(undefined),
    }).pipe(Stream.runCollect))

    expect(Chunk.toReadonlyArray(values)).toEqual(expect.arrayContaining([
      { query: "GetModelSlots", revision: 4 },
      { query: "ListProjects" },
      { query: "InspectProject" },
      { query: "ListSessions" },
      { query: "ListRecentSessionDirectories" },
      { query: "GetSession" },
    ]))
  })
})
