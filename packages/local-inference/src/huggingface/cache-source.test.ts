import * as BunContext from "@effect/platform-bun/BunContext"
import * as FileSystem from "@effect/platform/FileSystem"
import * as Path from "@effect/platform/Path"
import { describe, expect, it } from "vitest"
import { Chunk, Effect, Option, Stream } from "effect"
import { makeHuggingFaceCacheSource } from "./cache-source"

describe("Hugging Face cache source", () => {
  it("discovers the real snapshot/blob link topology and rejects escaping links", async () => {
    const events = await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const fs = yield* FileSystem.FileSystem
      const path = yield* Path.Path
      const temporary = yield* fs.makeTempDirectoryScoped()
      const root = path.join(temporary, "cache")
      const repository = path.join(root, "models--owner--model")
      const blobs = path.join(repository, "blobs")
      const commit = "a".repeat(40)
      const snapshot = path.join(repository, "snapshots", commit)
      yield* fs.makeDirectory(snapshot, { recursive: true })
      yield* fs.makeDirectory(blobs, { recursive: true })
      const blob = path.join(blobs, "model-blob")
      yield* fs.writeFileString(blob, "gguf")
      yield* fs.symlink(blob, path.join(snapshot, "model.gguf"))
      const outside = path.join(temporary, "outside.gguf")
      yield* fs.writeFileString(outside, "outside")
      yield* fs.symlink(outside, path.join(snapshot, "escape.gguf"))

      const source = yield* makeHuggingFaceCacheSource({ root, label: Option.none() })
      return Chunk.toReadonlyArray(yield* Stream.runCollect(source.discover({ refresh: "full" })))
    }).pipe(Effect.provide(BunContext.layer))))

    const set = events.find((event) => event._tag === "FileSet")
    const escape = events.find((event) => event._tag === "Issue" && event.issue.code === "unsafe_path")
    expect(set?._tag).toBe("FileSet")
    if (set?._tag !== "FileSet") return
    expect(set.set.entries.map(({ relativePath }) => relativePath)).toEqual(["model.gguf"])
    expect(Option.getOrNull(set.set.origin)?.repository).toBe("owner/model")
    expect(Option.getOrNull(Option.getOrNull(set.set.origin)?.revision ?? Option.none())).toBe("a".repeat(40))
    expect(escape?._tag).toBe("Issue")
  })
})
