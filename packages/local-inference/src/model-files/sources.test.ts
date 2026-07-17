import * as BunContext from "@effect/platform-bun/BunContext"
import * as FileSystem from "@effect/platform/FileSystem"
import * as Path from "@effect/platform/Path"
import { describe, expect, it } from "vitest"
import { Chunk, Effect, Option, Stream } from "effect"
import {
  ModelArtifactKey,
  ModelFileSourceId,
  Sha256Digest,
  SourceFileKey,
} from "./identity"
import { sha256File } from "./platform"
import { makeDirectoryModelSource, makeMagnitudeModelSource } from "./sources"

describe("model-file sources", () => {
  it("keeps directory discovery contained and resolves the exact discovered set", async () => {
    const result = await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const fs = yield* FileSystem.FileSystem
      const path = yield* Path.Path
      const temporary = yield* fs.makeTempDirectoryScoped()
      const root = path.join(temporary, "models")
      const outside = path.join(temporary, "outside.gguf")
      yield* fs.makeDirectory(root)
      yield* fs.writeFileString(path.join(root, "model.gguf"), "model")
      yield* fs.writeFileString(outside, "outside")
      yield* fs.symlink(outside, path.join(root, "escape.gguf"))

      const source = yield* makeDirectoryModelSource({
        id: ModelFileSourceId.make("directory-test"),
        label: Option.none(),
        root,
        recursive: true,
        followSymlinks: true,
        maxDepth: 8,
        ignore: Option.none(),
      })
      const events = Chunk.toReadonlyArray(yield* Stream.runCollect(source.discover({ refresh: "full" })))
      const setEvent = events.find((event) => event._tag === "FileSet")
      const issue = events.find((event) => event._tag === "Issue" && event.issue.code === "unsafe_path")
      if (setEvent?._tag !== "FileSet") return yield* Effect.dieMessage("expected discovered file set")
      const resolved = yield* source.resolve(setEvent.set.id)
      return { set: setEvent.set, resolved: resolved.set, issue }
    }).pipe(Effect.provide(BunContext.layer))))

    expect(result.set.entries.map(({ relativePath }) => relativePath)).toEqual(["model.gguf"])
    expect(result.resolved.id).toBe(result.set.id)
    expect(result.resolved.entries[0]?.key).toBe(result.set.entries[0]?.key)
    expect(result.issue?._tag).toBe("Issue")
  })

  it("emits one source entry when a symlink and its target name the same physical file", async () => {
    const entries = await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const fs = yield* FileSystem.FileSystem
      const path = yield* Path.Path
      const temporary = yield* fs.makeTempDirectoryScoped()
      const root = path.join(temporary, "hub")
      const blobs = path.join(root, "model", "blobs")
      const snapshot = path.join(root, "model", "snapshots", "revision")
      yield* fs.makeDirectory(blobs, { recursive: true })
      yield* fs.makeDirectory(snapshot, { recursive: true })
      const blob = path.join(blobs, "content-hash")
      yield* fs.writeFileString(blob, "model")
      yield* fs.symlink(blob, path.join(snapshot, "model.gguf"))

      const source = yield* makeDirectoryModelSource({
        id: ModelFileSourceId.make("directory-deduplication-test"),
        label: Option.none(),
        root,
        recursive: true,
        followSymlinks: true,
        maxDepth: 8,
        ignore: Option.none(),
      })
      const events = Chunk.toReadonlyArray(yield* Stream.runCollect(source.discover({ refresh: "full" })))
      return events.flatMap((event) => event._tag === "FileSet" ? event.set.entries : [])
    }).pipe(Effect.provide(BunContext.layer))))

    expect(entries).toHaveLength(1)
    expect(entries[0]?.path).toContain("/blobs/content-hash")
  })

  it("publishes verified content atomically and removes owned artifacts", async () => {
    const result = await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const fs = yield* FileSystem.FileSystem
      const path = yield* Path.Path
      const temporary = yield* fs.makeTempDirectoryScoped()
      const root = path.join(temporary, "owned")
      const staged = path.join(temporary, "staged.gguf")
      yield* fs.writeFileString(staged, "verified-content")
      const digest = yield* sha256File(staged)
      const info = yield* fs.stat(staged)
      const source = yield* makeMagnitudeModelSource({ root, id: Option.none(), label: Option.none() })
      const artifactKey = ModelArtifactKey.make("owned-artifact")
      const modelId = yield* source.publish({
        artifactKey,
        files: [{
          key: SourceFileKey.make("model.gguf"),
          stagedPath: staged,
          publishedRelativePath: "model.gguf",
          role: "primary",
          shardIndex: Option.none(),
          sizeBytes: Number(info.size),
          sha256: Sha256Digest.make(digest),
        }],
        relationships: [],
        origin: Option.none(),
      })
      const eventsBefore = Chunk.toReadonlyArray(yield* Stream.runCollect(source.discover({ refresh: "full" })))
      const published = eventsBefore.find((event) => event._tag === "FileSet")
      const manifestText = published?._tag === "FileSet" && published.set.entries[0]
        ? yield* fs.readFileString(path.join(path.dirname(published.set.entries[0].path), "manifest.json"))
        : undefined
      yield* source.remove(modelId)
      const eventsAfter = Chunk.toReadonlyArray(yield* Stream.runCollect(source.discover({ refresh: "full" })))
      return { modelId, eventsBefore, eventsAfter, manifestText }
    }).pipe(Effect.provide(BunContext.layer))))

    expect(result.modelId).toMatch(/^mf_[a-f0-9]{64}$/)
    expect(result.eventsBefore.some((event) => event._tag === "FileSet")).toBe(true)
    expect(result.manifestText).not.toContain('"shardIndex"')
    expect(result.manifestText).not.toContain('"origin"')
    expect(result.eventsAfter.some((event) => event._tag === "FileSet")).toBe(false)
  })
})
