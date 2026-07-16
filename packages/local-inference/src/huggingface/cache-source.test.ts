import { createHash } from "node:crypto"
import * as BunContext from "@effect/platform-bun/BunContext"
import * as FileSystem from "@effect/platform/FileSystem"
import * as Path from "@effect/platform/Path"
import { Chunk, Effect, Option, Schema, Stream } from "effect"
import { describe, expect, it } from "vitest"
import { ModelFileSourceId, Sha256Digest } from "../model-files"
import { HuggingFaceLfsContent } from "./contracts"
import { makeHuggingFaceArtifactId } from "./artifact-identity"
import { makeHuggingFaceCacheSource } from "./cache-source"
import { HuggingFaceCommitId, HuggingFaceFilePath, HuggingFaceRepositoryId, HuggingFaceRevision } from "./identity"
import { HuggingFaceInstallationManifestJson } from "./installation-schema"

describe("Hugging Face cache source", () => {
  it("discovers only manifest-backed installations and preserves roles", async () => {
    const events = await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const fs = yield* FileSystem.FileSystem
      const path = yield* Path.Path
      const temporary = yield* fs.makeTempDirectoryScoped()
      const cacheRoot = path.join(temporary, "cache")
      const installationRoot = path.join(temporary, "installations")
      const repository = HuggingFaceRepositoryId.make("owner/model")
      const commit = HuggingFaceCommitId.make("a".repeat(40))
      const relative = HuggingFaceFilePath.make(`models--owner--model/snapshots/${commit}/model.gguf`)
      const pointer = path.join(cacheRoot, relative)
      const blob = path.join(cacheRoot, "models--owner--model/blobs/blob")
      yield* fs.makeDirectory(path.dirname(pointer), { recursive: true })
      yield* fs.makeDirectory(path.dirname(blob), { recursive: true })
      yield* fs.writeFileString(blob, "gguf")
      yield* fs.symlink(blob, pointer)
      yield* fs.writeFileString(path.join(cacheRoot, "unmanifested.gguf"), "ignored")
      yield* fs.makeDirectory(installationRoot, { recursive: true })
      const digest = Sha256Digest.make(createHash("sha256").update("gguf").digest("hex"))
      const file = { path: HuggingFaceFilePath.make("model.gguf"), role: "primary" as const, shardIndex: Option.none<number>(), sizeBytes: 4, content: new HuggingFaceLfsContent({ sha256: digest }) }
      const identity = { repository, commit, files: [file], relationships: [] }
      const id = makeHuggingFaceArtifactId(identity)
      const artifact = { id, requestedRevision: HuggingFaceRevision.make("main"), ...identity, totalBytes: 4 }
      const encoded = yield* Schema.encode(HuggingFaceInstallationManifestJson)({ version: 1, artifact, files: [{ ...file, snapshotRelativePath: relative }], installedAt: new Date() })
      yield* fs.writeFileString(path.join(installationRoot, `${id}.json`), encoded)

      const source = yield* makeHuggingFaceCacheSource({ store: { cacheRoot, installationRoot, sourceId: ModelFileSourceId.make("huggingface-cache") }, label: Option.none() })
      return Chunk.toReadonlyArray(yield* Stream.runCollect(source.discover({ refresh: "full" })))
    }).pipe(Effect.provide(BunContext.layer))))

    const set = events.find((event) => event._tag === "FileSet")
    expect(events).toHaveLength(1)
    expect(set?._tag).toBe("FileSet")
    if (set?._tag !== "FileSet") return
    expect(set.set.entries).toHaveLength(1)
    expect(Option.getOrNull(set.set.entries[0].declaredRole)).toBe("primary")
    expect(Option.getOrNull(set.set.origin)?.repository).toBe("owner/model")
  })
})
