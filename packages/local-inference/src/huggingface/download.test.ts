import { createHash } from "node:crypto"
import * as BunContext from "@effect/platform-bun/BunContext"
import * as FileSystem from "@effect/platform/FileSystem"
import * as Path from "@effect/platform/Path"
import { Chunk, Effect, Fiber, Option, Stream } from "effect"
import { describe, expect, it } from "vitest"
import { ModelFileSourceId, Sha256Digest } from "../model-files"
import { HuggingFaceLfsContent, StorageCapacity } from "./contracts"
import { makeHuggingFaceArtifactId } from "./artifact-identity"
import { makeHuggingFaceDownloadFromUpstream } from "./download"
import { HuggingFaceArtifactId, HuggingFaceCommitId, HuggingFaceFilePath, HuggingFaceRepositoryId, HuggingFaceRevision } from "./identity"
import type { HuggingFaceUpstreamApi } from "./upstream"

describe("Hugging Face download", () => {
  it("emits logical-byte progress and publishes one installation manifest", async () => {
    const result = await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const fs = yield* FileSystem.FileSystem
      const path = yield* Path.Path
      const temporary = yield* fs.makeTempDirectoryScoped()
      const cacheRoot = path.join(temporary, "cache")
      const installationRoot = path.join(temporary, "installations")
      const bytes = new TextEncoder().encode("abcdefgh")
      const digest = Sha256Digest.make(createHash("sha256").update(bytes).digest("hex"))
      const file = { path: HuggingFaceFilePath.make("model.gguf"), role: "primary" as const, shardIndex: Option.none<number>(), sizeBytes: bytes.length, content: new HuggingFaceLfsContent({ sha256: digest }) }
      const identity = { repository: HuggingFaceRepositoryId.make("owner/model"), commit: HuggingFaceCommitId.make("a".repeat(40)), files: [file], relationships: [] }
      const artifact = { id: makeHuggingFaceArtifactId(identity), requestedRevision: HuggingFaceRevision.make("main"), ...identity, totalBytes: bytes.length }
      const upstream: HuggingFaceUpstreamApi = {
        searchModels: () => Stream.empty,
        resolveRevision: () => Effect.die("unused"),
        pathsInfo: () => Effect.die("unused"),
        downloadToCache: ({ cacheDir, commit }) => Effect.gen(function* () {
          const repositoryRoot = path.join(cacheDir, "models--owner--model")
          const blob = path.join(repositoryRoot, "blobs", digest)
          const incomplete = `${blob}.incomplete`
          const pointer = path.join(repositoryRoot, "snapshots", commit, "model.gguf")
          yield* fs.makeDirectory(path.dirname(blob), { recursive: true })
          yield* fs.makeDirectory(path.dirname(pointer), { recursive: true })
          yield* fs.writeFile(incomplete, bytes.slice(0, 3))
          yield* Effect.sleep("150 millis")
          yield* fs.writeFile(incomplete, bytes)
          yield* fs.rename(incomplete, blob)
          yield* fs.symlink(blob, pointer)
          return pointer
        }),
      }
      const download = yield* makeHuggingFaceDownloadFromUpstream({ store: { cacheRoot, installationRoot, sourceId: ModelFileSourceId.make("huggingface-cache") }, reserveBytes: 0, progressIntervalMillis: 10, upstream })
      const events = Chunk.toReadonlyArray(yield* Stream.runCollect(download.download(artifact)))
      return { events, manifests: yield* fs.readDirectory(installationRoot) }
    }).pipe(
      Effect.provideService(StorageCapacity, { availableBytes: () => Effect.succeed(1_000_000) }),
      Effect.provide(BunContext.layer),
    )))

    expect(result.events.map(({ _tag }) => _tag)).toContain("CheckingSpace")
    expect(result.events.map(({ _tag }) => _tag)).toContain("Downloading")
    expect(result.events.map(({ _tag }) => _tag)).toContain("Verifying")
    expect(result.events.at(-1)?._tag).toBe("Ready")
    const downloading = result.events.filter((event) => event._tag === "Downloading")
    expect(downloading.some((event) => event.file.completedBytes === 3)).toBe(true)
    expect(result.manifests).toHaveLength(1)
  })

  it("interrupts work and removes the active incomplete file", async () => {
    const result = await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const fs = yield* FileSystem.FileSystem
      const path = yield* Path.Path
      const temporary = yield* fs.makeTempDirectoryScoped()
      const cacheRoot = path.join(temporary, "cache")
      const installationRoot = path.join(temporary, "installations")
      const digest = Sha256Digest.make("d".repeat(64))
      const file = { path: HuggingFaceFilePath.make("model.gguf"), role: "primary" as const, shardIndex: Option.none<number>(), sizeBytes: 8, content: new HuggingFaceLfsContent({ sha256: digest }) }
      const identity = { repository: HuggingFaceRepositoryId.make("owner/model"), commit: HuggingFaceCommitId.make("a".repeat(40)), files: [file], relationships: [] }
      const artifact = { id: makeHuggingFaceArtifactId(identity), requestedRevision: HuggingFaceRevision.make("main"), ...identity, totalBytes: 8 }
      const incomplete = path.join(cacheRoot, "models--owner--model", "blobs", `${digest}.incomplete`)
      const upstream: HuggingFaceUpstreamApi = {
        searchModels: () => Stream.empty,
        resolveRevision: () => Effect.die("unused"),
        pathsInfo: () => Effect.die("unused"),
        downloadToCache: () => Effect.gen(function* () {
          yield* fs.makeDirectory(path.dirname(incomplete), { recursive: true })
          yield* fs.writeFile(incomplete, new Uint8Array([1, 2, 3]))
          return yield* Effect.never
        }),
      }
      const download = yield* makeHuggingFaceDownloadFromUpstream({ store: { cacheRoot, installationRoot, sourceId: ModelFileSourceId.make("huggingface-cache") }, reserveBytes: 0, progressIntervalMillis: 100, upstream })
      const fiber = yield* Stream.runDrain(download.download(artifact)).pipe(Effect.fork)
      yield* Effect.sleep("150 millis")
      yield* Fiber.interrupt(fiber)
      return {
        incompleteExists: yield* fs.exists(incomplete),
        manifests: yield* fs.readDirectory(installationRoot),
      }
    }).pipe(
      Effect.provideService(StorageCapacity, { availableBytes: () => Effect.succeed(1_000_000) }),
      Effect.provide(BunContext.layer),
    )))

    expect(result.incompleteExists).toBe(false)
    expect(result.manifests).toEqual([])
  })

  it("rejects a forged artifact identity before touching the cache", async () => {
    const result = await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const fs = yield* FileSystem.FileSystem
      const path = yield* Path.Path
      const temporary = yield* fs.makeTempDirectoryScoped()
      const cacheRoot = path.join(temporary, "cache")
      const installationRoot = path.join(temporary, "installations")
      const file = { path: HuggingFaceFilePath.make("model.gguf"), role: "primary" as const, shardIndex: Option.none<number>(), sizeBytes: 8, content: new HuggingFaceLfsContent({ sha256: Sha256Digest.make("d".repeat(64)) }) }
      const artifact = { id: HuggingFaceArtifactId.make(`hf_${"f".repeat(64)}`), repository: HuggingFaceRepositoryId.make("owner/model"), requestedRevision: HuggingFaceRevision.make("main"), commit: HuggingFaceCommitId.make("a".repeat(40)), files: [file], relationships: [], totalBytes: 8 }
      const upstream: HuggingFaceUpstreamApi = {
        searchModels: () => Stream.empty,
        resolveRevision: () => Effect.die("unused"),
        pathsInfo: () => Effect.die("unused"),
        downloadToCache: () => Effect.die("must not download"),
      }
      const download = yield* makeHuggingFaceDownloadFromUpstream({ store: { cacheRoot, installationRoot, sourceId: ModelFileSourceId.make("huggingface-cache") }, reserveBytes: 0, progressIntervalMillis: 100, upstream })
      const exit = yield* Stream.runDrain(download.download(artifact)).pipe(Effect.either)
      return { exit, cacheExists: yield* fs.exists(cacheRoot) }
    }).pipe(
      Effect.provideService(StorageCapacity, { availableBytes: () => Effect.succeed(1_000_000) }),
      Effect.provide(BunContext.layer),
    )))

    expect(result.exit._tag).toBe("Left")
    if (result.exit._tag === "Left") expect(result.exit.left._tag).toBe("HuggingFaceArtifactInvalidError")
    expect(result.cacheExists).toBe(false)
  })
})
