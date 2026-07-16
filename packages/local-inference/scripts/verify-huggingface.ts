import * as BunContext from "@effect/platform-bun/BunContext"
import * as BunRuntime from "@effect/platform-bun/BunRuntime"
import * as FileSystem from "@effect/platform/FileSystem"
import * as Path from "@effect/platform/Path"
import { Chunk, Effect, Layer, Option, Redacted, Stream } from "effect"
import {
  ModelFileSourceId,
  ModelFileSourceRegistration,
  makeGgufFormat,
  makeModelFileRegistry,
} from "../src/model-files"
import {
  HuggingFaceDownload,
  HuggingFaceLive,
  HuggingFaceRepositoryId,
  HuggingFaceRevision,
  HuggingFaceHub,
  StorageCapacity,
  StorageCapacityLive,
  makeHuggingFaceCacheSource,
  type DownloadProgress,
  type HuggingFaceConnectionOptions,
} from "../src/huggingface"

const repository = HuggingFaceRepositoryId.make(process.env.HF_VERIFY_REPOSITORY ?? "aladar/tiny-random-LlamaForCausalLM-GGUF")
const filePath = process.env.HF_VERIFY_FILE ?? "tiny-random-LlamaForCausalLM.gguf"
const revision = HuggingFaceRevision.make(process.env.HF_VERIFY_REVISION ?? "main")

function assert(condition: unknown, message: string): asserts condition {
  if (!condition) throw new Error(`Hugging Face manual verification failed: ${message}`)
}

const percentage = ({ completedBytes, totalBytes }: { readonly completedBytes: number; readonly totalBytes: number }): number =>
  totalBytes === 0 ? 100 : Math.round((completedBytes / totalBytes) * 10_000) / 100

const delayedLargeResponseFetch = (async (input, init) => {
  const response = await fetch(input, init)
  const length = Number(response.headers.get("content-length") ?? 0)
  if (!response.body || length < 512 * 1024) return response

  const delayedBody = response.body.pipeThrough(new TransformStream<Uint8Array, Uint8Array>({
    async transform(chunk, controller) {
      await new Promise((resolve) => setTimeout(resolve, 30))
      controller.enqueue(chunk)
    },
  }))
  const delayed = new Response(delayedBody, {
    status: response.status,
    statusText: response.statusText,
    headers: response.headers,
  })
  Object.defineProperty(delayed, "url", { value: response.url })
  return delayed
}) as typeof fetch

const filesUnder = (root: string): Effect.Effect<readonly string[], unknown, FileSystem.FileSystem | Path.Path> => Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const path = yield* Path.Path
  if (!(yield* fs.exists(root))) return []
  const entries = yield* fs.readDirectory(root)
  const nested = yield* Effect.forEach(entries, (entry) => Effect.gen(function* () {
    const child = path.join(root, entry)
    const info = yield* fs.stat(child)
    return info.type === "Directory" ? yield* filesUnder(child) : [child]
  }))
  return nested.flat()
})

const connection: HuggingFaceConnectionOptions = {
  hubUrl: Option.fromNullable(process.env.HF_ENDPOINT).pipe(Option.map((value) => new URL(value))),
  token: Option.fromNullable(process.env.HF_TOKEN).pipe(Option.map(Redacted.make)),
  fetch: Option.some(delayedLargeResponseFetch),
}

const program = Effect.scoped(Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const path = yield* Path.Path
  const temporaryRoot = yield* fs.makeTempDirectoryScoped({ prefix: "magnitude-hf-verify-" })
  const store = {
    cacheRoot: path.join(temporaryRoot, "cache"),
    installationRoot: path.join(temporaryRoot, "installations"),
    sourceId: ModelFileSourceId.make("huggingface-manual-verification"),
  }
  const services = Layer.merge(
    HuggingFaceLive({ store, reserveBytes: 0, progressIntervalMillis: 100, connection }),
    StorageCapacityLive,
  )

  const verification = Effect.gen(function* () {
    const hub = yield* HuggingFaceHub
    const download = yield* HuggingFaceDownload
    const capacity = yield* StorageCapacity

    const searchResults = Chunk.toReadonlyArray(yield* Stream.runCollect(hub.searchModels({
      query: "tiny-random-LlamaForCausalLM-GGUF",
      owner: Option.some("aladar"),
      tags: ["gguf"],
      apps: [],
      sort: Option.none(),
      limit: Option.some(10),
    })))
    assert(searchResults.some((model) => model.repository === repository), `search did not return ${repository}`)

    const artifact = yield* hub.resolveArtifact({
      repository,
      revision,
      files: [{ path: filePath, role: "primary", shardIndex: Option.none() }],
      relationships: [],
    })
    assert(artifact.files.length === 1, "artifact did not contain exactly one selected file")
    assert(artifact.files[0]?.path === filePath, "artifact resolved the wrong path")
    assert(String(artifact.commit) !== String(revision), "mutable revision was not pinned to an immutable commit")
    assert(artifact.totalBytes > 0, "artifact size was not resolved")

    yield* fs.makeDirectory(store.cacheRoot, { recursive: true })
    const availableBytes = yield* capacity.availableBytes(store.cacheRoot)
    assert(availableBytes > artifact.totalBytes, "capacity service did not report enough free space")

    const firstEvents = Chunk.toReadonlyArray(yield* Stream.runCollect(download.download(artifact)))
    const firstTags = firstEvents.map((event) => event._tag)
    assert(firstTags[0] === "CheckingSpace", "first download did not begin with CheckingSpace")
    assert(firstTags.includes("Downloading"), "first download did not emit Downloading")
    assert(firstTags.includes("Verifying"), "first download did not emit Verifying")
    assert(firstTags.at(-1) === "Ready", "first download did not end with Ready")
    const downloading = firstEvents.filter((event): event is Extract<DownloadProgress, { readonly _tag: "Downloading" }> => event._tag === "Downloading")
    assert(downloading[0]?.file.completedBytes === 0, "download progress did not begin at zero bytes")
    assert(downloading.at(-1)?.file.completedBytes === artifact.totalBytes, "download progress did not reach the resolved byte total")
    assert(downloading.every((event, index) => index === 0 || event.aggregate.completedBytes >= downloading[index - 1]!.aggregate.completedBytes), "download progress went backwards")
    const intermediatePercentages = downloading
      .map((event) => percentage(event.aggregate))
      .filter((value) => value > 0 && value < 100)
    assert(intermediatePercentages.length > 0, "real download did not expose an intermediate percentage")

    const firstReady = firstEvents.at(-1)
    assert(firstReady?._tag === "Ready", "missing Ready event")
    const filesBeforeDelete = yield* filesUnder(store.cacheRoot)
    const manifestsBeforeDelete = (yield* fs.readDirectory(store.installationRoot)).filter((name) => name.endsWith(".json"))
    assert(manifestsBeforeDelete.length === 1, "download did not publish exactly one installation manifest")

    const cachedEvents = Chunk.toReadonlyArray(yield* Stream.runCollect(download.download(artifact)))
    assert(!cachedEvents.some((event) => event._tag === "Downloading"), "cache hit unexpectedly downloaded bytes")
    assert(cachedEvents[0]?._tag === "CheckingSpace" && cachedEvents[0].requiredBytes === 0, "cache hit required additional download space")
    assert(cachedEvents.at(-1)?._tag === "Ready", "cache hit did not end with Ready")

    const source = yield* makeHuggingFaceCacheSource({ store, label: Option.none() })
    const discovered = Chunk.toReadonlyArray(yield* Stream.runCollect(source.discover({ refresh: "full" })))
    const discoveredSet = discovered.find((event) => event._tag === "FileSet")
    assert(discoveredSet?._tag === "FileSet", "cache source did not discover the installation")
    const resolvedSource = yield* source.resolve(discoveredSet.set.id)
    assert(resolvedSource.set.entries.length === 1, "cache source resolved the wrong number of files")
    assert(resolvedSource.set.origin.pipe(Option.exists((origin) => origin.revision.pipe(Option.exists((value) => String(value) === String(artifact.commit))))), "cache source lost the pinned Hub commit")

    const gguf = yield* makeGgufFormat()
    const registry = yield* makeModelFileRegistry({
      sources: [ModelFileSourceRegistration.Deletable({ source })],
      formats: [gguf],
    })
    const snapshot = yield* registry.inspect("full")
    assert(snapshot.issues.length === 0, `registry inspection reported ${snapshot.issues.length} issue(s)`)
    assert(snapshot.records.length === 1, "registry did not recognize exactly one GGUF model")
    assert(snapshot.records[0]!.id === firstReady.modelFileId, "download and registry disagreed on model identity")
    const resolvedModel = yield* registry.resolve(firstReady.modelFileId)
    assert(yield* fs.exists(resolvedModel.primaryPath), "registry primary model path does not exist")
    assert(resolvedModel.record.sizeBytes === artifact.totalBytes, "registry and Hub disagreed on model size")

    yield* registry.remove(firstReady.modelFileId)
    const afterDelete = Chunk.toReadonlyArray(yield* Stream.runCollect(source.discover({ refresh: "full" })))
    assert(!afterDelete.some((event) => event._tag === "FileSet"), "deleted installation remained discoverable")
    assert(!(yield* fs.exists(resolvedModel.primaryPath)), "unreferenced cached blob remained after deletion")
    const manifestsAfterDelete = (yield* fs.readDirectory(store.installationRoot)).filter((name) => name.endsWith(".json"))
    assert(manifestsAfterDelete.length === 0, "installation manifest remained after deletion")

    return {
      fixture: { repository, requestedRevision: revision, commit: artifact.commit, path: filePath, bytes: artifact.totalBytes },
      search: { results: searchResults.length, foundFixture: true },
      storage: { availableBytes },
      firstDownload: {
        events: firstEvents.length,
        downloadingEvents: downloading.length,
        intermediatePercentages,
        tags: firstTags,
      },
      cacheHit: { tags: cachedEvents.map((event) => event._tag), requiredBytes: 0 },
      cache: { filesBeforeDelete: filesBeforeDelete.map((file) => path.relative(store.cacheRoot, file)), manifestsBeforeDelete: manifestsBeforeDelete.length },
      registry: { records: snapshot.records.length, modelFileId: firstReady.modelFileId, displayName: snapshot.records[0]!.displayName },
      deletion: { discoveredFileSets: 0, manifests: manifestsAfterDelete.length, blobRemoved: true },
    }
  })

  return yield* verification.pipe(Effect.provide(services))
})).pipe(
  Effect.tap((result) => Effect.sync(() => console.log(JSON.stringify({ verifiedAt: new Date().toISOString(), ...result }, null, 2)))),
  Effect.provide(BunContext.layer),
)

BunRuntime.runMain(program)
