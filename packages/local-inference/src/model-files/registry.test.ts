import * as BunContext from "@effect/platform-bun/BunContext"
import * as FileSystem from "@effect/platform/FileSystem"
import * as Path from "@effect/platform/Path"
import { Effect, Option, Ref, Stream } from "effect"
import { describe, expect, it } from "vitest"
import {
  ModelArtifactKey,
  ModelFileFormatId,
  ModelFileSourceId,
  ModelFileSourceKind,
  SourceFileKey,
  SourceFileSetId,
} from "./identity"
import { makeModelFileRegistry } from "./registry"
import {
  ModelFileSourceRegistration,
  SourceDiscoveryEvent,
  SourceFileSetNotFound,
  type ModelFileFormat,
  type ModelFileMetadata,
  type ModelFileSource,
  type SourceFileSet,
} from "./types"

const emptyMetadata = (): ModelFileMetadata => ({
  name: Option.none(),
  architecture: Option.none(),
  ggufFileType: Option.none(),
  quantization: Option.none(),
  trainedContextTokens: Option.none(),
  parameterCount: Option.none(),
  embeddingLength: Option.none(),
  blockCount: Option.none(),
  attentionHeadCount: Option.none(),
  vocabularySize: Option.none(),
  feedForwardLength: Option.none(),
  expertCount: Option.none(),
  expertUsedCount: Option.none(),
  tokenizerModel: Option.none(),
  tokenizerPre: Option.none(),
  chatTemplate: Option.none(),
  baseModelNames: [],
  baseModelRepositories: [],
  inputModalities: Option.none(),
  outputModalities: Option.none(),
})

describe("ModelFileRegistry", () => {
  it("separates cached, changed, and full refresh and detects a resolved file change", async () => {
    const result = await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const fs = yield* FileSystem.FileSystem
      const path = yield* Path.Path
      const root = yield* fs.makeTempDirectoryScoped()
      const filePath = path.join(root, "model.gguf")
      yield* fs.writeFileString(filePath, "model-v1")
      const initialInfo = yield* fs.stat(filePath)

      const sourceId = ModelFileSourceId.make("registry-source")
      const setId = SourceFileSetId.make("registry-set")
      const fileKey = SourceFileKey.make("registry-file")
      const artifactKey = ModelArtifactKey.make("registry-artifact")
      let discoveryCount = 0
      let inspectionCount = 0

      const set: SourceFileSet = {
        id: setId,
        artifactKey: Option.some(artifactKey),
        sourceId,
        entries: [{
          key: fileKey,
          path: filePath,
          relativePath: "model.gguf",
          sizeBytes: Number(initialInfo.size),
          modifiedAtMillis: Option.map(initialInfo.mtime, (time) => time.getTime()),
          sha256: Option.none(),
          declaredRole: Option.some("primary"),
          shardIndex: Option.none(),
        }],
        relationships: [],
        origin: Option.none(),
      }

      const source: ModelFileSource = {
        id: sourceId,
        kind: ModelFileSourceKind.make("test"),
        label: "Registry test source",
        ownership: "external",
        discover: () => Stream.fromEffect(Effect.sync(() => {
          discoveryCount += 1
          return SourceDiscoveryEvent.FileSet({ set })
        })),
        resolve: (requested) => requested === setId
          ? Effect.succeed({ set })
          : Effect.fail(new SourceFileSetNotFound({ id: requested })),
      }

      const format: ModelFileFormat = {
        id: ModelFileFormatId.make("test-format"),
        recognize: () => Effect.succeed(true),
        inspect: () => Effect.sync(() => {
          inspectionCount += 1
          return [{
            key: artifactKey,
            displayName: "Registry model",
            parts: [{ entry: set.entries[0]!, role: "primary" }],
            metadata: emptyMetadata(),
            warnings: [],
          }]
        }),
      }

      const registry = yield* makeModelFileRegistry({
        sources: [ModelFileSourceRegistration.ReadOnly({ source })],
        formats: [format],
        initialIndex: Option.none(),
      })
      const changes = yield* Ref.make(0)
      yield* registry.changes.pipe(
        Stream.runForEach(() => Ref.update(changes, (current) => current + 1)),
        Effect.forkScoped,
      )
      yield* Effect.yieldNow()
      const initial = yield* registry.inspect("changed")
      const persistedIndex = yield* registry.artifactIndex
      const hydratedRegistry = yield* makeModelFileRegistry({
        sources: [ModelFileSourceRegistration.ReadOnly({ source })],
        formats: [format],
        initialIndex: Option.some(persistedIndex),
      })
      const hydrated = yield* hydratedRegistry.inspect("cached")
      const afterHydration = { discoveryCount, inspectionCount }
      yield* registry.inspect("cached")
      const afterCached = { discoveryCount, inspectionCount }
      yield* registry.inspect("changed")
      yield* Effect.yieldNow()
      const afterChanged = { discoveryCount, inspectionCount }
      yield* registry.inspect("full")
      const afterFull = { discoveryCount, inspectionCount }

      const modelId = initial.records[0]!.id
      const beforeChange = yield* registry.resolve(modelId)
      yield* fs.writeFileString(filePath, "model-v2-is-larger")
      const afterChange = yield* Effect.either(registry.resolve(modelId))

      return {
        afterCached,
        afterHydration,
        hydratedRecords: hydrated.records.length,
        afterChanged,
        afterFull,
        primaryPath: beforeChange.primaryPath,
        afterChange,
        changes: yield* Ref.get(changes),
      }
    }).pipe(Effect.provide(BunContext.layer))))

    expect(result.afterCached).toEqual({ discoveryCount: 1, inspectionCount: 1 })
    expect(result.afterHydration).toEqual({ discoveryCount: 1, inspectionCount: 1 })
    expect(result.hydratedRecords).toBe(1)
    expect(result.afterChanged).toEqual({ discoveryCount: 2, inspectionCount: 1 })
    expect(result.afterFull).toEqual({ discoveryCount: 3, inspectionCount: 2 })
    expect(result.primaryPath.endsWith("model.gguf")).toBe(true)
    expect(result.afterChange._tag).toBe("Left")
    if (result.afterChange._tag === "Left") {
      expect(result.afterChange.left.reason).toBe("changed")
    }
    expect(result.changes).toBe(1)
  })
})
