import * as BunContext from "@effect/platform-bun/BunContext"
import * as FileSystem from "@effect/platform/FileSystem"
import * as FetchHttpClient from "@effect/platform/FetchHttpClient"
import * as Path from "@effect/platform/Path"
import * as Scope from "effect/Scope"
import { afterAll, beforeAll, describe, expect, it } from "vitest"
import { Effect, Exit, Option, Redacted, Ref } from "effect"
import {
  ModelFileFormatId,
  ModelFileId,
  ModelFilePartId,
  ModelFileSourceId,
  SourceFileKey,
  type ModelFileRecord,
  type ModelFileRegistryApi,
} from "../model-files"
import {
  BatchSize,
  ContextSize,
  FlashAttentionSelection,
  GpuLayerSelection,
  LlamaBinaryFingerprint,
  LlamaModelRequest,
  LlamaServedModelId,
  MicroBatchSize,
  OutputLimit,
  makeLlamaExecutionProfile,
  makeLlamaInstanceRegistry,
  type LlamaCli,
} from "."

let loaded = false
let server: ReturnType<typeof Bun.serve>

beforeAll(() => {
  server = Bun.serve({
    port: 0,
    fetch: (request) => {
      const url = new URL(request.url)
      if (url.pathname === "/health") return Response.json({ status: "ok" })
      if (url.pathname === "/models") return Response.json({ data: [{ id: "managed-model", status: loaded ? "loaded" : "unloaded" }] })
      if (url.pathname === "/models/load") { loaded = true; return Response.json({}) }
      if (url.pathname === "/models/unload") { loaded = false; return Response.json({}) }
      if (url.pathname === "/props") return Response.json({})
      return new Response(null, { status: 404 })
    },
  })
})

afterAll(() => server.stop(true))

const emptyMetadata: ModelFileRecord["metadata"] = {
  name: Option.none(), architecture: Option.none(), ggufFileType: Option.none(), quantization: Option.none(),
  trainedContextTokens: Option.none(), parameterCount: Option.none(), embeddingLength: Option.none(), blockCount: Option.none(),
  attentionHeadCount: Option.none(), vocabularySize: Option.none(), feedForwardLength: Option.none(), expertCount: Option.none(),
  expertUsedCount: Option.none(), tokenizerModel: Option.none(), tokenizerPre: Option.none(), chatTemplate: Option.none(),
  baseModelNames: [], baseModelRepositories: [], inputModalities: Option.none(), outputModalities: Option.none(),
}

describe("llama instance registry", () => {
  it("protects a managed model with a scoped lease and releases process resources", async () => {
    loaded = false
    const closed = await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const processClosed = yield* Ref.make(false)
      const fs = yield* FileSystem.FileSystem
      const path = yield* Path.Path
      const temporary = yield* fs.makeTempDirectoryScoped()
      const modelFileId = ModelFileId.make("managed-file")
      const sourceId = ModelFileSourceId.make("source")
      const record: ModelFileRecord = {
        id: modelFileId,
        sourceId,
        displayName: "Managed model",
        format: ModelFileFormatId.make("gguf"),
        sizeBytes: 4,
        files: [{ id: ModelFilePartId.make("part"), role: "primary", sizeBytes: 4, sha256: Option.none() }],
        metadata: emptyMetadata,
        ownership: "magnitude",
        operations: { delete: true },
        warnings: [],
      }
      const modelFiles: ModelFileRegistryApi = {
        inspect: () => Effect.succeed({ records: [record], issues: [], capturedAt: new Date() }),
        get: () => Effect.succeed(record),
        resolve: () => Effect.succeed({
          record,
          primaryPath: "/models/model.gguf",
          shardPaths: [],
          projectorPath: Option.none(),
          auxiliaryPaths: [],
          version: [{ key: SourceFileKey.make("model"), sizeBytes: 4, modifiedAtMillis: Option.none() }],
        }),
        remove: () => Effect.void,
        index: Effect.succeed({ capturedAt: new Date(), sets: [], issues: [] }),
      }
      const cli: LlamaCli = {
        binary: {
          executable: "/test/llama-server",
          distributionDirectory: "/test",
          build: { versionOutput: "test", buildNumber: Option.none(), commit: Option.none() },
          source: "configured",
          fingerprint: LlamaBinaryFingerprint.make("test-fingerprint"),
        },
        version: Effect.succeed({ versionOutput: "test", buildNumber: Option.none(), commit: Option.none() }),
        capabilities: Effect.succeed({ fitPrint: false, listDevices: false, apiKey: true, presets: true, noModelsAutoload: true, modelsMax: true, modelSleepIdleSeconds: true, helpOutput: "" }),
        listDevices: Effect.dieMessage("unused"),
        assessFit: () => Effect.dieMessage("unused"),
        startRouter: () => Effect.gen(function* () {
          yield* Effect.addFinalizer(() => Ref.set(processClosed, true))
          return {
            origin: new URL(`http://127.0.0.1:${server.port}`),
            exited: Effect.sleep("1 hour").pipe(Effect.as(0)),
            sanitizedOutput: Effect.succeed(""),
          }
        }),
      }
      const profile = yield* makeLlamaExecutionProfile({
        contextSize: ContextSize.ModelDefault(), outputLimit: OutputLimit.RuntimeDefault(), parallelSlots: 1,
        gpuLayers: GpuLayerSelection.Fit(), splitMode: "layer", tensorSplit: Option.none(), kvCache: { key: "f16", value: "f16" },
        flashAttention: FlashAttentionSelection.RuntimeDefault(), batchSize: BatchSize.RuntimeDefault(), microBatchSize: MicroBatchSize.RuntimeDefault(), mmap: true, mlock: false,
      })
      const port = yield* Option.match(Option.fromNullable(server.port), {
        onNone: () => Effect.dieMessage("test server did not expose a port"),
        onSome: Effect.succeed,
      })
      const registry = yield* makeLlamaInstanceRegistry({
        cli: Option.some(cli), modelFiles, presetPath: path.join(temporary, "presets.ini"), host: "127.0.0.1", port,
        apiKey: Redacted.make("managed-secret"), modelsMax: 1,
        external: [],
      })
      const leaseScope = yield* Scope.make()
      const servedModelId = LlamaServedModelId.make("managed-model")
      yield* registry.acquire(LlamaModelRequest.Managed({ request: { modelFileId, servedModelId, profile } })).pipe(
        Effect.provideService(Scope.Scope, leaseScope),
      )
      const guarded = yield* registry.stopManaged.pipe(Effect.either)
      yield* Scope.close(leaseScope, Exit.void)
      yield* registry.stopManaged
      return { guarded: guarded._tag, processClosed: yield* Ref.get(processClosed) }
    }).pipe(Effect.provide([BunContext.layer, FetchHttpClient.layer]))))

    expect(closed.guarded).toBe("Left")
    expect(closed.processClosed).toBe(true)
  })
})
