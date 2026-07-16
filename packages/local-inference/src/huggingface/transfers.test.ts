import { createHash } from "node:crypto"
import * as BunContext from "@effect/platform-bun/BunContext"
import * as FileSystem from "@effect/platform/FileSystem"
import * as FetchHttpClient from "@effect/platform/FetchHttpClient"
import * as Path from "@effect/platform/Path"
import { afterAll, beforeAll, describe, expect, it } from "vitest"
import { Effect, Option, Ref, Stream } from "effect"
import {
  ModelFileId,
  ModelFileSourceId,
  ModelFileSourceKind,
  Sha256Digest,
  SourceFileSetNotFound,
  type ModelFilePublication,
  type WritableModelFileSource,
} from "../model-files"
import { HuggingFaceCommitId, HuggingFaceRepositoryId, HuggingFaceRevision } from "./identity"
import { makeModelTransferRegistry, type HuggingFaceArtifactRequest } from "./transfers"
import type { HuggingFaceHubClientApi } from "./hub-client"

const contents = new TextEncoder().encode("verified-model")
const digest = Sha256Digest.make(createHash("sha256").update(contents).digest("hex"))
let server: ReturnType<typeof Bun.serve>

beforeAll(() => {
  server = Bun.serve({
    port: 0,
    fetch: () => new Response(contents, { status: 200 }),
  })
})

afterAll(() => server.stop(true))

describe("model transfer registry", () => {
  it("pins Hub facts, streams, verifies, and publishes the exact artifact", async () => {
    const result = await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const fs = yield* FileSystem.FileSystem
      const path = yield* Path.Path
      const temporary = yield* fs.makeTempDirectoryScoped()
      const published = yield* Ref.make<Option.Option<ModelFilePublication>>(Option.none())
      const repository = HuggingFaceRepositoryId.make("owner/model")
      const commit = HuggingFaceCommitId.make("a".repeat(40))
      const hub: HuggingFaceHubClientApi = {
        searchModels: () => Effect.succeed([]),
        resolveRevision: (_repository, requested) => Effect.succeed({ repository, requested, commit }),
        listFiles: () => Effect.succeed([{
          path: "model.gguf",
          type: "file",
          sizeBytes: Option.some(contents.byteLength),
          oid: Option.none(),
          lfs: Option.some({ sizeBytes: contents.byteLength, sha256: digest }),
        }]),
        downloadUrl: () => new URL(`http://127.0.0.1:${server.port}/model.gguf`),
      }
      const destination: WritableModelFileSource = {
        id: ModelFileSourceId.make("transfer-destination"),
        kind: ModelFileSourceKind.make("test-destination"),
        label: "Test destination",
        ownership: "magnitude",
        discover: () => Stream.empty,
        resolve: (id) => Effect.fail(new SourceFileSetNotFound({ id })),
        publish: (publication) => Ref.set(published, Option.some(publication)).pipe(
          Effect.as(ModelFileId.make("published-model")),
        ),
      }
      const registry = yield* makeModelTransferRegistry({
        hub,
        destination,
        capacity: { availableBytes: () => Effect.succeed(1_000_000) },
        stagingRoot: path.join(temporary, "staging"),
        stateRoot: Option.none(),
        reserveBytes: 0,
      })
      const request: HuggingFaceArtifactRequest = {
        repository,
        revision: HuggingFaceRevision.make("main"),
        files: [{ path: "model.gguf", role: "primary", shardIndex: Option.none() }],
        relationships: [],
      }
      const plan = yield* registry.plan(request)
      const transferId = yield* registry.start(plan)
      const terminal = yield* registry.observe(transferId).pipe(
        Stream.filter((snapshot) => snapshot.status._tag === "Ready" || snapshot.status._tag === "Failed"),
        Stream.runHead,
        Effect.timeout("5 seconds"),
      )
      return { plan, terminal, publication: yield* Ref.get(published) }
    }).pipe(Effect.provide([BunContext.layer, FetchHttpClient.layer]))))

    expect(result.plan.commit).toBe("a".repeat(40))
    expect(result.plan.files[0]?.sha256).toBe(digest)
    expect(Option.getOrNull(result.terminal)?.status._tag).toBe("Ready")
    const publication = Option.getOrNull(result.publication)
    expect(publication?.files[0]?.sizeBytes).toBe(contents.byteLength)
    expect(publication?.files[0]?.sha256).toBe(digest)
    expect(publication?.origin.pipe(Option.getOrNull)?.revision).toBe("a".repeat(40))
  })
})
