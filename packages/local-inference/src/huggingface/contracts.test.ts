import { Effect, Option, Schema } from "effect"
import { describe, expect, it } from "vitest"
import { ModelFileId, Sha256Digest } from "../model-files"
import { makeHuggingFaceArtifactId } from "./artifact-identity"
import {
  DownloadProgress,
  DownloadReady,
  HuggingFaceArtifact,
  HuggingFaceLfsContent,
} from "./contracts"
import { HuggingFaceAccessDeniedError, HuggingFaceDownloadError } from "./errors"
import { HuggingFaceArtifactId, HuggingFaceCommitId, HuggingFaceFilePath, HuggingFaceRepositoryId, HuggingFaceRevision } from "./identity"

describe("Hugging Face schemas", () => {
  it("round-trips resolved artifacts and progress events", async () => {
    const file = {
      path: HuggingFaceFilePath.make("model.gguf"),
      role: "primary" as const,
      shardIndex: Option.none<number>(),
      sizeBytes: 8,
      content: new HuggingFaceLfsContent({ sha256: Sha256Digest.make("b".repeat(64)) }),
    }
    const identity = { repository: HuggingFaceRepositoryId.make("owner/model"), commit: HuggingFaceCommitId.make("a".repeat(40)), files: [file], relationships: [] }
    const artifact = { id: makeHuggingFaceArtifactId(identity), requestedRevision: HuggingFaceRevision.make("main"), ...identity, totalBytes: 8 }
    const encodedArtifact = await Effect.runPromise(Schema.encode(HuggingFaceArtifact)(artifact))
    const decodedArtifact = await Effect.runPromise(Schema.decodeUnknown(HuggingFaceArtifact)(encodedArtifact))
    expect(decodedArtifact).toEqual(artifact)

    const ready = new DownloadReady({ artifactId: HuggingFaceArtifactId.make(artifact.id), modelFileId: ModelFileId.make("mf_fixture"), aggregate: { completedBytes: 8, totalBytes: 8 } })
    const encodedProgress = await Effect.runPromise(Schema.encode(DownloadProgress)(ready))
    expect(encodedProgress).toEqual({ _tag: "Ready", artifactId: artifact.id, modelFileId: "mf_fixture", aggregate: { completedBytes: 8, totalBytes: 8 } })

    const encodedError = await Effect.runPromise(Schema.encode(HuggingFaceDownloadError)(new HuggingFaceAccessDeniedError({ operation: "download", repository: Option.some(identity.repository) })))
    expect(encodedError).toEqual({ _tag: "HuggingFaceAccessDeniedError", operation: "download", repository: "owner/model" })
  })
})
