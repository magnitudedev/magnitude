import { HubApiError } from "@huggingface/hub"
import { Effect, Option, Stream } from "effect"
import { describe, expect, it } from "vitest"
import { HuggingFaceRepositoryId, HuggingFaceRevision } from "./identity"
import { makeHuggingFaceHubFromUpstream, mapHuggingFaceHubError } from "./hub"
import { HuggingFaceUpstreamFailure, type HuggingFaceUpstreamApi } from "./upstream"

describe("Hugging Face Hub adapter", () => {
  it("pins revisions and resolves exact selected file identities", async () => {
    const upstream: HuggingFaceUpstreamApi = {
      searchModels: () => Stream.empty,
      resolveRevision: () => Effect.succeed("a".repeat(40)),
      pathsInfo: () => Effect.succeed([{ path: "model.gguf", type: "file", size: 4, lfs: { oid: "b".repeat(64), size: 4, pointerSize: 1 } }]),
      downloadToCache: () => Effect.die("unused"),
    }
    const hub = makeHuggingFaceHubFromUpstream(upstream)
    const artifact = await Effect.runPromise(hub.resolveArtifact({
      repository: HuggingFaceRepositoryId.make("owner/model"),
      revision: HuggingFaceRevision.make("main"),
      files: [{ path: "model.gguf", role: "primary", shardIndex: Option.none() }],
      relationships: [],
    }))
    expect(artifact.commit).toBe("a".repeat(40))
    expect(artifact.totalBytes).toBe(4)
    expect(artifact.files[0].content._tag).toBe("LfsSha256")
    expect(artifact.id).toMatch(/^hf_[a-f0-9]{64}$/)
  })

  it("preserves actionable Hub errors across the upstream boundary", () => {
    const error = mapHuggingFaceHubError(
      new HuggingFaceUpstreamFailure({ cause: new HubApiError("https://huggingface.co/owner/model", 403) }),
      { operation: "download", repository: Option.some(HuggingFaceRepositoryId.make("owner/model")), revision: Option.some(HuggingFaceRevision.make("main")), path: Option.none() },
    )
    expect(error._tag).toBe("HuggingFaceAccessDeniedError")
  })
})
