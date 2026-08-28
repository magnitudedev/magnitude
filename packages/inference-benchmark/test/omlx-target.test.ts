import { describe, expect, it } from "vitest"
import { targetFor } from "../src/run"
import type { PreparedExperiment } from "../src/preparation"

function prepared(speculativeDecoding: unknown, withDraft = false): PreparedExperiment {
  const artifact = {
    kind: "mlx", modelId: "model", modelSource: { repository: "owner/model", revision: "base" },
    modelContextLimit: 8192, repository: "owner/target", revision: "target", manifest: "/locks/target.json",
    quantization: { family: "mlx-affine", bits: 4, groupSize: 64 },
  }
  return {
    version: 5, experimentPath: "/experiments/example.ts", preparedAt: new Date(0).toISOString(),
    experiment: {
      id: "example", title: "Example", comparisonProtocol: { kind: "fixed-speculative-policy" }, suite: { kind: "agent-core", profile: "smoke" },
      requestPolicy: { contextTokensPerSequence: 4096, parallelSequences: 1, maxOutputTokens: 32, requestTimeoutMs: 300_000, temperature: 0, topP: 1, seed: 42, enableThinking: false },
      variants: [{ id: "variant", artifact, engine: { kind: "omlx", pythonProject: "/engine", cache: { kind: "disabled" }, memoryGuard: { kind: "off" }, speculativeDecoding } }],
      execution: { variantOrder: "declared", blocks: 1 },
    },
    corpusRoot: "/corpus", corpusDigest: "corpus", planModel: { id: "model", contextLimit: 8192 },
    artifacts: [
      { variantId: "variant", role: "target", kind: "mlx", path: "/snapshots/target", repository: "owner/target", revision: "target", quantization: "4-bit/group-64", digest: "target-digest", manifestPath: "/locks/target.json", manifestDigest: "manifest" },
      ...(withDraft ? [{ variantId: "variant", role: "drafter" as const, kind: "mlx" as const, path: "/snapshots/draft", repository: "owner/draft", revision: "draft", quantization: "bfloat16", digest: "draft-digest", manifestPath: "/locks/draft.json", manifestDigest: "draft-manifest" }] : []),
    ],
    engines: [{
      variantId: "variant", kind: "omlx", pythonProject: "/engine", omlxVersion: "0.6.3", omlxRevision: "omlx",
      mlxVersion: "0.32.0", mlxLmRevision: "mlx-lm", mlxVlmRevision: "mlx-vlm", dflashRevision: "dflash",
      lockDigest: "lock", adapterDigest: "adapter",
    }],
    host: { hostname: "host", platform: "darwin", release: "release", arch: "arm64", cpu: "cpu", logicalCpus: 1, totalMemoryBytes: 1 },
    digest: "prepared",
  } as unknown as PreparedExperiment
}

describe("oMLX target construction", () => {
  it("uses the frozen adapter, private base, alias, capacity, and explicit baseline settings", () => {
    const target = targetFor(prepared({ kind: "none" }), "variant", 9000, "/run/server.log", "/run/private-omlx")
    expect(target.kind).toBe("managed")
    if (target.kind !== "managed") return
    expect(target.engine).toBe("omlx")
    expect(target.servedModel).toBe("model")
    expect(target.speculativeBackend).toBe("none")
    expect(target.args).toEqual(expect.arrayContaining([
      "--frozen", "--no-sync", "--base-path", "/run/private-omlx",
      "--context-capacity", "4096", "--max-concurrent-requests", "1",
      "--cache-policy", "disabled", "--memory-guard", "off", "--speculative-method", "none",
    ]))
    expect(target.readiness.kind).toBe("omlx")
  })

  it("passes only the locked DFlash draft snapshot and requested block size", () => {
    const target = targetFor(prepared({ kind: "dflash", draftArtifact: {}, blockSize: 4 }, true), "variant", 9000, "/run/server.log", "/run/private-omlx")
    expect(target.kind).toBe("managed")
    if (target.kind !== "managed") return
    expect(target.args).toEqual(expect.arrayContaining([
      "--speculative-method", "dflash", "--dflash-draft", "/snapshots/draft", "--dflash-block-size", "4",
    ]))
  })
})
