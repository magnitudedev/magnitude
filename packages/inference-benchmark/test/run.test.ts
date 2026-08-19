import { describe, expect, it } from "vitest"
import { assertSpeculativeEvidence, targetFor } from "../src/run"

const experiment = (engine: unknown, temperature = 0) => ({
  requestPolicy: { temperature },
  variants: [{ id: "speculative", engine }],
}) as never

const mtpExperiment = experiment({ kind: "llama.cpp", speculativeDecoding: { kind: "mtp" } })
const dflashExperiment = (temperature: number) =>
  experiment({ kind: "icn", speculativeDecoding: { kind: "dflash" } }, temperature)

const comparison = (timings: unknown) => ({
  results: [{
    target: { id: "speculative" },
    trials: [{
      requests: [{ requestId: "request", terminal: { timings } }],
    }],
  }],
}) as never

describe("speculative run evidence", () => {
  it("preserves the experiment request timeout on managed ICN targets", () => {
    const target = targetFor({
      experiment: {
        requestPolicy: { parallelSequences: 1, requestTimeoutMs: 3_600_000 },
        variants: [{
          id: "icn",
          artifact: { modelId: "model" },
          engine: { kind: "icn", speculativeDecoding: { kind: "none" } },
        }],
      },
      artifacts: [{
        variantId: "icn", role: "target", kind: "gguf", repository: "owner/model",
        revision: "revision", quantization: "Q4_K_M", path: "/model.gguf", digest: "a".repeat(64),
      }],
      engines: [{ variantId: "icn", kind: "icn", executable: "/magnitude-icn" }],
      planModel: { id: "model", contextLimit: 65_536 },
    } as never, "icn", 8091, "/run.log")

    expect(target.requestTimeoutMs).toBe(3_600_000)
  })

  it("requires consistent, nonzero native draft counters", () => {
    expect(() => assertSpeculativeEvidence(mtpExperiment, comparison({ draftTokens: 2, acceptedDraftTokens: 1 }))).not.toThrow()
    expect(() => assertSpeculativeEvidence(mtpExperiment, comparison({}))).toThrow("did not return native mtp draft counters")
    expect(() => assertSpeculativeEvidence(mtpExperiment, comparison({ draftTokens: 1, acceptedDraftTokens: 2 }))).toThrow("inconsistent mtp draft counters")
    expect(() => assertSpeculativeEvidence(mtpExperiment, comparison({ draftTokens: 0, acceptedDraftTokens: 0 }))).toThrow("did not demonstrate active mtp drafting")
  })

  it("requires accepted DFlash drafts and distribution-backed drafts under stochastic sampling", () => {
    const distributed = { draftTokens: 3, acceptedDraftTokens: 2, proposalDistributionDraftTokens: 3 }
    expect(() => assertSpeculativeEvidence(dflashExperiment(0.8), comparison(distributed))).not.toThrow()
    expect(() => assertSpeculativeEvidence(dflashExperiment(0.8), comparison({ draftTokens: 3, acceptedDraftTokens: 0 })))
      .toThrow("accepted no DFlash draft tokens")
    expect(() => assertSpeculativeEvidence(dflashExperiment(0.8), comparison({ draftTokens: 3, acceptedDraftTokens: 2 })))
      .toThrow("no proposal-distribution-backed drafts")
    expect(() => assertSpeculativeEvidence(dflashExperiment(0.8), comparison({ draftTokens: 3, acceptedDraftTokens: 2, proposalDistributionDraftTokens: 4 })))
      .toThrow("inconsistent dflash draft counters")
    expect(() => assertSpeculativeEvidence(dflashExperiment(0), comparison({ draftTokens: 3, acceptedDraftTokens: 2 }))).not.toThrow()
  })
})
