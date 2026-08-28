import { describe, expect, it } from "vitest"
import { assertSpeculativeEvidence } from "../src/run"

const experiment = {
  variants: [{
    id: "mtp",
    engine: { kind: "llama.cpp", speculativeDecoding: { kind: "mtp" } },
  }],
} as never

const comparison = (timings: unknown) => ({
  results: [{
    target: { id: "mtp" },
    trials: [{
      requests: [{ requestId: "request", terminal: { timings } }],
    }],
  }],
}) as never

describe("MTP run evidence", () => {
  it("requires consistent, nonzero native draft counters", () => {
    expect(() => assertSpeculativeEvidence(experiment, comparison({ draftTokens: 2, acceptedDraftTokens: 1 }))).not.toThrow()
    expect(() => assertSpeculativeEvidence(experiment, comparison({}))).toThrow("did not return native mtp draft counters")
    expect(() => assertSpeculativeEvidence(experiment, comparison({ draftTokens: 1, acceptedDraftTokens: 2 }))).toThrow("inconsistent mtp draft counters")
    expect(() => assertSpeculativeEvidence(experiment, comparison({ draftTokens: 0, acceptedDraftTokens: 0 }))).toThrow("did not demonstrate active mtp drafting")
  })

  it("requires the exact oMLX backend and rejects speculative evidence on a baseline", () => {
    const omlxMtp = {
      variants: [{ id: "mtp", engine: { kind: "omlx", speculativeDecoding: { kind: "mtp" } } }],
    } as never
    expect(() => assertSpeculativeEvidence(
      omlxMtp,
      comparison({ draftTokens: 2, acceptedDraftTokens: 1, speculativeBackend: "dflash" }),
    )).toThrow("expected mtp but reported dflash")

    const baseline = {
      variants: [{ id: "mtp", engine: { kind: "omlx", speculativeDecoding: { kind: "none" } } }],
    } as never
    expect(() => assertSpeculativeEvidence(
      baseline,
      comparison({ draftTokens: 2, acceptedDraftTokens: 1, speculativeBackend: "mtp" }),
    )).toThrow("baseline request")
  })
})
