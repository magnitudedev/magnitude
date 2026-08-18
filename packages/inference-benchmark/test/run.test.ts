import { describe, expect, it } from "vitest"
import { assertMtpEvidence } from "../src/run"

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
    expect(() => assertMtpEvidence(experiment, comparison({ draftTokens: 2, acceptedDraftTokens: 1 }))).not.toThrow()
    expect(() => assertMtpEvidence(experiment, comparison({}))).toThrow("did not return native MTP draft counters")
    expect(() => assertMtpEvidence(experiment, comparison({ draftTokens: 1, acceptedDraftTokens: 2 }))).toThrow("inconsistent MTP draft counters")
    expect(() => assertMtpEvidence(experiment, comparison({ draftTokens: 0, acceptedDraftTokens: 0 }))).toThrow("did not demonstrate active MTP drafting")
  })
})
