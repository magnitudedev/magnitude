import { describe, expect, it } from "vitest"
import { Schema } from "effect"
import {
  CatalogIntelligenceSchema,
  ModelParameterizationSchema,
  ModelReleaseDateSchema,
} from "./model-state"

describe("ModelReleaseDateSchema", () => {
  it("accepts real ISO calendar dates", () => {
    expect(Schema.decodeUnknownSync(ModelReleaseDateSchema)("2024-02-29")).toBe("2024-02-29")
  })

  it("rejects malformed and impossible dates", () => {
    for (const value of ["2026-8-13", "2026-02-29", "2026-13-01", "0000-01-01"]) {
      expect(() => Schema.decodeUnknownSync(ModelReleaseDateSchema)(value)).toThrow()
    }
  })
})

describe("ModelParameterizationSchema", () => {
  it("accepts valid dense and mixture-of-experts parameterization", () => {
    expect(Schema.decodeUnknownSync(ModelParameterizationSchema)({
      architecture: "dense",
      totalParameters: 8_000_000_000,
    })).toEqual({ architecture: "dense", totalParameters: 8_000_000_000 })
    expect(Schema.decodeUnknownSync(ModelParameterizationSchema)({
      architecture: "mixtureOfExperts",
      totalParameters: 35_000_000_000,
      activeParameters: 3_000_000_000,
    })).toEqual({
      architecture: "mixtureOfExperts",
      totalParameters: 35_000_000_000,
      activeParameters: 3_000_000_000,
    })
  })

  it("rejects nonpositive counts and active counts at or above the total", () => {
    expect(() => Schema.decodeUnknownSync(ModelParameterizationSchema)({
      architecture: "dense",
      totalParameters: 0,
    })).toThrow()
    expect(() => Schema.decodeUnknownSync(ModelParameterizationSchema)({
      architecture: "mixtureOfExperts",
      totalParameters: 3_000_000_000,
      activeParameters: 3_000_000_000,
    })).toThrow()
  })
})

describe("CatalogIntelligenceSchema", () => {
  it("preserves direct and estimated provenance as distinct variants", () => {
    const direct = {
      score: 20.4,
      provenance: {
        kind: "artificialAnalysisIntelligenceIndex",
        methodologyVersion: "4.1.1",
        asOfDate: "2026-08-26",
        url: "https://artificialanalysis.ai/models/qwen3-5-4b",
      },
    }
    const estimate = {
      score: 7.3,
      provenance: {
        kind: "estimate",
        target: "artificialAnalysisIntelligenceIndex",
        methodologyVersion: "4.1.1",
        asOfDate: "2026-08-26",
        confidence: "moderate",
        methodology: "Compared with the exact parent model.",
        evidenceUrls: ["https://example.com/evidence"],
      },
    }
    expect(Schema.decodeUnknownSync(CatalogIntelligenceSchema)(direct)).toEqual(direct)
    expect(Schema.decodeUnknownSync(CatalogIntelligenceSchema)(estimate)).toEqual(estimate)
  })

  it("rejects malformed dates, non-HTTPS evidence, and empty estimate evidence", () => {
    const provenance = {
      kind: "estimate",
      target: "artificialAnalysisIntelligenceIndex",
      methodologyVersion: "4.1.1",
      asOfDate: "2026-08-26",
      confidence: "low",
      methodology: "Peer comparison.",
      evidenceUrls: ["https://example.com/evidence"],
    }
    for (const invalid of [
      { ...provenance, asOfDate: "2026-02-29" },
      { ...provenance, evidenceUrls: ["http://example.com/evidence"] },
      { ...provenance, evidenceUrls: [] },
    ]) {
      expect(() => Schema.decodeUnknownSync(CatalogIntelligenceSchema)({
        score: 1,
        provenance: invalid,
      })).toThrow()
    }
  })
})
