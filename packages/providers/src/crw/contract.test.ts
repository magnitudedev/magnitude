import { Schema } from "effect"
import { describe, expect, it } from "vitest"
import {
  CrwSearchRequestSchema,
  CrwSearchResponseSchema,
} from "./contract"

describe("fastCRW search contract", () => {
  it("round-trips the documented JSON request shape", () => {
    const wire = {
      query: "latest developments in LLMs",
      limit: 10,
    } as const

    const decoded = Schema.decodeUnknownSync(CrwSearchRequestSchema)(wire)
    expect(Schema.encodeSync(CrwSearchRequestSchema)(decoded)).toEqual(wire)
  })

  it("rejects an empty query", () => {
    expect(() => Schema.decodeUnknownSync(CrwSearchRequestSchema)({ query: "" })).toThrow()
  })

  // Both payload shapes are live: the hosted endpoint answers with a bare array,
  // a self-hosted engine nests the same rows under `results`.
  it("decodes the hosted response shape", () => {
    const wire = {
      success: true,
      data: [{
        url: "https://magnitude.dev",
        title: "Magnitude",
        description: "An open source coding agent.",
        snippet: "An open source coding agent.",
        position: 1,
        score: 0.92,
        category: "general",
      }],
    } as const

    const decoded = Schema.decodeUnknownSync(CrwSearchResponseSchema)(wire)
    expect(decoded.data._tag).toBe("Some")
  })

  it("decodes the self-hosted response shape", () => {
    const wire = {
      success: true,
      data: {
        results: [{
          url: "https://magnitude.dev",
          title: null,
          description: null,
          snippet: "An open source coding agent.",
        }],
      },
    } as const

    const decoded = Schema.decodeUnknownSync(CrwSearchResponseSchema)(wire)
    expect(decoded.data._tag).toBe("Some")
  })

  it("decodes a failure envelope", () => {
    const decoded = Schema.decodeUnknownSync(CrwSearchResponseSchema)({
      success: false,
      error: "Invalid or missing API key",
    })
    expect(decoded.success._tag).toBe("Some")
    expect(decoded.error._tag).toBe("Some")
  })

  it("rejects a non-object payload", () => {
    expect(() => Schema.decodeUnknownSync(CrwSearchResponseSchema)({
      success: true,
      data: "not results",
    })).toThrow()
  })
})
