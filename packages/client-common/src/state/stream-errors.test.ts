import { RpcClientError } from "@effect/rpc"
import { Cause } from "effect"
import { describe, expect, it } from "vitest"
import { DownloadFailed } from "@magnitudedev/sdk"
import { classifyStreamError } from "./stream-errors"

describe("stream error classification", () => {
  it("preserves a structured ACN availability error nested in transport recovery", () => {
    const cause = new DownloadFailed({
      url: "https://example.invalid/acn",
      status: 503,
      reason: "artifact unavailable",
    })
    const failure = new RpcClientError.RpcClientError({
      reason: "Unknown",
      message: "ACN unavailable: DownloadFailed",
      cause,
    })
    const classified = classifyStreamError(Cause.fail(failure))
    expect(classified.isAcnAvailabilityError).toBe(true)
    expect(classified.invariantViolation).toBe(false)
    expect(classified.message).toContain("artifact unavailable")
  })
})
