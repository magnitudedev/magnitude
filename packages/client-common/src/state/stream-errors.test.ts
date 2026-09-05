import { RpcClientError } from "@effect/rpc"
import { Cause } from "effect"
import { describe, expect, it } from "vitest"
import { ServiceStartFailed, ServiceUnavailable } from "@magnitudedev/sdk"
import { classifyStartupError, classifyStreamError } from "./stream-errors"

const healthUnavailable = new ServiceUnavailable({
  origin: "http://127.0.0.1:10100",
  message: "Magnitude service at http://127.0.0.1:10100 is unavailable: connection refused",
})

describe("stream error classification", () => {
  it("preserves a structured ACN availability error nested in transport recovery", () => {
    const cause = new ServiceStartFailed({
      message: "artifact unavailable",
    })
    const failure = new RpcClientError.RpcClientError({
      reason: "Unknown",
      message: "Service start failed",
      cause,
    })
    const classified = classifyStreamError(Cause.fail(failure))
    expect(classified.isAcnAvailabilityError).toBe(true)
    expect(classified.invariantViolation).toBe(false)
    expect(classified.message).toContain("artifact unavailable")
  })

  it("presents health diagnostics in startup UI without internal error tags", () => {
    const classified = classifyStartupError(healthUnavailable)

    expect(classified.isAcnAvailabilityError).toBe(true)
    expect(classified.invariantViolation).toBe(false)
    expect(classified.message).toContain("http://127.0.0.1:10100")
    expect(classified.message).toContain("connection refused")
    expect(classified.message).not.toContain("AcnHealthUnavailable")
    expect(classified.message).not.toContain("AcnEnsuranceFailed")
  })

  it("preserves health diagnostics nested in transport recovery without internal error tags", () => {
    const failure = new RpcClientError.RpcClientError({
      reason: "Unknown",
      message: "service unavailable",
      cause: healthUnavailable,
    })
    const classified = classifyStreamError(Cause.fail(failure))

    expect(classified.isAcnAvailabilityError).toBe(true)
    expect(classified.invariantViolation).toBe(false)
    expect(classified.message).toContain("connection refused")
    expect(classified.message).not.toContain("AcnHealthUnavailable")
    expect(classified.message).not.toContain("AcnEnsuranceFailed")
  })
})
