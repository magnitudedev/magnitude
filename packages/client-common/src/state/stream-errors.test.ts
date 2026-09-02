import { RpcClientError } from "@effect/rpc"
import { Cause, Schema } from "effect"
import { describe, expect, it } from "vitest"
import {
  AcnHealthRequestFailed,
  AcnHealthResponseInvalid,
  AcnHealthUnavailable,
  DownloadFailed,
} from "@magnitudedev/sdk"
import { AcnOwnerRecordSchema } from "@magnitudedev/acn-protocol/coordination"
import { classifyStartupError, classifyStreamError } from "./stream-errors"

const healthUnavailable = new AcnHealthUnavailable({
  owner: Schema.decodeUnknownSync(AcnOwnerRecordSchema)({
    pid: 42,
    processStartIdentity: "start-1",
    port: 14_000,
  }),
  attempts: [
    new AcnHealthRequestFailed({ message: "connection refused" }),
    new AcnHealthResponseInvalid({ message: "unexpected health service" }),
  ],
})

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

  it("presents health diagnostics in startup UI without internal error tags", () => {
    const classified = classifyStartupError(healthUnavailable)

    expect(classified.isAcnAvailabilityError).toBe(true)
    expect(classified.invariantViolation).toBe(false)
    expect(classified.message).toContain("http://127.0.0.1:14000/health (PID 42)")
    expect(classified.message).toContain("Health check 1: request failed: connection refused")
    expect(classified.message).toContain("Health check 2: response was invalid: unexpected health service")
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
    expect(classified.message).toContain("Health check 1: request failed: connection refused")
    expect(classified.message).toContain("Health check 2: response was invalid: unexpected health service")
    expect(classified.message).not.toContain("AcnHealthUnavailable")
    expect(classified.message).not.toContain("AcnEnsuranceFailed")
  })
})
