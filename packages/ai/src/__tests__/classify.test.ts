import { describe, it, expect } from "vitest"
import {
  formatStreamStartFailureMessage,
  snapshotModelAttemptFailure,
  streamStartFailureFromRejectedResponse,
} from "../errors/classify"
import { rejectedHttpResponse, type ProviderCall } from "../errors/failure"

const call: ProviderCall = {
  provider: "test-provider",
  model: "test-model",
  method: "POST",
  url: "https://example.test/chat/completions",
}

function rejected(status: number, body: string, headers: HeadersInit = {}) {
  return rejectedHttpResponse(status, new Headers(headers), body)
}

describe("streamStartFailureFromRejectedResponse", () => {
  it("classifies a valid provider error envelope as StreamStartProviderRejection", () => {
    const failure = streamStartFailureFromRejectedResponse(
      call,
      rejected(429, JSON.stringify({
        error: {
          message: "rate limited",
          type: "rate_limit_error",
          code: "rate_limit",
          param: "model",
          details: { retry: "later" },
        },
      }), { "retry-after": "5" }),
    )

    expect(failure._tag).toBe("StreamStartProviderRejection")
    if (failure._tag !== "StreamStartProviderRejection") return
    expect(failure.response.status).toBe(429)
    expect(failure.response.retryAfterMs).toBe(5000)
    expect(failure.rejection).toEqual({
      _tag: "RateLimited",
      message: "rate limited",
      retryPolicy: {
        retry: true,
        retryAfterMs: { value: 5000 },
      },
    })
  })

  it("classifies a rejected response without a valid error envelope as provider correctness violation", () => {
    const failure = streamStartFailureFromRejectedResponse(
      call,
      rejected(500, "<html>not an error envelope</html>", { "content-type": "text/html" }),
    )

    expect(failure._tag).toBe("StreamStartProviderCorrectnessViolation")
    if (failure._tag !== "StreamStartProviderCorrectnessViolation") return
    expect(failure.violation._tag).toBe("InvalidErrorEnvelope")
    expect(failure.violation.status).toBe(500)
    expect(failure.violation.body.text).toContain("not an error envelope")
  })

  it("requires a non-empty error.message in the envelope", () => {
    const failure = streamStartFailureFromRejectedResponse(
      call,
      rejected(400, JSON.stringify({ error: { type: "invalid_request_error", code: "bad_request" } })),
    )

    expect(failure._tag).toBe("StreamStartProviderCorrectnessViolation")
    if (failure._tag !== "StreamStartProviderCorrectnessViolation") return
    expect(failure.violation._tag).toBe("InvalidErrorEnvelope")
  })

  it("formats and snapshots stream-start provider errors", () => {
    const failure = streamStartFailureFromRejectedResponse(
      call,
      rejected(
        400,
        JSON.stringify({ error: { message: "prompt is too long", type: "invalid_request_error", code: "context_length_exceeded" } }),
      ),
    )

    const message = formatStreamStartFailureMessage(failure)
    expect(message).toContain("Model provider rejected the request")
    expect(message).toContain("ContextLimitExceeded")
    expect(message).toContain("prompt is too long")

    const snapshot = snapshotModelAttemptFailure(failure)
    expect(snapshot.phase).toBe("stream_start")
    expect(snapshot.tag).toBe("StreamStartProviderRejection")
    expect(snapshot.detailTag).toBe("ContextLimitExceeded")
    expect(snapshot.responseStatus).toBe(400)
    expect(snapshot.providerMessage).toBe("prompt is too long")
  })
})
