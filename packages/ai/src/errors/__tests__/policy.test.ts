import { describe, expect, it } from "vitest"
import { Effect, Option } from "effect"
import {
  AiRetryPolicy,
  AiRetryPolicyLive,
  UpstreamRetryability,
  defaultRetryabilityForFailure,
  AiBillingPolicy,
  AiBillingPolicyLive,
} from "../policy"
import {
  StreamOperationalFailure,
  StreamProviderCorrectnessViolation,
  StreamProviderError,
  StreamStartClientCorrectnessViolation,
  StreamStartOperationalFailure,
  StreamStartProviderRejection,
  acceptedHttpResponse,
  rejectedHttpResponse,
  payloadSample,
  type ProviderRejection,
} from "../failure"
import type {
  ProviderCall,
  StreamFailure,
  StreamOperationalFailureReason,
  StreamStartFailure,
} from "../failure"

const call: ProviderCall = {
  provider: "test",
  model: "test",
  method: "POST",
  url: "https://test",
}

const progress = { dataPayloadsDecoded: 0, modelEventsEmitted: 0 }

function acceptedResponse() {
  return acceptedHttpResponse(200, new Headers())
}

function makeStreamOperationalFailure(reason: StreamOperationalFailureReason): StreamFailure {
  return new StreamOperationalFailure({
    call,
    response: acceptedResponse(),
    reason,
    progress,
  })
}

function makeBodyReadFailure(): StreamFailure {
  return makeStreamOperationalFailure({
    _tag: "BodyReadFailure",
    readError: {
      _tag: "EffectResponseBodyError",
      effectReason: "Decode",
      cause: { _tag: "ErrorCause", name: "TypeError", message: "terminated" },
    },
  })
}

function makeStallTimeout(): StreamFailure {
  return makeStreamOperationalFailure({
    _tag: "StallTimeout",
    timeoutMs: 120_000,
    lastActivity: { _tag: "NoActivity" },
  })
}

function makeClosedWithoutTerminal(): StreamFailure {
  return makeStreamOperationalFailure({
    _tag: "ConnectionClosedWithoutTerminalOutcome",
    expectation: { _tag: "InitialChunk" },
  })
}

function makeInvalidJsonChunk(): StreamFailure {
  return new StreamProviderCorrectnessViolation({
    call,
    response: acceptedResponse(),
    violation: {
      _tag: "InvalidProviderChunk",
      problem: {
        _tag: "InvalidJson",
        payload: payloadSample("not json"),
        cause: { _tag: "ErrorCause", name: "Error", message: "bad json" },
      },
    },
    progress,
  })
}

function makeDoneWithoutTerminal(): StreamFailure {
  return new StreamProviderCorrectnessViolation({
    call,
    response: acceptedResponse(),
    violation: {
      _tag: "SignaledDoneWithoutTerminalOutcome",
      expectation: { _tag: "FinishReasonOrMoreChunks" },
    },
    progress,
  })
}

function makeStreamProviderError(
  message: string,
  code: string | null = null,
): StreamProviderError {
  return new StreamProviderError({
    call,
    response: acceptedResponse(),
    providerError: { message, type: "test", code, param: null },
    payload: payloadSample('{"error":"boom"}'),
    progress,
  })
}

function makeRejection(
  status: number,
  retryAfterMs: number | null = null,
  body?: string,
): ProviderRejection {
  if (status === 429) {
    return {
      _tag: "RateLimited",
      message: "provider error",
      retryPolicy: {
        retry: true,
        retryAfterMs: retryAfterMs === null ? Option.none() : Option.some(retryAfterMs),
      },
    }
  }

  if (status >= 500) {
    return {
      _tag: "UpstreamFailure",
      message: "provider error",
      retryPolicy: { retry: true, retryAfterMs: Option.none() },
    }
  }

  return {
    _tag: "InvalidRequest",
    message: "provider error",
  }
}

function makeStartProviderRejection(status: number, retryAfterMs: number | null = null): StreamStartProviderRejection {
  return new StreamStartProviderRejection({
    call,
    response: rejectedHttpResponse(
      status,
      retryAfterMs !== null ? new Headers({ "retry-after": String(retryAfterMs / 1000) }) : new Headers(),
      JSON.stringify({ error: { message: "provider error", type: "test", code: "provider_error" } }),
    ),
    rejection: makeRejection(status, retryAfterMs),
  })
}

function makeStartOperationalFailure(): StreamStartFailure {
  return new StreamStartOperationalFailure({
    call,
    reason: {
      _tag: "RequestFailedBeforeResponse",
      cause: { _tag: "ErrorCause", name: "Error", message: "network down" },
    },
  })
}

function makeStartClientCorrectnessViolation(): StreamStartFailure {
  return new StreamStartClientCorrectnessViolation({
    call,
    component: "message_encoder",
    message: "Could not encode message",
    evidence: {
      _tag: "MessageEncodingFailed",
      cause: { _tag: "ErrorCause", name: "Error", message: "bad message" },
    },
  })
}

function assertNotRetryable(result: UpstreamRetryability): asserts result is Extract<UpstreamRetryability, { readonly _tag: "UpstreamNotRetryable" }> {
  expect(result._tag).toBe("UpstreamNotRetryable")
}

function assertRetryable(result: UpstreamRetryability): asserts result is Extract<UpstreamRetryability, { readonly _tag: "UpstreamRetryable" }> {
  expect(result._tag).toBe("UpstreamRetryable")
}

describe("defaultRetryabilityForFailure", () => {
  it("stream body read failures are retryable operational failures", () => {
    const result = defaultRetryabilityForFailure(makeBodyReadFailure())
    assertRetryable(result)
  })

  it("stream stall timeouts are retryable operational failures", () => {
    const result = defaultRetryabilityForFailure(makeStallTimeout())
    assertRetryable(result)
  })

  it("stream closes without a terminal outcome are retryable operational failures", () => {
    const result = defaultRetryabilityForFailure(makeClosedWithoutTerminal())
    assertRetryable(result)
  })

  it("invalid provider chunk data is a non-retryable provider correctness violation", () => {
    const result = defaultRetryabilityForFailure(makeInvalidJsonChunk())
    assertNotRetryable(result)
    expect(result.reason).toBe("malformed_provider_data")
  })

  it("provider DONE before terminal outcome is a non-retryable provider correctness violation", () => {
    const result = defaultRetryabilityForFailure(makeDoneWithoutTerminal())
    assertNotRetryable(result)
    expect(result.reason).toBe("malformed_provider_data")
  })

  it("stream provider errors with retryable text are retryable", () => {
    const result = defaultRetryabilityForFailure(makeStreamProviderError("timeout"))
    assertRetryable(result)
  })

  it("stream provider errors with non-retryable text are not retryable", () => {
    const result = defaultRetryabilityForFailure(makeStreamProviderError("bad request"))
    assertNotRetryable(result)
    expect(result.reason).toBe("provider_error_not_retryable")
  })

  it("stream-start provider 429 honors Retry-After", () => {
    const result = defaultRetryabilityForFailure(makeStartProviderRejection(429, 5000))
    assertRetryable(result)
    expect(result.retryAfter._tag).toBe("RetryAfterMs")
    if (result.retryAfter._tag === "RetryAfterMs") {
      expect(result.retryAfter.ms).toBe(5000)
    }
  })

  it("stream-start provider 500 is retryable", () => {
    const result = defaultRetryabilityForFailure(makeStartProviderRejection(500))
    assertRetryable(result)
    expect(result.retryAfter._tag).toBe("NoRetryAfter")
  })

  it("stream-start provider 400 is not retryable", () => {
    const result = defaultRetryabilityForFailure(makeStartProviderRejection(400))
    assertNotRetryable(result)
    expect(result.reason).toBe("invalid_request")
  })

  it("stream-start transport failures are retryable", () => {
    const result = defaultRetryabilityForFailure(makeStartOperationalFailure())
    assertRetryable(result)
  })

  it("stream-start client correctness violations are non-retryable defects", () => {
    const result = defaultRetryabilityForFailure(makeStartClientCorrectnessViolation())
    assertNotRetryable(result)
    expect(result.reason).toBe("internal_defect")
  })
})

describe("AiRetryPolicyLive", () => {
  it("provides the default implementation via Effect service", async () => {
    const failure = makeBodyReadFailure()

    const program = Effect.gen(function* () {
      const policy = yield* AiRetryPolicy
      return yield* policy.upstreamRetryability(failure)
    })

    const result = await Effect.runPromise(Effect.provide(program, AiRetryPolicyLive))
    expect(result._tag).toBe("UpstreamRetryable")
  })
})

describe("AiBillingPolicyLive", () => {
  it("provides the default implementation via Effect service", async () => {
    const program = Effect.gen(function* () {
      const policy = yield* AiBillingPolicy
      return yield* policy.billingDisposition({} as any)
    })

    const result = await Effect.runPromise(Effect.provide(program, AiBillingPolicyLive))
    expect(result._tag).toBe("UseLocalEstimate")
    if (result._tag === "UseLocalEstimate") {
      expect(result.reason).toBe("terminal_not_completed_or_usage_missing")
    }
  })
})
