import type {
  CauseInfo,
  DecoderExpectation,
  LastStreamActivity,
  ModelAttemptFailure,
  ProviderCall,
  ProviderErrorEnvelope,
  ProviderRejection,
  RejectedHttpResponse,
  SchemaIssue,
  StreamClientCorrectnessEvidence,
  StreamFailure,
  StreamOperationalFailureReason,
  StreamProgress,
  StreamProviderCorrectnessViolationReason,
  StreamStartClientCorrectnessEvidence,
  StreamStartFailure,
  StreamStartProviderCorrectnessViolationReason,
} from "./failure"
import { Option } from "effect"
import {
  causeInfoText,
  payloadSample,
  StreamStartProviderCorrectnessViolation,
  StreamStartProviderRejection,
} from "./failure"

const CONTEXT_LIMIT_PATTERNS = [
  "prompt is too long",
  "token count exceeds the maximum",
  "maximum context length",
  "context_length_exceeded",
  "exceeded model token limit",
]

const TRANSIENT_PROVIDER_PATTERNS = [
  "timeout",
  "timed_out",
  "temporarily_unavailable",
  "unavailable",
  "overloaded",
  "rate_limit",
  "server_error",
  "internal_error",
  "upstream_unavailable",
  "stream_interrupted",
]

function hasPattern(text: string, patterns: readonly string[]): boolean {
  return patterns.some((pattern) => text.includes(pattern))
}

function tryParseJsonObject(text: string): Record<string, unknown> | null {
  try {
    const parsed = JSON.parse(text)
    if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
      return parsed as Record<string, unknown>
    }
  } catch {
    // fall through
  }

  let offset = 0
  while ((offset = text.indexOf("{", offset)) !== -1) {
    try {
      const parsed = JSON.parse(text.slice(offset))
      if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
        return parsed as Record<string, unknown>
      }
    } catch {
      // try next object start
    }
    offset += 1
  }

  return null
}

function getNestedErrorObject(text: string): Record<string, unknown> | null {
  const parsed = tryParseJsonObject(text)
  const error = parsed?.error
  return error && typeof error === "object" && !Array.isArray(error)
    ? (error as Record<string, unknown>)
    : null
}

function providerErrorEnvelopeFromBody(body: string): ProviderErrorEnvelope | null {
  const error = getNestedErrorObject(body)
  if (error === null) return null

  const message = error.message
  if (typeof message !== "string" || message.trim().length === 0) return null

  const type = error.type
  const code = error.code
  const param = error.param
  return {
    message,
    type: typeof type === "string" ? type : null,
    code: typeof code === "string" ? code : null,
    param: typeof param === "string" ? param : null,
  }
}

function retryPolicy(retry: boolean, retryAfterMs: number | null): {
  readonly retry: boolean
  readonly retryAfterMs: Option.Option<number>
} {
  return {
    retry,
    retryAfterMs: retryAfterMs === null ? Option.none() : Option.some(retryAfterMs),
  }
}

function defaultProviderRejection(
  envelope: ProviderErrorEnvelope,
  response: RejectedHttpResponse,
): ProviderRejection {
  const text = providerErrorText(envelope)
  const status = response.status
  const base = {
    message: envelope.message,
  }

  if (status === 401 || status === 403 || hasPattern(text, [
    "missing_scope",
    "insufficient_scope",
    "invalid_token",
    "token expired",
    "expired token",
    "unauthorized",
    "forbidden",
    "authentication",
    "invalid_api_key",
  ])) {
    return {
      _tag: "AuthRejected",
      ...base,
    }
  }

  if (hasPattern(text, CONTEXT_LIMIT_PATTERNS)) {
    return {
      _tag: "ContextLimitExceeded",
      ...base,
    }
  }

  if (status === 429) {
    return {
      _tag: "RateLimited",
      ...base,
      retryPolicy: retryPolicy(true, response.retryAfterMs),
    }
  }

  if (status >= 400 && status < 500) {
    return {
      _tag: "InvalidRequest",
      ...base,
    }
  }

  return {
    _tag: "UpstreamFailure",
    ...base,
    retryPolicy: retryPolicy(
      status >= 500 || hasPattern(text, TRANSIENT_PROVIDER_PATTERNS),
      null,
    ),
  }
}

export function streamStartFailureFromRejectedResponse(
  call: ProviderCall,
  response: RejectedHttpResponse,
): StreamStartProviderRejection | StreamStartProviderCorrectnessViolation {
  const providerError = providerErrorEnvelopeFromBody(response.body)
  if (providerError !== null) {
    return new StreamStartProviderRejection({
      call,
      response,
      rejection: defaultProviderRejection(providerError, response),
    })
  }

  return new StreamStartProviderCorrectnessViolation({
    call,
    response,
    violation: {
      _tag: "InvalidErrorEnvelope",
      status: response.status,
      body: payloadSample(response.body),
      issue: { message: "Rejected response body did not contain a valid error envelope" },
    },
  })
}

export function formatStreamStartFailureMessage(
  failure: StreamStartFailure,
): string {
  switch (failure._tag) {
    case "StreamStartOperationalFailure":
      return [
        "Model request failed before any response was accepted",
        `request: ${failure.call.method} ${failure.call.url}`,
        `reason: ${failure.reason._tag}`,
        `cause: ${causeInfoText(failure.reason.cause)}`,
      ].join("\n")
    case "StreamStartProviderRejection":
      return [
        "Model provider rejected the request",
        `response: ${failure.response.status} ${failure.call.method} ${failure.call.url}`,
        `rejection: ${failure.rejection._tag}`,
        `message: ${failure.rejection.message}`,
      ].join("\n")
    case "StreamStartProviderCorrectnessViolation":
      return [
        "Model provider rejected the request with an invalid error response",
        `response: ${failure.response?.status ?? "unavailable"} ${failure.call.method} ${failure.call.url}`,
        `violation: ${formatStreamStartProviderViolation(failure.violation)}`,
      ].join("\n")
    case "StreamStartClientCorrectnessViolation":
      return [
        "Stream-start client correctness violation",
        `component: ${failure.component}`,
        `message: ${failure.message}`,
        `evidence: ${formatStreamStartClientCorrectnessEvidence(failure.evidence)}`,
      ].join("\n")
  }
}

export function formatStreamFailureMessage(failure: StreamFailure): string {
  switch (failure._tag) {
    case "StreamOperationalFailure":
      return [
        "Model response stream failed operationally",
        `response: ${failure.response.status} ${failure.call.method} ${failure.call.url}`,
        `reason: ${formatStreamOperationalReason(failure.reason)}`,
      ].join("\n")
    case "StreamProviderError":
      return [
        "Model stream ended with provider error envelope",
        `response: ${failure.response.status} ${failure.call.method} ${failure.call.url}`,
        `message: ${failure.providerError.message}`,
        ...(failure.providerError.type !== null ? [`type: ${failure.providerError.type}`] : []),
        ...(failure.providerError.code !== null ? [`code: ${failure.providerError.code}`] : []),
        ...(failure.providerError.param !== null ? [`param: ${failure.providerError.param}`] : []),
      ].join("\n")
    case "StreamProviderCorrectnessViolation":
      return [
        "Model provider violated the stream/output contract",
        `response: ${failure.response.status} ${failure.call.method} ${failure.call.url}`,
        `violation: ${formatStreamProviderViolation(failure.violation)}`,
      ].join("\n")
    case "StreamClientCorrectnessViolation":
      return [
        "Stream client correctness violation",
        `component: ${failure.component}`,
        `message: ${failure.message}`,
        `evidence: ${formatStreamClientCorrectnessEvidence(failure.evidence)}`,
      ].join("\n")
  }
}

export function formatModelAttemptFailureMessage(
  failure: ModelAttemptFailure,
): string {
  return failure._tag.startsWith("StreamStart")
    ? formatStreamStartFailureMessage(failure as StreamStartFailure)
    : formatStreamFailureMessage(failure as StreamFailure)
}

export interface ModelAttemptFailureSnapshot {
  readonly phase: "stream_start" | "stream"
  readonly tag: ModelAttemptFailure["_tag"]
  readonly detailTag: string
  readonly message: string
  /** Exact provider-authored message, when the failure carried a valid provider error envelope. */
  readonly providerMessage?: string
  readonly call: ProviderCall
  readonly responseStatus: number | null
  readonly progress: StreamProgress | null
  readonly retryable: boolean
  readonly retryAfterMs: number | null
}

export function snapshotModelAttemptFailure(
  failure: ModelAttemptFailure,
): ModelAttemptFailureSnapshot {
  const retry = retryHintForSnapshot(failure)
  return {
    phase: failure._tag.startsWith("StreamStart") ? "stream_start" : "stream",
    tag: failure._tag,
    detailTag: detailTag(failure),
    message: formatModelAttemptFailureMessage(failure),
    ...(failure._tag === "StreamProviderError"
      ? { providerMessage: failure.providerError.message }
      : failure._tag === "StreamStartProviderRejection"
        ? { providerMessage: failure.rejection.message }
        : {}),
    call: failure.call,
    responseStatus: "response" in failure && failure.response !== null ? failure.response.status : null,
    progress: "progress" in failure ? failure.progress : null,
    retryable: retry.retryable,
    retryAfterMs: retry.retryAfterMs,
  }
}

function detailTag(
  failure: ModelAttemptFailure,
): string {
  switch (failure._tag) {
    case "StreamStartOperationalFailure":
      return failure.reason._tag
    case "StreamStartProviderRejection":
      return failure.rejection._tag
    case "StreamStartProviderCorrectnessViolation":
      return failure.violation._tag
    case "StreamStartClientCorrectnessViolation":
      return failure.evidence._tag
    case "StreamOperationalFailure":
      return failure.reason._tag
    case "StreamProviderError":
      return failure.providerError.code ?? failure.providerError.type ?? "ProviderError"
    case "StreamProviderCorrectnessViolation":
      return streamProviderViolationDetailTag(failure.violation)
    case "StreamClientCorrectnessViolation":
      return failure.evidence._tag
  }
}

function streamProviderViolationDetailTag(violation: StreamProviderCorrectnessViolationReason): string {
  switch (violation._tag) {
    case "InvalidProviderChunk":
      return `${violation._tag}.${violation.problem._tag}`
    case "InvalidConstrainedOutput":
      return `${violation._tag}.${violation.output._tag}`
    case "SignaledDoneWithoutTerminalOutcome":
      return violation._tag
  }
}

function retryHintForSnapshot(
  failure: ModelAttemptFailure,
): { readonly retryable: boolean; readonly retryAfterMs: number | null } {
  switch (failure._tag) {
    case "StreamStartOperationalFailure":
    case "StreamOperationalFailure":
      return { retryable: true, retryAfterMs: null }
    case "StreamStartProviderRejection":
      if ("retryPolicy" in failure.rejection) {
        return {
          retryable: failure.rejection.retryPolicy.retry,
          retryAfterMs: Option.getOrNull(failure.rejection.retryPolicy.retryAfterMs),
        }
      }
      return { retryable: false, retryAfterMs: null }
    case "StreamProviderError":
      return { retryable: providerErrorRetryable(failure.providerError, failure.response.status), retryAfterMs: null }
    case "StreamStartProviderCorrectnessViolation":
    case "StreamStartClientCorrectnessViolation":
    case "StreamProviderCorrectnessViolation":
    case "StreamClientCorrectnessViolation":
      return { retryable: false, retryAfterMs: null }
  }
}

function providerErrorText(error: ProviderErrorEnvelope): string {
  return [error.message, error.type ?? "", error.code ?? "", error.param ?? ""]
    .join(" ")
    .toLowerCase()
}

function providerErrorRetryable(error: ProviderErrorEnvelope, status: number): boolean {
  const text = providerErrorText(error)
  if (status === 429 || status >= 500) return !hasPattern(text, CONTEXT_LIMIT_PATTERNS)
  return hasPattern(text, TRANSIENT_PROVIDER_PATTERNS)
}

function formatStreamOperationalReason(reason: StreamOperationalFailureReason): string {
  switch (reason._tag) {
    case "BodyReadFailure":
      return [
        "BodyReadFailure",
        reason.readError._tag === "EffectResponseBodyError"
          ? `effectReason=${reason.readError.effectReason}`
          : "reader=ReadableStream",
        `cause=${causeInfoText(reason.readError.cause)}`,
      ].join(" ")
    case "StallTimeout":
      return `StallTimeout timeoutMs=${reason.timeoutMs} lastActivity=${formatLastActivity(reason.lastActivity)}`
    case "ConnectionClosedWithoutTerminalOutcome":
      return `ConnectionClosedWithoutTerminalOutcome expected=${formatDecoderExpectation(reason.expectation)}`
  }
}

function formatStreamStartProviderViolation(
  violation: StreamStartProviderCorrectnessViolationReason,
): string {
  switch (violation._tag) {
    case "InvalidErrorEnvelope":
      return `InvalidErrorEnvelope status=${violation.status} issue=${violation.issue.message}`
    case "MissingRequiredResponseMetadata":
      return `MissingRequiredResponseMetadata field=${violation.field} status=${violation.status}`
    case "UnexpectedResponseShape":
      return `UnexpectedResponseShape status=${violation.status} issue=${violation.issue.message}`
  }
}

function formatStreamProviderViolation(violation: StreamProviderCorrectnessViolationReason): string {
  switch (violation._tag) {
    case "SignaledDoneWithoutTerminalOutcome":
      return `SignaledDoneWithoutTerminalOutcome expected=${formatDecoderExpectation(violation.expectation)}`
    case "InvalidProviderChunk":
      switch (violation.problem._tag) {
        case "InvalidJson":
          return `InvalidProviderChunk problem=InvalidJson cause=${causeInfoText(violation.problem.cause)} payload=${violation.problem.payload.text}`
        case "InvalidChunkSchema":
          return `InvalidProviderChunk problem=InvalidChunkSchema issue=${violation.problem.issue.message} cause=${causeInfoText(violation.problem.cause)} payload=${violation.problem.payload.text}`
      }
    case "InvalidConstrainedOutput":
      return `InvalidConstrainedOutput output=${violation.output._tag} tool=${violation.output.toolName} issue=${violation.output.issue.message}`
  }
}

function formatLastActivity(activity: LastStreamActivity): string {
  switch (activity._tag) {
    case "NoActivity":
      return "none"
    case "ResponseAccepted":
      return `response accepted at ${activity.atEpochMs}ms epoch`
    case "BodyBytesRead":
      return `body bytes read at ${activity.atEpochMs}ms epoch`
    case "DataPayloadDecoded":
      return `data payload decoded at ${activity.atEpochMs}ms epoch`
  }
}

function formatDecoderExpectation(expectation: DecoderExpectation): string {
  switch (expectation._tag) {
    case "InitialChunk":
      return "initial chunk"
    case "FinishReasonOrMoreChunks":
      return "finish reason or more chunks"
    case "UsageChunk":
      return `usage chunk after ${expectation.pendingReason}`
  }
}

function formatStreamStartClientCorrectnessEvidence(
  evidence: StreamStartClientCorrectnessEvidence,
): string {
  switch (evidence._tag) {
    case "MessageEncodingFailed":
    case "ToolSchemaEncodingFailed":
    case "RequestBodyEncodingFailed":
    case "AuthApplicationFailed":
    case "UnexpectedDefectCaught":
      return `${evidence._tag}: ${causeInfoText(evidence.cause)}`
  }
}

function formatStreamClientCorrectnessEvidence(
  evidence: StreamClientCorrectnessEvidence,
): string {
  switch (evidence._tag) {
    case "InvariantViolated":
      return `invariant violated: ${evidence.invariant}`
    case "UnexpectedDefectCaught":
      return causeInfoText(evidence.cause)
  }
}
