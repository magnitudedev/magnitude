import { Cause, Data, Option } from "effect"
import type { ResponseUsage } from "../response/usage"
import type { FinishReason } from "../response/events"
import type { ToolCallId, ProviderToolCallId } from "../prompt/ids"

export interface ProviderCall {
  readonly provider: string
  readonly model: string
  readonly method: "POST"
  readonly url: string
}

export type HeaderList = ReadonlyArray<readonly [name: string, value: string]>

export interface AcceptedHttpResponse {
  readonly status: number
  readonly headers: HeaderList
  readonly requestId: string | null
}

export interface RejectedHttpResponse {
  readonly status: number
  readonly headers: HeaderList
  readonly body: string
  readonly requestId: string | null
  readonly retryAfterMs: number | null
}

export type CauseInfo =
  | { readonly _tag: "ErrorCause"; readonly name: string; readonly message: string }
  | { readonly _tag: "StringCause"; readonly message: string }
  | { readonly _tag: "Cause"; readonly pretty: string }
  | { readonly _tag: "UnknownCause"; readonly description: string }

export interface PayloadSample {
  readonly text: string
  readonly encodedBytes: number
  readonly truncated: boolean
}

export interface StreamProgress {
  readonly dataPayloadsDecoded: number
  readonly modelEventsEmitted: number
}

export type BodyReadError =
  | { readonly _tag: "EffectResponseBodyError"; readonly effectReason: string; readonly cause: CauseInfo }
  | { readonly _tag: "ReadableStreamError"; readonly cause: CauseInfo }

export type LastStreamActivity =
  | { readonly _tag: "NoActivity" }
  | { readonly _tag: "ResponseAccepted"; readonly atEpochMs: number }
  | { readonly _tag: "BodyBytesRead"; readonly atEpochMs: number }
  | { readonly _tag: "DataPayloadDecoded"; readonly atEpochMs: number }

export type DecoderExpectation =
  | { readonly _tag: "InitialChunk" }
  | { readonly _tag: "FinishReasonOrMoreChunks" }
  | { readonly _tag: "UsageChunk"; readonly pendingReason: "completed" | "validation_failure" }

export interface ProviderErrorEnvelope {
  readonly message: string
  readonly type: string | null
  readonly code: string | null
  readonly param: string | null
}

export type RetryPolicy = {
  readonly retry: boolean
  readonly retryAfterMs: Option.Option<number>
}

export type ProviderRejection =
  | { readonly _tag: "AuthRejected"; readonly message: string }
  | { readonly _tag: "InsufficientCredits"; readonly message: string; readonly balanceCents: Option.Option<number> }
  | { readonly _tag: "ModelUnavailable"; readonly message: string }
  | { readonly _tag: "ModelCapabilityMissing"; readonly message: string }
  | { readonly _tag: "ProviderCapabilityMissing"; readonly message: string }
  | { readonly _tag: "ContextLimitExceeded"; readonly message: string }
  | { readonly _tag: "InvalidRequest"; readonly message: string }
  | { readonly _tag: "ProviderInvariantViolation"; readonly message: string }
  | { readonly _tag: "RateLimited"; readonly message: string; readonly retryPolicy: RetryPolicy }
  | { readonly _tag: "UpstreamFailure"; readonly message: string; readonly retryPolicy: RetryPolicy }

export interface SchemaIssue {
  readonly message: string
  /** Path into the offending value, when the validator reports one. */
  readonly path?: readonly PropertyKey[]
}

export type StreamStartOperationalFailureReason =
  | { readonly _tag: "RequestFailedBeforeResponse"; readonly cause: CauseInfo }

export class StreamStartOperationalFailure extends Data.TaggedError("StreamStartOperationalFailure")<{
  readonly call: ProviderCall
  readonly reason: StreamStartOperationalFailureReason
}> {}

export class StreamStartProviderRejection extends Data.TaggedError("StreamStartProviderRejection")<{
  readonly call: ProviderCall
  readonly response: RejectedHttpResponse
  readonly rejection: ProviderRejection
}> {}

export type StreamStartProviderCorrectnessViolationReason =
  | { readonly _tag: "InvalidErrorEnvelope"; readonly status: number; readonly body: PayloadSample; readonly issue: SchemaIssue }
  | { readonly _tag: "MissingRequiredResponseMetadata"; readonly field: "requestId"; readonly status: number; readonly body: PayloadSample }
  | { readonly _tag: "UnexpectedResponseShape"; readonly status: number; readonly body: PayloadSample; readonly issue: SchemaIssue }

export class StreamStartProviderCorrectnessViolation extends Data.TaggedError("StreamStartProviderCorrectnessViolation")<{
  readonly call: ProviderCall
  readonly response: RejectedHttpResponse | null
  readonly violation: StreamStartProviderCorrectnessViolationReason
}> {}

export type StreamStartClientCorrectnessEvidence =
  | { readonly _tag: "MessageEncodingFailed"; readonly cause: CauseInfo }
  | { readonly _tag: "ToolSchemaEncodingFailed"; readonly cause: CauseInfo }
  | { readonly _tag: "RequestBodyEncodingFailed"; readonly cause: CauseInfo }
  | { readonly _tag: "AuthApplicationFailed"; readonly cause: CauseInfo }
  | { readonly _tag: "UnexpectedDefectCaught"; readonly cause: CauseInfo }

export type StreamStartClientComponent =
  | "message_encoder"
  | "tool_schema_encoder"
  | "request_body_encoder"
  | "auth_applicator"
  | "request_builder"

export class StreamStartClientCorrectnessViolation extends Data.TaggedError("StreamStartClientCorrectnessViolation")<{
  readonly call: ProviderCall
  readonly component: StreamStartClientComponent
  readonly message: string
  readonly evidence: StreamStartClientCorrectnessEvidence
}> {}

export type StreamStartFailure =
  | StreamStartOperationalFailure
  | StreamStartProviderRejection
  | StreamStartProviderCorrectnessViolation
  | StreamStartClientCorrectnessViolation

export type StreamOperationalFailureReason =
  | { readonly _tag: "BodyReadFailure"; readonly readError: BodyReadError }
  | { readonly _tag: "StallTimeout"; readonly timeoutMs: number; readonly lastActivity: LastStreamActivity }
  | { readonly _tag: "ConnectionClosedWithoutTerminalOutcome"; readonly expectation: DecoderExpectation }

export class StreamOperationalFailure extends Data.TaggedError("StreamOperationalFailure")<{
  readonly call: ProviderCall
  readonly response: AcceptedHttpResponse
  readonly progress: StreamProgress
  readonly reason: StreamOperationalFailureReason
}> {}

export class StreamProviderError extends Data.TaggedError("StreamProviderError")<{
  readonly call: ProviderCall
  readonly response: AcceptedHttpResponse
  readonly providerError: ProviderErrorEnvelope
  readonly payload: PayloadSample
  readonly progress: StreamProgress
}> {}

export type InvalidProviderChunkProblem =
  | { readonly _tag: "InvalidJson"; readonly payload: PayloadSample; readonly cause: CauseInfo }
  | { readonly _tag: "InvalidChunkSchema"; readonly payload: PayloadSample; readonly issue: SchemaIssue; readonly cause: CauseInfo }

export type InvalidConstrainedOutput =
  | {
      readonly _tag: "InvalidToolInput"
      readonly toolCallId: ToolCallId
      readonly providerToolCallId: ProviderToolCallId
      readonly toolName: string
      readonly issue: SchemaIssue
    }

export type StreamProviderCorrectnessViolationReason =
  | { readonly _tag: "SignaledDoneWithoutTerminalOutcome"; readonly expectation: DecoderExpectation }
  | { readonly _tag: "InvalidProviderChunk"; readonly problem: InvalidProviderChunkProblem }
  | { readonly _tag: "InvalidConstrainedOutput"; readonly output: InvalidConstrainedOutput }

export class StreamProviderCorrectnessViolation extends Data.TaggedError("StreamProviderCorrectnessViolation")<{
  readonly call: ProviderCall
  readonly response: AcceptedHttpResponse
  readonly progress: StreamProgress
  readonly violation: StreamProviderCorrectnessViolationReason
}> {}

export type StreamClientCorrectnessEvidence =
  | { readonly _tag: "InvariantViolated"; readonly invariant: string }
  | { readonly _tag: "UnexpectedDefectCaught"; readonly cause: CauseInfo }

export type StreamClientComponent =
  | "request_builder"
  | "transport"
  | "framing"
  | "chunk_decoder"
  | "model_event_reducer"
  | "harness_adapter"
  | "direct_stream_collector"

export class StreamClientCorrectnessViolation extends Data.TaggedError("StreamClientCorrectnessViolation")<{
  readonly call: ProviderCall
  readonly response: AcceptedHttpResponse
  readonly component: StreamClientComponent
  readonly message: string
  readonly evidence: StreamClientCorrectnessEvidence
  readonly progress: StreamProgress
}> {}

export type StreamFailure =
  | StreamOperationalFailure
  | StreamProviderError
  | StreamProviderCorrectnessViolation
  | StreamClientCorrectnessViolation

export type ModelAttemptFailure = StreamStartFailure | StreamFailure

export function isStreamOperationalFailure(
  failure: StreamFailure,
): failure is Extract<StreamFailure, { readonly _tag: "StreamOperationalFailure" }> {
  return failure._tag === "StreamOperationalFailure"
}

export function isStreamProviderCorrectnessViolation(
  failure: StreamFailure,
): failure is Extract<StreamFailure, { readonly _tag: "StreamProviderCorrectnessViolation" }> {
  return failure._tag === "StreamProviderCorrectnessViolation"
}

// ---------------------------------------------------------------------------
// Usage at Termination (§4.1)
// ---------------------------------------------------------------------------

export type UsageAtTermination =
  | { readonly _tag: "UsageReported"; readonly usage: ResponseUsage }
  | { readonly _tag: "UsageNotReported"; readonly reason: UsageMissingReason }

export type UsageMissingReason =
  | "stream_failed_before_usage"
  | "provider_closed_before_usage"
  | "provider_does_not_report_usage"
  | "usage_chunk_never_arrived"

// ---------------------------------------------------------------------------
// Model Stream Terminal (§9)
// ---------------------------------------------------------------------------

export type ModelStreamTerminal = Data.TaggedEnum<{
  StreamCompleted: {
    readonly call: ProviderCall
    readonly response: AcceptedHttpResponse
    readonly finishReason: FinishReason
    readonly progress: StreamProgress
    readonly usage: UsageAtTermination
  }
  StreamFailed: {
    readonly cause: StreamFailure
    readonly usage: UsageAtTermination
  }
}>

const makeModelStreamTerminal = Data.taggedEnum<ModelStreamTerminal>()

type ModelStreamTerminalTag = ModelStreamTerminal["_tag"]
type ModelStreamTerminalArgs<Tag extends ModelStreamTerminalTag> =
  Omit<Extract<ModelStreamTerminal, { readonly _tag: Tag }>, "_tag">

function makeTerminal<Tag extends ModelStreamTerminalTag>(
  tag: Tag,
): (args: ModelStreamTerminalArgs<Tag>) => Extract<ModelStreamTerminal, { readonly _tag: Tag }> {
  return (args) => ({ _tag: tag, ...args }) as Extract<ModelStreamTerminal, { readonly _tag: Tag }>
}

export const ModelStreamTerminal = {
  StreamCompleted: makeTerminal("StreamCompleted"),
  StreamFailed: makeTerminal("StreamFailed"),

  hadPartialOutput: (terminal: ModelStreamTerminal): boolean => {
    const progress = makeModelStreamTerminal.$match({
      StreamCompleted: (t) => t.progress,
      StreamFailed: (t) => t.cause.progress,
    })(terminal)
    return progress.modelEventsEmitted > 0
  },
  $is: makeModelStreamTerminal.$is,
  $match: makeModelStreamTerminal.$match,
}

export interface StreamFailureContext {
  readonly responseHeaders: Headers
  readonly call: ProviderCall
  readonly response: AcceptedHttpResponse
}

const REQUEST_ID_HEADER = "x-request-id"

const textEncoder = new TextEncoder()

export function headerListFromHeaders(headers: Headers | Record<string, string | undefined>): HeaderList {
  const result: Array<readonly [string, string]> = []
  if (headers instanceof Headers) {
    headers.forEach((value, name) => {
      result.push([name.toLowerCase(), value])
    })
  } else {
    for (const [name, value] of Object.entries(headers)) {
      if (value !== undefined) result.push([name.toLowerCase(), value])
    }
  }
  return result
}

export function headersFromHeaderList(headers: HeaderList): Headers {
  const result = new Headers()
  for (const [name, value] of headers) {
    result.set(name, value)
  }
  return result
}

export function getHeader(headers: HeaderList, name: string): string | null {
  const lower = name.toLowerCase()
  for (const [headerName, value] of headers) {
    if (headerName.toLowerCase() === lower) return value
  }
  return null
}

export function retryAfterMsFromHeaders(headers: HeaderList): number | null {
  const value = getHeader(headers, "retry-after")
  if (value === null) return null

  const seconds = Number(value)
  if (Number.isFinite(seconds)) return seconds * 1000

  const date = Date.parse(value)
  if (Number.isFinite(date)) return Math.max(0, date - Date.now())

  return null
}

export function acceptedHttpResponse(
  status: number,
  headers: Headers | Record<string, string | undefined>,
): AcceptedHttpResponse {
  const headerList = headerListFromHeaders(headers)
  return {
    status,
    headers: headerList,
    requestId: getHeader(headerList, REQUEST_ID_HEADER),
  }
}

export function rejectedHttpResponse(
  status: number,
  headers: Headers | Record<string, string | undefined>,
  body: string,
): RejectedHttpResponse {
  const headerList = headerListFromHeaders(headers)
  return {
    status,
    headers: headerList,
    body,
    requestId: getHeader(headerList, REQUEST_ID_HEADER),
    retryAfterMs: retryAfterMsFromHeaders(headerList),
  }
}

export function toCauseInfo(cause: unknown): CauseInfo {
  if (Cause.isCause(cause)) {
    return { _tag: "Cause", pretty: Cause.pretty(cause) }
  }
  if (cause instanceof Error) {
    return {
      _tag: "ErrorCause",
      name: cause.name || "Error",
      message: cause.message || cause.name || "Error",
    }
  }
  if (typeof cause === "string") {
    return { _tag: "StringCause", message: cause }
  }
  return { _tag: "UnknownCause", description: describeUnknown(cause) }
}

export function payloadSample(text: string, maxBytes = 4096): PayloadSample {
  const bytes = textEncoder.encode(text)
  if (bytes.byteLength <= maxBytes) {
    return { text, encodedBytes: bytes.byteLength, truncated: false }
  }

  const slice = bytes.slice(0, maxBytes)
  return {
    text: new TextDecoder().decode(slice),
    encodedBytes: bytes.byteLength,
    truncated: true,
  }
}

export function causeInfoText(cause: CauseInfo): string {
  switch (cause._tag) {
    case "ErrorCause":
      return `${cause.name}: ${cause.message}`
    case "StringCause":
      return cause.message
    case "Cause":
      return cause.pretty
    case "UnknownCause":
      return cause.description
  }
}

function describeUnknown(value: unknown): string {
  if (value == null) return "unavailable"
  try {
    return JSON.stringify(value)
  } catch {
    return String(value)
  }
}
