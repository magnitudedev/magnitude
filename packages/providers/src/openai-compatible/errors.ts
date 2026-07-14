import { Option } from "effect"
import {
  payloadSample,
  StreamStartProviderCorrectnessViolation,
  StreamStartProviderRejection,
  type ProviderCall,
  type ProviderRejection,
  type RejectedHttpResponse,
} from "@magnitudedev/ai"

interface OpenAiErrorBody {
  readonly message: string
  readonly type?: string
  readonly code?: string
}

function parseError(body: string): OpenAiErrorBody | null {
  try {
    const parsed: unknown = JSON.parse(body)
    if (typeof parsed !== "object" || parsed === null) return null
    const envelope = parsed as Record<string, unknown>
    const candidate = typeof envelope.error === "object" && envelope.error !== null
      ? envelope.error as Record<string, unknown>
      : envelope
    if (typeof candidate.message !== "string") return null
    return {
      message: candidate.message,
      ...(typeof candidate.type === "string" ? { type: candidate.type } : {}),
      ...(typeof candidate.code === "string" ? { code: candidate.code } : {}),
    }
  } catch {
    return null
  }
}

function classify(response: RejectedHttpResponse, error: OpenAiErrorBody | null): ProviderRejection {
  const message = error?.message ?? `HTTP ${response.status}`
  const normalized = message.toLowerCase()
  if (response.status === 401 || response.status === 403) return { _tag: "AuthRejected", message }
  if (response.status === 404 || error?.code === "model_not_found") return { _tag: "ModelUnavailable", message }
  if (response.status === 429) {
    return {
      _tag: "RateLimited",
      message,
      retryPolicy: {
        retry: true,
        retryAfterMs: response.retryAfterMs === null ? Option.none() : Option.some(response.retryAfterMs),
      },
    }
  }
  if (response.status === 413 || normalized.includes("context length") || normalized.includes("too many tokens")) {
    return { _tag: "ContextLimitExceeded", message }
  }
  if (response.status >= 500) {
    return { _tag: "UpstreamFailure", message, retryPolicy: { retry: true, retryAfterMs: Option.none() } }
  }
  return { _tag: "InvalidRequest", message }
}

export function classifyOpenAiCompatibleRejectedResponse(
  providerName: string,
  call: ProviderCall,
  response: RejectedHttpResponse,
): StreamStartProviderRejection | StreamStartProviderCorrectnessViolation {
  const parsed = parseError(response.body)
  if (!parsed && response.body.trim()) {
    return new StreamStartProviderCorrectnessViolation({
      call,
      response,
      violation: {
        _tag: "InvalidErrorEnvelope",
        status: response.status,
        body: payloadSample(response.body),
        issue: { message: `${providerName} returned an invalid OpenAI-compatible error envelope` },
      },
    })
  }
  return new StreamStartProviderRejection({
    call,
    response,
    rejection: classify(response, parsed),
  })
}

