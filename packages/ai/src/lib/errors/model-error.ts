import { Data } from "effect"

export class NotConfigured extends Data.TaggedError("NotConfigured")<{
  readonly providerId: string
  readonly message: string
}> {}

export class AuthFailed extends Data.TaggedError("AuthFailed")<{
  readonly providerId: string
  readonly status: number | null
  readonly message: string
}> {}

export class RateLimited extends Data.TaggedError("RateLimited")<{
  readonly providerId: string
  readonly status: number | null
  readonly message: string
  readonly retryAfterMs: number | null
}> {}

export class UsageLimitExceeded extends Data.TaggedError("UsageLimitExceeded")<{
  readonly providerId: string
  readonly status: number | null
  readonly message: string
}> {}

export class ContextLimitExceeded extends Data.TaggedError("ContextLimitExceeded")<{
  readonly providerId: string
  readonly status: number | null
  readonly message: string
}> {}

export class InvalidRequest extends Data.TaggedError("InvalidRequest")<{
  readonly providerId: string
  readonly status: number | null
  readonly message: string
}> {}

export class TransportError extends Data.TaggedError("TransportError")<{
  readonly providerId: string
  readonly status: number | null
  readonly message: string
  readonly retryable: boolean
}> {}

export class ParseError extends Data.TaggedError("ParseError")<{
  readonly providerId: string
  readonly message: string
}> {}

export type ModelError =
  | NotConfigured
  | AuthFailed
  | RateLimited
  | UsageLimitExceeded
  | ContextLimitExceeded
  | InvalidRequest
  | TransportError
  | ParseError
