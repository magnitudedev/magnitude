import { Data } from "effect"

export class AuthFailed extends Data.TaggedError("AuthFailed")<{
  readonly status: number
  readonly message: string
}> {}

export class RateLimited extends Data.TaggedError("RateLimited")<{
  readonly status: number
  readonly message: string
  readonly retryAfterMs: number | null
}> {}

export class UsageLimitExceeded extends Data.TaggedError("UsageLimitExceeded")<{
  readonly status: number
  readonly message: string
}> {}

export class ContextLimitExceeded extends Data.TaggedError("ContextLimitExceeded")<{
  readonly status: number
  readonly message: string
}> {}

export class InvalidRequest extends Data.TaggedError("InvalidRequest")<{
  readonly status: number
  readonly message: string
}> {}

export class TransportError extends Data.TaggedError("TransportError")<{
  readonly status: number | null
  readonly message: string
  readonly retryable: boolean
}> {}

export class ParseError extends Data.TaggedError("ParseError")<{
  readonly message: string
}> {}

export type ConnectionError =
  | AuthFailed
  | RateLimited
  | UsageLimitExceeded
  | ContextLimitExceeded
  | InvalidRequest
  | TransportError

export type StreamError =
  | TransportError
  | ParseError
