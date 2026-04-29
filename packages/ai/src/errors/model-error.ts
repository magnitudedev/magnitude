import { Data } from "effect"

export class AuthFailed extends Data.TaggedError("AuthFailed")<{
  readonly sourceId: string
  readonly status: number
  readonly message: string
}> {}

export class RateLimited extends Data.TaggedError("RateLimited")<{
  readonly sourceId: string
  readonly status: number
  readonly message: string
  readonly retryAfterMs: number | null
}> {}

export class UsageLimitExceeded extends Data.TaggedError("UsageLimitExceeded")<{
  readonly sourceId: string
  readonly status: number
  readonly message: string
}> {}

export class ContextLimitExceeded extends Data.TaggedError("ContextLimitExceeded")<{
  readonly sourceId: string
  readonly status: number
  readonly message: string
}> {}

export class InvalidRequest extends Data.TaggedError("InvalidRequest")<{
  readonly sourceId: string
  readonly status: number
  readonly message: string
}> {}

export class TransportError extends Data.TaggedError("TransportError")<{
  readonly sourceId: string
  readonly status: number | null
  readonly message: string
  readonly retryable: boolean
}> {}

export class ParseError extends Data.TaggedError("ParseError")<{
  readonly sourceId: string
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
