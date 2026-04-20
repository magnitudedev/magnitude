import { Data } from 'effect'

export class NotConfigured extends Data.TaggedError('NotConfigured')<{
  readonly message: string
}> {}

export class ProviderDisconnected extends Data.TaggedError('ProviderDisconnected')<{
  readonly providerId: string
  readonly providerName: string
  readonly message: string
}> {}

export class AuthFailed extends Data.TaggedError('AuthFailed')<{
  readonly message: string
}> {}

export class ContextLimitExceeded extends Data.TaggedError('ContextLimitExceeded')<{
  readonly message: string
}> {}

export class RateLimited extends Data.TaggedError('RateLimited')<{
  readonly message: string
  readonly retryAfterMs: number | null
}> {}

export class TransportError extends Data.TaggedError('TransportError')<{
  readonly message: string
  readonly status: number | null
}> {}

export class ParseError extends Data.TaggedError('ParseError')<{
  readonly message: string
  readonly raw: unknown
}> {}

export class SubscriptionRequired extends Data.TaggedError('SubscriptionRequired')<{
  readonly message: string
  readonly code: string
}> {}

export class UsageLimitExceeded extends Data.TaggedError('UsageLimitExceeded')<{
  readonly message: string
  readonly code: string
}> {}

export type ModelError =
  | NotConfigured
  | ProviderDisconnected
  | AuthFailed
  | ContextLimitExceeded
  | RateLimited
  | TransportError
  | ParseError
  | SubscriptionRequired
  | UsageLimitExceeded