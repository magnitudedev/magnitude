import { Schema } from 'effect'

/**
 * DriverError — the single error type emitted by all Driver implementations.
 *
 * reason  — human-readable description of what went wrong
 * status  — HTTP status code if applicable, null otherwise
 * body    — raw response body or provider error object if available
 */
export class DriverError extends Schema.TaggedError<DriverError>()(
  'DriverError',
  {
    reason: Schema.String,
    status: Schema.NullOr(Schema.Number),
    body:   Schema.Unknown,
  },
) {}
