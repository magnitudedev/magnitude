import { Schema } from "effect"

export class SessionNotFound extends Schema.TaggedError<SessionNotFound>()(
  "SessionNotFound",
  { sessionId: Schema.String }
) {}

export class SessionAlreadyExists extends Schema.TaggedError<SessionAlreadyExists>()(
  "SessionAlreadyExists",
  { sessionId: Schema.String }
) {}

export class SessionStartFailed extends Schema.TaggedError<SessionStartFailed>()(
  "SessionStartFailed",
  { sessionId: Schema.String, reason: Schema.String }
) {}

export class SessionOperationFailed extends Schema.TaggedError<SessionOperationFailed>()(
  "SessionOperationFailed",
  { operation: Schema.String, reason: Schema.String }
) {}

export class DisplayViewNotOpen extends Schema.TaggedError<DisplayViewNotOpen>()(
  "DisplayViewNotOpen",
  { sessionId: Schema.String, viewId: Schema.String }
) {}

export class InvalidSessionPath extends Schema.TaggedError<InvalidSessionPath>()(
  "InvalidSessionPath",
  { path: Schema.String }
) {}

export const SessionError = Schema.Union(
  SessionNotFound,
  SessionAlreadyExists,
  SessionStartFailed,
  SessionOperationFailed,
  DisplayViewNotOpen,
  InvalidSessionPath
)
export type SessionError = Schema.Schema.Type<typeof SessionError>

export class LocalInferenceError extends Schema.TaggedError<LocalInferenceError>()(
  "LocalInferenceError",
  {
    code: Schema.Literal(
      "distribution_missing",
      "unsupported_platform",
      "invalid_selection",
      "artifact_unavailable",
      "license_required",
      "insufficient_disk_space",
      "integrity_failed",
      "operation_conflict",
      "operation_not_found",
      "artifact_not_owned",
      "artifact_active",
      "context_mismatch",
      "server_start_failed",
      "external_server_unavailable",
      "configuration_failed",
      "runtime_probe_failed",
      "cancelled",
    ),
    operation: Schema.String,
    message: Schema.String,
    retryable: Schema.Boolean,
  },
) {}

export class OnboardingError extends Schema.TaggedError<OnboardingError>()(
  "OnboardingError",
  {
    operation: Schema.String,
    message: Schema.String,
  },
) {}
