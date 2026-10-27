import { Effect, Schema } from "effect"
import { OperationFailed } from "@magnitudedev/acn-protocol"

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

export class SessionNotArchived extends Schema.TaggedError<SessionNotArchived>()(
  "SessionNotArchived",
  { sessionId: Schema.String }
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
  OperationFailed,
  SessionNotArchived,
  DisplayViewNotOpen,
  InvalidSessionPath
)
export type SessionError = Schema.Schema.Type<typeof SessionError>

export class InvalidSessionPageCursor extends Schema.TaggedError<InvalidSessionPageCursor>()(
  "InvalidSessionPageCursor",
  {},
) {}

export class InvalidDirectoryPageCursor extends Schema.TaggedError<InvalidDirectoryPageCursor>()(
  "InvalidDirectoryPageCursor",
  {},
) {}

export class SessionMetadataWriteFailed extends Schema.TaggedError<SessionMetadataWriteFailed>()(
  "SessionMetadataWriteFailed",
  { sessionId: Schema.String },
) {}

export class SessionMetadataUnreadable extends Schema.TaggedError<SessionMetadataUnreadable>()(
  "SessionMetadataUnreadable",
  { sessionId: Schema.String },
) {}

export class SessionInspectionUnavailable extends Schema.TaggedError<SessionInspectionUnavailable>()(
  "SessionInspectionUnavailable",
  {},
) {}

export const tryPromiseNull = <T>(fn: () => Promise<T>): Effect.Effect<T | null, never> =>
  Effect.tryPromise(fn).pipe(
    Effect.orElseSucceed(() => null),
  )

const stringifyCauseObject = (cause: object): string => {
  try {
    return JSON.stringify(cause, (_key, value) => {
      if (value instanceof Error) {
        return {
          name: value.name,
          message: value.message,
        }
      }
      return value
    })
  } catch {
    return String(cause)
  }
}

export function formatUnknownCause(cause: unknown): string {
  if (cause && typeof cause === "object") {
    const tagged = cause as { readonly _tag?: unknown }
    if (typeof tagged._tag === "string") {
      return stringifyCauseObject(cause)
    }
    if (cause instanceof Error && cause.message.length > 0) {
      return cause.message
    }
    return stringifyCauseObject(cause)
  }

  if (cause === null || cause === undefined) {
    return "Session operation failed"
  }

  const message = String(cause)
  return message.length > 0 ? message : "Session operation failed"
}

export function toSessionError(sessionId: string, cause: unknown): SessionError {
  if (
    cause instanceof SessionNotFound ||
    cause instanceof SessionAlreadyExists ||
    cause instanceof SessionStartFailed ||
    cause instanceof OperationFailed ||
    cause instanceof DisplayViewNotOpen ||
    cause instanceof InvalidSessionPath
  ) {
    return cause
  }
  return new SessionStartFailed({
    sessionId,
    reason: formatUnknownCause(cause),
  })
}

export function sessionErrorMessage(error: SessionError): string {
  switch (error._tag) {
    case "SessionNotFound":
      return `Session not found: ${error.sessionId}`
    case "SessionAlreadyExists":
      return `Session already exists: ${error.sessionId}`
    case "SessionStartFailed":
      return error.reason
    case "OperationFailed":
      return `${error.operation}: ${error.reason}`
    case "DisplayViewNotOpen":
      return `Display view not open: ${error.viewId}`
    case "InvalidSessionPath":
      return `Invalid session path: ${error.path}`
  }
  return "Session operation failed"
}
