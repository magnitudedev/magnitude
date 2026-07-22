import { Effect } from "effect"
import {
  DisplayViewNotOpen,
  InvalidSessionPath,
  SessionAlreadyExists,
  SessionNotFound,
  SessionOperationFailed,
  SessionStartFailed,
  type SessionError,
} from "@magnitudedev/protocol"

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
    cause instanceof SessionOperationFailed ||
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
    case "SessionOperationFailed":
      return `${error.operation}: ${error.reason}`
    case "DisplayViewNotOpen":
      return `Display view not open: ${error.viewId}`
    case "InvalidSessionPath":
      return `Invalid session path: ${error.path}`
  }
  return "Session operation failed"
}
