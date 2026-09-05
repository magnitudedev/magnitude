/**
 * Stream error classification — shared between CLI and web.
 *
 * Moved from CLI's `app.tsx`. Both apps consume `StreamErrorInfo` via
 * `StreamCallbacks.onError`. The CLI's `FatalErrorScreen` and the web's
 * `DaemonConnectionError` both read `invariantViolation` from the info.
 */
import { Cause, Chunk, Schema, type Stream } from "effect"
import { ConnectionErrorSchema, formatConnectionError, type ConnectionError, type MagnitudeClient } from "@magnitudedev/sdk"

export type StreamDisplayViewFailure = Stream.Stream.Error<ReturnType<MagnitudeClient["display"]["streamDisplayView"]>>

/**
 * Structured stream error info consumed by both apps.
 */
export interface StreamErrorInfo {
  readonly message: string
  readonly invariantViolation: boolean
  readonly isAcnAvailabilityError: boolean
}

type AcnAvailabilityError = ConnectionError

export function isAcnAvailabilityError(error: unknown): error is AcnAvailabilityError {
  return Schema.is(ConnectionErrorSchema)(error)
}

export function acnAvailabilityErrorMessage(error: AcnAvailabilityError): string {
  return `Magnitude service unavailable:\n${formatConnectionError(error)}`
}

function caughtErrorDetails(error: unknown): string {
  if (error instanceof Error) {
    return error.stack ?? error.message
  }
  return `Unexpected non-Error thrown value: ${String(error)}`
}

export function classifyStartupError(error: unknown): StreamErrorInfo {
  if (isAcnAvailabilityError(error)) {
    return {
      message: acnAvailabilityErrorMessage(error),
      invariantViolation: false,
      isAcnAvailabilityError: true,
    }
  }
  return {
    message: caughtErrorDetails(error),
    invariantViolation: true,
    isAcnAvailabilityError: false,
  }
}

function formatDefect(defect: unknown): string {
  return defect instanceof Error ? (defect.stack ?? defect.message) : String(defect)
}

function formatStreamFailure(failure: StreamDisplayViewFailure): string {
  if (failure._tag === "RpcClientError") {
    return [
      `RpcClientError(${failure.reason}): ${failure.message}`,
      failure.cause !== undefined ? `cause: ${formatDefect(failure.cause)}` : null,
    ]
      .filter((part): part is string => part !== null)
      .join("\n")
  }
  return JSON.stringify(failure, null, 2) ?? String(failure)
}

function fullStreamErrorDetails(cause: Cause.Cause<StreamDisplayViewFailure>): string {
  const failures = Chunk.toReadonlyArray(Cause.failures(cause)).map(formatStreamFailure)
  const defects = Chunk.toReadonlyArray(Cause.defects(cause)).map(formatDefect)
  return [
    "StreamDisplayView failed",
    failures.length > 0 ? `failures:\n${failures.join("\n\n")}` : null,
    defects.length > 0 ? `defects:\n${defects.join("\n\n")}` : null,
    `effect cause:\n${Cause.pretty(cause)}`,
  ]
    .filter((part): part is string => part !== null)
    .join("\n\n")
}

/**
 * Classify a StreamDisplayView failure cause into structured error info.
 *
 * Display streams recover from daemon deaths transparently (SDK operation
 * contract), so an error landing here is terminal: either fatal daemon
 * unavailability (RpcClientError, ensurance error in `cause`) or a domain
 * error like the session disappearing — the latter is an invariant violation.
 */
export function classifyStreamError(cause: Cause.Cause<StreamDisplayViewFailure>): StreamErrorInfo {
  for (const failure of Chunk.toReadonlyArray(Cause.failures(cause))) {
    if (isAcnAvailabilityError(failure)) {
      return {
        message: acnAvailabilityErrorMessage(failure),
        invariantViolation: false,
        isAcnAvailabilityError: true,
      }
    }
    if (failure._tag === "RpcClientError") {
      if (isAcnAvailabilityError(failure.cause)) {
        return {
          message: acnAvailabilityErrorMessage(failure.cause),
          invariantViolation: false,
          isAcnAvailabilityError: true,
        }
      }
      return {
        message: `Lost connection to the Magnitude service and could not recover.\n\n${formatStreamFailure(failure)}`,
        invariantViolation: false,
        isAcnAvailabilityError: false,
      }
    }
  }
  return {
    message: fullStreamErrorDetails(cause),
    invariantViolation: true,
    isAcnAvailabilityError: false,
  }
}
