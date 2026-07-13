/**
 * Stream error classification — shared between CLI and web.
 *
 * Moved from CLI's `app.tsx`. Both apps consume `StreamErrorInfo` via
 * `StreamCallbacks.onError`. The CLI's `FatalErrorScreen` and the web's
 * `DaemonConnectionError` both read `invariantViolation` from the info.
 */
import { Cause, Chunk } from "effect"
import {
  BinaryNotFound,
  BinaryVersionMismatch,
  DaemonCrashed,
  DaemonSpawnFailed,
  DownloadFailed,
  type StreamDisplayViewFailure,
} from "@magnitudedev/sdk"

/**
 * Structured stream error info consumed by both apps.
 */
export interface StreamErrorInfo {
  readonly message: string
  readonly invariantViolation: boolean
  readonly isDaemonResolutionError: boolean
}

type DaemonResolutionError =
  | BinaryNotFound
  | BinaryVersionMismatch
  | DaemonSpawnFailed
  | DaemonCrashed
  | DownloadFailed

export function isDaemonResolutionError(error: unknown): error is DaemonResolutionError {
  return (
    error instanceof BinaryNotFound ||
    error instanceof BinaryVersionMismatch ||
    error instanceof DaemonSpawnFailed ||
    error instanceof DaemonCrashed ||
    error instanceof DownloadFailed
  )
}

export function daemonErrorMessage(error: DaemonResolutionError): string {
  if (error instanceof BinaryNotFound) {
    return "Magnitude daemon is missing. Please restart Magnitude to reinstall it."
  }
  if (error instanceof BinaryVersionMismatch) {
    return `Magnitude daemon version does not match this client. Expected ${error.expected}, got ${error.actual}.`
  }
  if (error instanceof DaemonCrashed) {
    return `Magnitude daemon crashed on startup (exit code ${error.exitCode}).`
  }
  if (error instanceof DownloadFailed) {
    return `Failed to download the Magnitude daemon: ${error.reason}`
  }
  return `Magnitude daemon failed to start: ${error.reason}`
}

function caughtErrorDetails(error: unknown): string {
  if (error instanceof Error) {
    return error.stack ?? error.message
  }
  return `Unexpected non-Error thrown value: ${String(error)}`
}

export function classifyStartupError(error: unknown): StreamErrorInfo {
  if (isDaemonResolutionError(error)) {
    return {
      message: daemonErrorMessage(error),
      invariantViolation: false,
      isDaemonResolutionError: true,
    }
  }
  return {
    message: caughtErrorDetails(error),
    invariantViolation: true,
    isDaemonResolutionError: false,
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
 * unavailability (RpcClientError, resolution error in `cause`) or a domain
 * error like the session disappearing — the latter is an invariant violation.
 */
export function classifyStreamError(cause: Cause.Cause<StreamDisplayViewFailure>): StreamErrorInfo {
  for (const failure of Chunk.toReadonlyArray(Cause.failures(cause))) {
    if (isDaemonResolutionError(failure)) {
      return {
        message: daemonErrorMessage(failure),
        invariantViolation: false,
        isDaemonResolutionError: true,
      }
    }
    if (failure._tag === "RpcClientError") {
      return {
        message: `Lost connection to the Magnitude daemon and could not recover.\n\n${formatStreamFailure(failure)}`,
        invariantViolation: false,
        isDaemonResolutionError: false,
      }
    }
  }
  return {
    message: fullStreamErrorDetails(cause),
    invariantViolation: true,
    isDaemonResolutionError: false,
  }
}
