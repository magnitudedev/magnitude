import { Schema } from "effect"
import { ConnectionErrorSchema, formatConnectionError } from "@magnitudedev/sdk"

export const explainError = (error: unknown): string => Schema.is(ConnectionErrorSchema)(error)
  ? formatConnectionError(error)
  : typeof error === "object"
  && error !== null
  && "reason" in error
  && typeof error.reason === "string"
  ? error.reason
  : error instanceof Error
    ? error.message
    : String(error)

export const explainServiceStartupFailure = (error: unknown): string =>
  `Magnitude service failed to start:\n${explainError(error)}`
