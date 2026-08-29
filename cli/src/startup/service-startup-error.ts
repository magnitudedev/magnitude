import {
  AcnEnsuranceError,
  formatAcnEnsuranceError,
} from "@magnitudedev/sdk"
import { Schema } from "effect"

export const explainError = (error: unknown): string => typeof error === "object"
  && error !== null
  && "reason" in error
  && typeof error.reason === "string"
  ? error.reason
  : error instanceof Error
    ? error.message
    : String(error)

export const explainInteractiveFailure = (error: unknown): string =>
  Schema.is(AcnEnsuranceError)(error)
    ? `Magnitude service failed to start:\n${formatAcnEnsuranceError(error)}`
    : explainError(error)

export const explainServiceStartupFailure = (error: unknown): string =>
  `Magnitude service failed to start:\n${Schema.is(AcnEnsuranceError)(error)
    ? formatAcnEnsuranceError(error)
    : explainError(error)}`
