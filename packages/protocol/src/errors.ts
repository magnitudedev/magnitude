import { PlatformError } from "@effect/platform/Error"
import {
  JsonParseError,
  SchemaDecodeError,
  SchemaEncodeError,
} from "@magnitudedev/storage"
import { ErrorResponse } from "@magnitudedev/icn/generated"
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

export class IcnRequestEncodingFailed extends Schema.TaggedError<IcnRequestEncodingFailed>()(
  "IcnRequestEncodingFailed",
  {
    operationId: Schema.String,
    location: Schema.Literal("path", "query", "headers", "payload"),
    message: Schema.String,
  },
) {}

export class IcnTransportFailed extends Schema.TaggedError<IcnTransportFailed>()(
  "IcnTransportFailed",
  { operationId: Schema.String, message: Schema.String },
) {}

export class IcnRemoteRejected extends Schema.TaggedError<IcnRemoteRejected>()(
  "IcnRemoteRejected",
  {
    operationId: Schema.String,
    status: Schema.Number,
    body: ErrorResponse,
  },
) {}

export class IcnInvalidResponse extends Schema.TaggedError<IcnInvalidResponse>()(
  "IcnInvalidResponse",
  {
    operationId: Schema.String,
    status: Schema.Number,
    message: Schema.String,
  },
) {}

export class IcnIncompleteStream extends Schema.TaggedError<IcnIncompleteStream>()(
  "IcnIncompleteStream",
  {
    operationId: Schema.String,
    termination: Schema.Literal("sentinel", "long-lived"),
  },
) {}

export class LocalModelRecipeNotFound extends Schema.TaggedError<LocalModelRecipeNotFound>()(
  "LocalModelRecipeNotFound",
  { configurationId: Schema.String },
) {}

export class LocalInventoryModelNotFound extends Schema.TaggedError<LocalInventoryModelNotFound>()(
  "LocalInventoryModelNotFound",
  { modelId: Schema.String },
) {}

export class LocalConfigurationFailed extends Schema.TaggedError<LocalConfigurationFailed>()(
  "LocalConfigurationFailed",
  {
    failure: Schema.Union(
      PlatformError,
      JsonParseError,
      SchemaDecodeError,
      SchemaEncodeError,
    ),
  },
) {}

export const LocalInferenceError = Schema.Union(
  IcnRequestEncodingFailed,
  IcnTransportFailed,
  IcnRemoteRejected,
  IcnInvalidResponse,
  IcnIncompleteStream,
  LocalModelRecipeNotFound,
  LocalInventoryModelNotFound,
  LocalConfigurationFailed,
)
export type LocalInferenceError = Schema.Schema.Type<typeof LocalInferenceError>

export class OnboardingError extends Schema.TaggedError<OnboardingError>()(
  "OnboardingError",
  {
    operation: Schema.String,
    message: Schema.String,
  },
) {}
