import { Cause, Match, ParseResult } from "effect"
import type { PlatformError } from "@effect/platform/Error"
import type {
  GeneratedClientError,
  Generated,
} from "@magnitudedev/icn"
import type {
  JsonParseError,
  SchemaDecodeError,
  SchemaEncodeError,
} from "@magnitudedev/storage"
import {
  IcnIncompleteStream,
  IcnInvalidResponse,
  IcnRemoteRejected,
  IcnRequestEncodingFailed,
  IcnTransportFailed,
  LocalConfigurationFailed,
  LocalInventoryModelNotFound,
  LocalModelRecipeNotFound,
  type LocalInferenceError,
} from "@magnitudedev/protocol"
import { InventoryModelNotFound, ModelRecipeNotFound } from "./models"

export type LocalInferenceInternalError =
  | GeneratedClientError<Generated.ErrorResponse>
  | ModelRecipeNotFound
  | InventoryModelNotFound
  | PlatformError
  | JsonParseError
  | SchemaDecodeError
  | SchemaEncodeError

export const localInferenceRpcError = (
  error: LocalInferenceInternalError,
): LocalInferenceError => Match.value(error).pipe(
  Match.tag("GeneratedClientInputError", (input) => new IcnRequestEncodingFailed({
    operationId: input.operationId,
    location: input.location,
    message: ParseResult.TreeFormatter.formatErrorSync(input.cause),
  })),
  Match.tag("GeneratedClientTransportError", (transport) => new IcnTransportFailed({
    operationId: transport.operationId,
    message: Cause.pretty(Cause.fail(transport.cause)),
  })),
  Match.tag("GeneratedClientRemoteError", (remote) => new IcnRemoteRejected({
    operationId: remote.operationId,
    status: remote.status,
    body: remote.body,
  })),
  Match.tag("GeneratedClientInvalidResponseError", (invalid) => new IcnInvalidResponse({
    operationId: invalid.operationId,
    status: invalid.status,
    message: invalid.message,
  })),
  Match.tag("GeneratedClientIncompleteStreamError", (incomplete) => new IcnIncompleteStream({
    operationId: incomplete.operationId,
    termination: incomplete.termination,
  })),
  Match.tag("ModelRecipeNotFound", (missing) => new LocalModelRecipeNotFound({
    configurationId: missing.configurationId,
  })),
  Match.tag("InventoryModelNotFound", (missing) => new LocalInventoryModelNotFound({
    modelId: missing.modelId,
  })),
  Match.tags({
    BadArgument: (failure) => new LocalConfigurationFailed({ failure }),
    SystemError: (failure) => new LocalConfigurationFailed({ failure }),
    JsonParseError: (failure) => new LocalConfigurationFailed({ failure }),
    SchemaDecodeError: (failure) => new LocalConfigurationFailed({ failure }),
    SchemaEncodeError: (failure) => new LocalConfigurationFailed({ failure }),
  }),
  Match.exhaustive,
)
