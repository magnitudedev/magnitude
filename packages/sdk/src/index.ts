export { MagnitudeClient, ConnectionStateSchema, ServiceInfoSchema, type MagnitudeClientError, type MagnitudeConnection, type ConnectionState, type ServiceInfo } from "./client"
export { MagnitudeServiceStarter } from "./service-starter"
export * from "./connection-errors"
export * from "./inference-progress"
export {
  makeInferenceClient,
  InferenceModelSchema,
  InferenceModelsResponseSchema,
  ResponseObjectSchema,
  type InferenceClient,
  type InferenceModel,
  type InferenceModelsResponse,
  type ResponseObject,
} from "./inference-client"

export type { ModelInstancesSnapshot as InferenceInstancesSnapshot } from "@magnitudedev/icn-protocol/schemas"


/**
 * The RPC declaration tree, wire schemas, and operation errors.
 */
export * from "@magnitudedev/acn-protocol"
export {
  DisplayState as DisplayStateSchema,
  DisplayTimeline as DisplayTimelineSchema,
  DisplayViewShape as DisplayViewShapeSchema,
  StreamEvent as StreamEventSchema,
} from "@magnitudedev/acn-protocol"

export { isRoleId, ROLE_IDS, ROLE_TO_SLOT, DEFAULT_REASONING_EFFORT, SLOT_IDS, SLOT_DISPLAY_NAMES, SLOT_DESCRIPTIONS } from "@magnitudedev/roles/constants"
export type { RoleId } from "@magnitudedev/roles/constants"

export { isEnvFlagOn } from "@magnitudedev/utils"
export { normalizeReferencedPath } from "./path-utils"
export { isRpcOutcomeUnknown } from "./mutation-outcome"

export {
  type ToolCallId,
  type ChatCompletionsStreamChunk,
  type StreamFailure,
  type StreamStartFailure,
  type Prompt,
  type ToolDefinition,
} from "@magnitudedev/ai"
export * from "@magnitudedev/ai/provider/model"
// Select the wire contract where the AI model module exports the same names.
export { ProviderModelDisabledReasonSchema, type ProviderModelDisabledReason } from "@magnitudedev/acn-protocol"
