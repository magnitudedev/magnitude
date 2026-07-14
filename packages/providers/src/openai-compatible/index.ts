export { createOpenAiCompatibleCatalog } from "./catalog"
export { classifyOpenAiCompatibleRejectedResponse } from "./errors"
export {
  createOpenAiCompatibleSpec,
  composeReasoningRequest,
  wrapOpenAiCompatibleAsBaseModel,
  type OpenAiCompatibleCallOptions,
  type OpenAiCompatibleSpecConfig,
  type ReasoningRequestMode,
} from "./models"
export type {
  OpenAiCompatibleCatalogConfig,
  OpenAiCompatibleModelInfo,
  OpenAiCompatibleModelsResponse,
  OpenAiCompatibleProviderInstance,
  OpenAiCompatibleRawModel,
} from "./contract"
