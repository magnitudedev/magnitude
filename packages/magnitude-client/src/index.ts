export * from "./contract"
export { createModelCatalog, type ModelCatalog, type ModelCatalogConfig } from "./catalog"
export {
  SubscriptionRequired,
  TrialExpired,
  MagnitudeUsageLimitExceeded,
  ModelNotGrammarCompatible,
  RoleNotFound,
  tryParseErrorBody,
  classifyMagnitudeConnectionError,
  type MagnitudeConnectionError,
} from "./errors"
export { createRoleSpec, createMagnitudeCompatibleSpec, toModelProfile, type MagnitudeModelSpec, type MagnitudeStreamError, type MagnitudeCompatibleSpecConfig, type ModelProfile } from "./models"
export { createMagnitudeClient, type MagnitudeClientConfig } from "./client"
