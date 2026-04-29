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
export { createRoleSpec } from "./models"
export { createMagnitudeClient, type MagnitudeClientConfig } from "./client"
