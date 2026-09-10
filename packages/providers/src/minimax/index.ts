export {
  createMiniMaxProvider,
  MINIMAX_ENDPOINTS,
  type MiniMaxClientConfig,
  type MiniMaxProviderInstance,
} from "./provider"
export { createMiniMaxCatalog, MINIMAX_MODELS, MINIMAX_PROVIDER_ID } from "./catalog"
export {
  createMiniMaxCompatibleSpec,
  type MiniMaxCallOptions,
  type MiniMaxCompatibleSpecConfig,
} from "./models"
export { classifyMiniMaxRejectedResponse } from "./errors"
export type {
  MiniMaxAuthentication,
  MiniMaxEndpointConfig,
  MiniMaxModelInfo,
  MiniMaxRegion,
} from "./contract"
