// Barrel export for @magnitudedev/client-common

// State
export * from './connection/connection'
export {
  ServiceLifecycleStateSchema,
  type ServiceLifecycleState,
  type ServiceLifecycle,
  type ServiceStartingPhase,
} from './connection/lifecycle'
export * from './state/agent-client'
export * from './state/agent-client-context'

// Utils
export * from './utils/format-bytes'
export * from './utils/palette'
export * from './utils/model-presentation'
export * from './utils/model-load'
export * from './utils/local-model-radar'
export * from './utils/hardware-memory'

// Local models and connections
export * from './local-models/projection'
export * from './local-models/options'
export * from './local-models/failure-messages'
export * from './local-models/service'
export * from './files/service'
export * from './project-files/service'
export * from './harness-connections/service'

// Hooks
export * from './hooks/use-local-inference-state'

// Desktop
export * from "./desktop/service"
export * from "./desktop/update"
export { DesktopConnectRequest, ConnectionInspection, DesktopHarnessConnection, DesktopConnectionsSnapshot } from "./desktop/connections"
