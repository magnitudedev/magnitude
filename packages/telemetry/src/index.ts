export { initTelemetry, shutdownTelemetry, capture, isTelemetryEnabled } from './client'
export type { TelemetryInitOptions } from './client'
export { getOrCreateMachineId } from './machine-id'
export {
  trackSessionStart,
  trackSessionEnd,
  trackUserMessage,
  trackTurnCompleted,
  trackToolUsage,
  trackAgentSpawned,
  trackAgentCompleted,
  trackProviderConnected,
  trackCompaction,
} from './events'
export { SessionTracker } from './session-tracker'
