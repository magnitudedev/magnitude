/**
 * Browser-safe SDK entry.
 *
 * Renderer builds resolve `@magnitudedev/sdk` here. ACN ensurance is
 * delegated to the host over HTTP.
 */
export {
  AcnInstanceManager,
  AcnEnsureEventSchema,
  AcnEnsureRequestSchema,
  AcnReadyInstanceSchema,
  RemoteAcnEnsureMessageSchema,
} from "./acn-jit/acn-instance-manager"
export type { AcnEnsureEvent, AcnEnsureRequest, AcnInstance } from "./acn-jit/acn-instance-manager"
export { makeRemoteAcnInstanceManager } from "./acn-jit/remote-acn-instance-manager"
export { formatAcnEnsuranceError } from "./acn-jit/lifecycle"
export {
  makeAcnConnection,
} from "./acn-jit/acn-recovering-client"
export type {
  AcnConnection,
} from "./acn-jit/acn-recovering-client"
export { isRpcOutcomeUnknown } from "./mutation-outcome"
export {
  MagnitudeBoundary,
  magnitudeImplementationsLayer,
  type MagnitudeImplementationError,
} from "./inference"
export {
  AcnEnsuranceFailed,
  AcnAdministrationFailed,
  BinaryNotFound,
  BinaryRevisionMismatch,
  BinaryVersionMismatch,
  DownloadFailed,
  ChecksumMismatch,
  AcnEnsuranceError,
  type StreamDisplayViewFailure,
  type WatchFileFailure,
} from "./errors"

/**
 * The complete ACN boundary: its root group, domain groups, operation
 * declarations (query, mutation, subscription), schemas, and errors.
 */
export * from "@magnitudedev/acn-protocol"
export {
  DisplayState as DisplayStateSchema,
  DisplayViewShape as DisplayViewShapeSchema,
  StreamEvent as StreamEventSchema,
} from "@magnitudedev/acn-protocol"

export {
  ProviderIdSchema,
  ProviderModelIdSchema,
  ReasoningEffortSchema,
  ReasoningProperty,
  VisionProperty,
} from "@magnitudedev/ai/provider/model"

export { normalizeReferencedPath } from "./path-utils"

export {
  isRoleId,
  ROLE_IDS,
  ROLE_TO_SLOT,
  DEFAULT_REASONING_EFFORT,
  SLOT_IDS,
  SLOT_DISPLAY_NAMES,
  SLOT_DESCRIPTIONS,
} from "@magnitudedev/roles/constants"
export type { RoleId } from "@magnitudedev/roles/constants"
