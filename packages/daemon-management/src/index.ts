export { makeServiceStarter } from "./service-starter"
export {
  AcnInstanceManager,
  AcnEnsureRequestSchema,
  AcnEnsureEventSchema,
  AcnReadyInstanceSchema,
  runAcnEnsure,
  type AcnEnsureRequest,
  type AcnEnsureEvent,
  type AcnInstance,
} from "./acn-jit/acn-instance-manager"

export {
  makeLocalAcnInstanceManager,
  type LocalAcnInstanceManagerOptions,
  type AcnLaunchOverride,
} from "./acn-jit/local-acn-instance-manager"

export {
  ChildProcessSpawner,
  scopeAcnCandidate,
  type SpawnedAcnCandidate,
} from "./acn-jit/child-process"

export { BunDetachedChildProcessSpawner } from "./acn-jit/bun-spawn-process"

export { formatAcnEnsuranceError } from "./acn-jit/format-error"

export { acnInstallationPresent, resolveBinaryCommand, defaultBinaryPath, defaultDataDir, type BinaryAcquisitionEvent, type ResolveBinaryOptions, type ResolvedBinaryCommand } from "./binary"

export { DAEMON_VERSION, DAEMON_REVISION, DAEMON_TARGET } from "./version"

export {
  AcnAdministrationFailed,
  AcnOwnerRecordReadUnavailable,
  AcnOwnerRecordInvalid,
  AcnProcessIdentityObservationTimedOut,
  AcnCandidateSpawnFailed,
  AcnCandidateIdentityUnavailable,
  AcnCandidateExitedBeforeAdmission,
  AcnCandidateExitedAfterAdmission,
  AcnCandidateAdmissionTimedOut,
  AcnCandidateParentChannelReleaseFailed,
  AcnCandidateOwnershipLost,
  AcnCandidateFailureSchema,
  type AcnCandidateFailure,
  AcnCandidateBootstrapProcessStopFailed,
  AcnCandidateBootstrapProcessExitUnproven,
  AcnDaemonTargetUnsupported,
  AcnLaunchOverrideTargetMismatch,
  AcnDaemonShutdownFailed,
  AcnDaemonShutdownControlFailureSchema,
  type AcnDaemonShutdownControlFailure,
  AcnDaemonStartupTimedOut,
  AcnEnsuranceConvergenceTimedOut,
  AcnEnsuranceFailed,
  AcnHealthRequestFailed,
  AcnHealthAttemptTimedOut,
  AcnHealthResponseInvalid,
  AcnHealthAttemptFailureSchema,
  type AcnHealthAttemptFailure,
  AcnHealthUnavailable,
  BinaryNotFound,
  BinaryRevisionMismatch,
  BinaryVersionMismatch,
  DownloadFailed,
  ChecksumMismatch,
  AcnEnsuranceError,
} from "./errors"
export { ACN_EXECUTABLE_NAME } from "@magnitudedev/release/executables"
