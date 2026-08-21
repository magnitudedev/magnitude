import { Schema } from "effect"
import { Rpc, RpcClientError } from "@effect/rpc"
import { AcnTargetSchema, StreamDisplayView, WatchFile } from "@magnitudedev/acn-protocol"
import {
  AcnOwnerRecordSchema,
  ExactProcessSchema,
  ExactProcessIdentityObservationFailed,
  ProcessGroupObservationFailed,
} from "@magnitudedev/acn-protocol/coordination"

export class AcnEnsuranceFailed extends Schema.TaggedError<AcnEnsuranceFailed>()(
  "AcnEnsuranceFailed",
  { reason: Schema.String.pipe(Schema.minLength(1)) },
) {}

export class AcnOwnerRecordReadUnavailable extends Schema.TaggedError<AcnOwnerRecordReadUnavailable>()(
  "AcnOwnerRecordReadUnavailable",
  { path: Schema.String, message: Schema.String },
) {}

export class AcnOwnerRecordInvalid extends Schema.TaggedError<AcnOwnerRecordInvalid>()(
  "AcnOwnerRecordInvalid",
  { path: Schema.String, message: Schema.String },
) {}

export class AcnProcessIdentityObservationTimedOut extends Schema.TaggedError<AcnProcessIdentityObservationTimedOut>()(
  "AcnProcessIdentityObservationTimedOut",
  { pid: Schema.Number.pipe(Schema.int(), Schema.positive()) },
) {}

export class AcnCandidateSpawnFailed extends Schema.TaggedError<AcnCandidateSpawnFailed>()(
  "AcnCandidateSpawnFailed",
  { message: Schema.String },
) {}

export class AcnCandidateProcessGroupObservationFailed extends Schema.TaggedError<AcnCandidateProcessGroupObservationFailed>()(
  "AcnCandidateProcessGroupObservationFailed",
  { pid: Schema.Number.pipe(Schema.int(), Schema.positive()), message: Schema.String },
) {}

export class AcnCandidateProcessGroupTerminationFailed extends Schema.TaggedError<AcnCandidateProcessGroupTerminationFailed>()(
  "AcnCandidateProcessGroupTerminationFailed",
  { pid: Schema.Number.pipe(Schema.int(), Schema.positive()), message: Schema.String },
) {}

export class AcnCandidateProcessGroupTerminationPermissionDenied extends Schema.TaggedError<AcnCandidateProcessGroupTerminationPermissionDenied>()(
  "AcnCandidateProcessGroupTerminationPermissionDenied",
  { pid: Schema.Number.pipe(Schema.int(), Schema.positive()), message: Schema.String },
) {}

export class AcnCandidateProcessGroupKillFailed extends Schema.TaggedError<AcnCandidateProcessGroupKillFailed>()(
  "AcnCandidateProcessGroupKillFailed",
  { pid: Schema.Number.pipe(Schema.int(), Schema.positive()), message: Schema.String },
) {}

export class AcnCandidateProcessGroupKillPermissionDenied extends Schema.TaggedError<AcnCandidateProcessGroupKillPermissionDenied>()(
  "AcnCandidateProcessGroupKillPermissionDenied",
  { pid: Schema.Number.pipe(Schema.int(), Schema.positive()), message: Schema.String },
) {}

export class AcnCandidateProcessGroupAbsenceUnproven extends Schema.TaggedError<AcnCandidateProcessGroupAbsenceUnproven>()(
  "AcnCandidateProcessGroupAbsenceUnproven",
  { pid: Schema.Number.pipe(Schema.int(), Schema.positive()) },
) {}

export class AcnCandidateProcessGroupLeaderChanged extends Schema.TaggedError<AcnCandidateProcessGroupLeaderChanged>()(
  "AcnCandidateProcessGroupLeaderChanged",
  { candidate: ExactProcessSchema, observedLeader: ExactProcessSchema },
) {}

export class AcnCandidateBootstrapProcessStopFailed extends Schema.TaggedError<AcnCandidateBootstrapProcessStopFailed>()(
  "AcnCandidateBootstrapProcessStopFailed",
  { pid: Schema.Number.pipe(Schema.int(), Schema.positive()), message: Schema.String },
) {}

export class AcnCandidateBootstrapProcessExitUnproven extends Schema.TaggedError<AcnCandidateBootstrapProcessExitUnproven>()(
  "AcnCandidateBootstrapProcessExitUnproven",
  { pid: Schema.Number.pipe(Schema.int(), Schema.positive()) },
) {}

export class AcnCandidateExactProcessPidMismatch extends Schema.TaggedError<AcnCandidateExactProcessPidMismatch>()(
  "AcnCandidateExactProcessPidMismatch",
  {
    candidatePid: Schema.Number.pipe(Schema.int(), Schema.positive()),
    observed: ExactProcessSchema,
  },
) {}

export class AcnCandidateExactProcessAlreadyConfirmed extends Schema.TaggedError<AcnCandidateExactProcessAlreadyConfirmed>()(
  "AcnCandidateExactProcessAlreadyConfirmed",
  { confirmed: ExactProcessSchema, attempted: ExactProcessSchema },
) {}

export class AcnCandidateAdmissionBeforeExactProcessConfirmed extends Schema.TaggedError<AcnCandidateAdmissionBeforeExactProcessConfirmed>()(
  "AcnCandidateAdmissionBeforeExactProcessConfirmed",
  { pid: Schema.Number.pipe(Schema.int(), Schema.positive()) },
) {}

export class AcnCandidateParentChannelReleaseFailed extends Schema.TaggedError<AcnCandidateParentChannelReleaseFailed>()(
  "AcnCandidateParentChannelReleaseFailed",
  { pid: Schema.Number.pipe(Schema.int(), Schema.positive()), message: Schema.String },
) {}

export class AcnCandidateAdmissionAlreadyAcknowledged extends Schema.TaggedError<AcnCandidateAdmissionAlreadyAcknowledged>()(
  "AcnCandidateAdmissionAlreadyAcknowledged",
  { pid: Schema.Number.pipe(Schema.int(), Schema.positive()) },
) {}

export class AcnCandidateLaunchAlreadyAttempted extends Schema.TaggedError<AcnCandidateLaunchAlreadyAttempted>()(
  "AcnCandidateLaunchAlreadyAttempted",
  {},
) {}

export class AcnCandidateIdentityUnavailable extends Schema.TaggedError<AcnCandidateIdentityUnavailable>()(
  "AcnCandidateIdentityUnavailable",
  { pid: Schema.Number.pipe(Schema.int(), Schema.positive()) },
) {}

export class AcnCandidateExitedBeforeIdentityObserved extends Schema.TaggedError<AcnCandidateExitedBeforeIdentityObserved>()(
  "AcnCandidateExitedBeforeIdentityObserved",
  {
    pid: Schema.Number.pipe(Schema.int(), Schema.positive()),
    code: Schema.Number,
    stderr: Schema.String,
  },
) {}

export class AcnCandidateAdmissionAcknowledgementTimedOut extends Schema.TaggedError<AcnCandidateAdmissionAcknowledgementTimedOut>()(
  "AcnCandidateAdmissionAcknowledgementTimedOut",
  { pid: Schema.Number.pipe(Schema.int(), Schema.positive()) },
) {}

export class AcnCandidateReadyBeforeAdmission extends Schema.TaggedError<AcnCandidateReadyBeforeAdmission>()(
  "AcnCandidateReadyBeforeAdmission",
  {},
) {}

export class AcnCandidateReadyInstanceMismatch extends Schema.TaggedError<AcnCandidateReadyInstanceMismatch>()(
  "AcnCandidateReadyInstanceMismatch",
  { candidate: ExactProcessSchema, ready: ExactProcessSchema },
) {}

export class AcnDaemonTargetUnsupported extends Schema.TaggedError<AcnDaemonTargetUnsupported>()(
  "AcnDaemonTargetUnsupported",
  { requested: AcnTargetSchema, supported: AcnTargetSchema },
) {}

export class AcnLaunchOverrideTargetMismatch extends Schema.TaggedError<AcnLaunchOverrideTargetMismatch>()(
  "AcnLaunchOverrideTargetMismatch",
  { requested: AcnTargetSchema, override: AcnTargetSchema },
) {}

const CandidateExitFailureFields = {
  process: ExactProcessSchema,
  code: Schema.Number,
  stderr: Schema.String,
}
export class AcnCandidateExitedBeforeAdmissionFailure extends Schema.TaggedError<AcnCandidateExitedBeforeAdmissionFailure>()(
  "AcnCandidateExitedBeforeAdmissionFailure",
  CandidateExitFailureFields,
) {}

export class AcnCandidateExitedAfterAdmissionFailure extends Schema.TaggedError<AcnCandidateExitedAfterAdmissionFailure>()(
  "AcnCandidateExitedAfterAdmissionFailure",
  CandidateExitFailureFields,
) {}

export class AcnCandidateAdmissionTimedOut extends Schema.TaggedError<AcnCandidateAdmissionTimedOut>()(
  "AcnCandidateAdmissionTimedOut",
  { process: ExactProcessSchema },
) {}

export class AcnCandidateAdmissionLost extends Schema.TaggedError<AcnCandidateAdmissionLost>()(
  "AcnCandidateAdmissionLost",
  { process: ExactProcessSchema },
) {}

export class AcnCandidateOwnershipLostAfterAdmission extends Schema.TaggedError<AcnCandidateOwnershipLostAfterAdmission>()(
  "AcnCandidateOwnershipLostAfterAdmission",
  { process: ExactProcessSchema },
) {}

export class AcnDaemonStartupTimedOut extends Schema.TaggedError<AcnDaemonStartupTimedOut>()(
  "AcnDaemonStartupTimedOut",
  { owner: AcnOwnerRecordSchema },
) {}

export class AcnEnsuranceConvergenceTimedOut extends Schema.TaggedError<AcnEnsuranceConvergenceTimedOut>()(
  "AcnEnsuranceConvergenceTimedOut",
  {},
) {}

export const AcnDaemonShutdownReasonSchema = Schema.Literal(
  "InvalidHealth",
  "RevisionTooOld",
  "HealthUnavailable",
  "StoppingExpired",
  "StartupExpired",
  "SurvivingProcessGroup",
  "AdministrativeStop",
)
export type AcnDaemonShutdownReason = typeof AcnDaemonShutdownReasonSchema.Type

export class AcnDaemonOwnerObservationFailed extends Schema.TaggedError<AcnDaemonOwnerObservationFailed>()(
  "AcnDaemonOwnerObservationFailed",
  {
    reason: Schema.String.pipe(Schema.minLength(1)),
    expectedOwner: AcnOwnerRecordSchema,
    shutdownReason: AcnDaemonShutdownReasonSchema,
    path: Schema.String,
    message: Schema.String,
  },
) {}

export class AcnDaemonIdentityObservationFailed extends Schema.TaggedError<AcnDaemonIdentityObservationFailed>()(
  "AcnDaemonIdentityObservationFailed",
  {
    reason: Schema.String.pipe(Schema.minLength(1)),
    expectedOwner: AcnOwnerRecordSchema,
    shutdownReason: AcnDaemonShutdownReasonSchema,
    message: Schema.String,
  },
) {}

export class AcnDaemonProcessGroupObservationFailed extends Schema.TaggedError<AcnDaemonProcessGroupObservationFailed>()(
  "AcnDaemonProcessGroupObservationFailed",
  {
    reason: Schema.String.pipe(Schema.minLength(1)),
    expectedOwner: AcnOwnerRecordSchema,
    shutdownReason: AcnDaemonShutdownReasonSchema,
    message: Schema.String,
  },
) {}

export class AcnDaemonProcessGroupTerminationPermissionDenied extends Schema.TaggedError<AcnDaemonProcessGroupTerminationPermissionDenied>()(
  "AcnDaemonProcessGroupTerminationPermissionDenied",
  {
    reason: Schema.String.pipe(Schema.minLength(1)),
    expectedOwner: AcnOwnerRecordSchema,
    shutdownReason: AcnDaemonShutdownReasonSchema,
    message: Schema.String,
  },
) {}

export class AcnDaemonProcessGroupTerminationFailed extends Schema.TaggedError<AcnDaemonProcessGroupTerminationFailed>()(
  "AcnDaemonProcessGroupTerminationFailed",
  {
    reason: Schema.String.pipe(Schema.minLength(1)),
    expectedOwner: AcnOwnerRecordSchema,
    shutdownReason: AcnDaemonShutdownReasonSchema,
    message: Schema.String,
  },
) {}

export class AcnDaemonProcessGroupKillPermissionDenied extends Schema.TaggedError<AcnDaemonProcessGroupKillPermissionDenied>()(
  "AcnDaemonProcessGroupKillPermissionDenied",
  {
    reason: Schema.String.pipe(Schema.minLength(1)),
    expectedOwner: AcnOwnerRecordSchema,
    shutdownReason: AcnDaemonShutdownReasonSchema,
    message: Schema.String,
  },
) {}

export class AcnDaemonProcessGroupKillFailed extends Schema.TaggedError<AcnDaemonProcessGroupKillFailed>()(
  "AcnDaemonProcessGroupKillFailed",
  {
    reason: Schema.String.pipe(Schema.minLength(1)),
    expectedOwner: AcnOwnerRecordSchema,
    shutdownReason: AcnDaemonShutdownReasonSchema,
    message: Schema.String,
  },
) {}

export class AcnDaemonProcessGroupAbsenceUnproven extends Schema.TaggedError<AcnDaemonProcessGroupAbsenceUnproven>()(
  "AcnDaemonProcessGroupAbsenceUnproven",
  {
    reason: Schema.String.pipe(Schema.minLength(1)),
    expectedOwner: AcnOwnerRecordSchema,
    shutdownReason: AcnDaemonShutdownReasonSchema,
  },
) {}

export class AcnAdministrationFailed extends Schema.TaggedError<AcnAdministrationFailed>()(
  "AcnAdministrationFailed",
  { reason: Schema.String.pipe(Schema.minLength(1)) },
) {}

export class BinaryNotFound extends Schema.TaggedError<BinaryNotFound>()(
  "BinaryNotFound",
  { path: Schema.String },
) {}

export class BinaryVersionMismatch extends Schema.TaggedError<BinaryVersionMismatch>()(
  "BinaryVersionMismatch",
  {
    path: Schema.String,
    expected: Schema.String,
    actual: Schema.String,
  },
) {}

export class BinaryRevisionMismatch extends Schema.TaggedError<BinaryRevisionMismatch>()(
  "BinaryRevisionMismatch",
  {
    path: Schema.String,
    expected: Schema.Number.pipe(Schema.int(), Schema.positive()),
    actual: Schema.Number.pipe(Schema.int(), Schema.positive()),
  },
) {}

export class DownloadFailed extends Schema.TaggedError<DownloadFailed>()(
  "DownloadFailed",
  {
    url: Schema.String,
    status: Schema.Number,
    reason: Schema.String,
  },
) {}

export class ChecksumMismatch extends Schema.TaggedError<ChecksumMismatch>()(
  "ChecksumMismatch",
  {
    path: Schema.String,
    expected: Schema.String,
    actual: Schema.String,
  },
) {}

export const AcnEnsuranceError = Schema.Union(
  AcnEnsuranceFailed,
  AcnOwnerRecordReadUnavailable,
  AcnOwnerRecordInvalid,
  AcnProcessIdentityObservationTimedOut,
  AcnCandidateSpawnFailed,
  AcnCandidateProcessGroupObservationFailed,
  AcnCandidateProcessGroupTerminationFailed,
  AcnCandidateProcessGroupTerminationPermissionDenied,
  AcnCandidateProcessGroupKillFailed,
  AcnCandidateProcessGroupKillPermissionDenied,
  AcnCandidateProcessGroupAbsenceUnproven,
  AcnCandidateProcessGroupLeaderChanged,
  AcnCandidateBootstrapProcessStopFailed,
  AcnCandidateBootstrapProcessExitUnproven,
  AcnCandidateExactProcessPidMismatch,
  AcnCandidateExactProcessAlreadyConfirmed,
  AcnCandidateAdmissionBeforeExactProcessConfirmed,
  AcnCandidateParentChannelReleaseFailed,
  AcnCandidateAdmissionAlreadyAcknowledged,
  AcnCandidateLaunchAlreadyAttempted,
  AcnCandidateIdentityUnavailable,
  AcnCandidateExitedBeforeIdentityObserved,
  AcnCandidateAdmissionAcknowledgementTimedOut,
  AcnCandidateReadyBeforeAdmission,
  AcnCandidateReadyInstanceMismatch,
  AcnDaemonTargetUnsupported,
  AcnLaunchOverrideTargetMismatch,
  AcnCandidateExitedBeforeAdmissionFailure,
  AcnCandidateExitedAfterAdmissionFailure,
  AcnCandidateAdmissionTimedOut,
  AcnCandidateAdmissionLost,
  AcnCandidateOwnershipLostAfterAdmission,
  AcnDaemonStartupTimedOut,
  AcnEnsuranceConvergenceTimedOut,
  ExactProcessIdentityObservationFailed,
  ProcessGroupObservationFailed,
  AcnDaemonOwnerObservationFailed,
  AcnDaemonIdentityObservationFailed,
  AcnDaemonProcessGroupObservationFailed,
  AcnDaemonProcessGroupTerminationPermissionDenied,
  AcnDaemonProcessGroupTerminationFailed,
  AcnDaemonProcessGroupKillPermissionDenied,
  AcnDaemonProcessGroupKillFailed,
  AcnDaemonProcessGroupAbsenceUnproven,
  BinaryNotFound,
  BinaryVersionMismatch,
  BinaryRevisionMismatch,
  DownloadFailed,
  ChecksumMismatch,
)
export type AcnEnsuranceError = typeof AcnEnsuranceError.Type

export type StreamDisplayViewFailure =
  | Rpc.ErrorExit<typeof StreamDisplayView>
  | RpcClientError.RpcClientError
  | AcnEnsuranceError

export type WatchFileFailure =
  | Rpc.ErrorExit<typeof WatchFile>
  | RpcClientError.RpcClientError
  | AcnEnsuranceError
