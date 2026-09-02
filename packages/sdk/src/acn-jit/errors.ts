import { Schema } from "effect"
import { RpcClientError } from "@effect/rpc"
import type { Subscription } from "@magnitudedev/effect-query"
import { AcnTargetSchema, Display, Files } from "@magnitudedev/acn-protocol"
import {
  AcnOwnerRecordSchema,
  ExactProcessIdentityObservationFailed,
  ProcessGroupAbsenceUnproven,
  ProcessGroupObservationFailed,
  ProcessGroupSignalFailed,
  ProcessGroupSignalPermissionDenied,
} from "@magnitudedev/acn-protocol/coordination"

const ProcessIdSchema = Schema.Number.pipe(Schema.int(), Schema.positive())

export class AcnEnsuranceFailed extends Schema.TaggedError<AcnEnsuranceFailed>()(
  "AcnEnsuranceFailed",
  { reason: Schema.String.pipe(Schema.minLength(1)) },
) {}

export class AcnHealthRequestFailed extends Schema.TaggedClass<AcnHealthRequestFailed>()(
  "AcnHealthRequestFailed",
  { message: Schema.String.pipe(Schema.minLength(1)) },
) {}

export class AcnHealthAttemptTimedOut extends Schema.TaggedClass<AcnHealthAttemptTimedOut>()(
  "AcnHealthAttemptTimedOut",
  {},
) {}

export class AcnHealthResponseInvalid extends Schema.TaggedClass<AcnHealthResponseInvalid>()(
  "AcnHealthResponseInvalid",
  { message: Schema.String.pipe(Schema.minLength(1)) },
) {}

export const AcnHealthAttemptFailureSchema = Schema.Union(
  AcnHealthRequestFailed,
  AcnHealthAttemptTimedOut,
  AcnHealthResponseInvalid,
)
export type AcnHealthAttemptFailure = typeof AcnHealthAttemptFailureSchema.Type

export class AcnHealthUnavailable extends Schema.TaggedError<AcnHealthUnavailable>()(
  "AcnHealthUnavailable",
  {
    owner: AcnOwnerRecordSchema,
    attempts: Schema.Tuple(AcnHealthAttemptFailureSchema, AcnHealthAttemptFailureSchema),
  },
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
  { pid: ProcessIdSchema },
) {}

export class AcnCandidateSpawnFailed extends Schema.TaggedError<AcnCandidateSpawnFailed>()(
  "AcnCandidateSpawnFailed",
  { message: Schema.String },
) {}

export class AcnCandidateIdentityUnavailable extends Schema.TaggedError<AcnCandidateIdentityUnavailable>()(
  "AcnCandidateIdentityUnavailable",
  { pid: ProcessIdSchema },
) {}

const CandidateExitFields = {
  pid: ProcessIdSchema,
  code: Schema.Number,
  stderr: Schema.String,
}

export class AcnCandidateExitedBeforeAdmission extends Schema.TaggedError<AcnCandidateExitedBeforeAdmission>()(
  "AcnCandidateExitedBeforeAdmission",
  CandidateExitFields,
) {}

export class AcnCandidateExitedAfterAdmission extends Schema.TaggedError<AcnCandidateExitedAfterAdmission>()(
  "AcnCandidateExitedAfterAdmission",
  CandidateExitFields,
) {}

export class AcnCandidateAdmissionTimedOut extends Schema.TaggedError<AcnCandidateAdmissionTimedOut>()(
  "AcnCandidateAdmissionTimedOut",
  { pid: ProcessIdSchema },
) {}

export class AcnCandidateParentChannelReleaseFailed extends Schema.TaggedError<AcnCandidateParentChannelReleaseFailed>()(
  "AcnCandidateParentChannelReleaseFailed",
  { pid: ProcessIdSchema, message: Schema.String },
) {}

export class AcnCandidateOwnershipLost extends Schema.TaggedError<AcnCandidateOwnershipLost>()(
  "AcnCandidateOwnershipLost",
  { pid: ProcessIdSchema },
) {}

/** Every way one supervised candidate occurrence terminates without becoming ready. */
export const AcnCandidateFailureSchema = Schema.Union(
  AcnCandidateSpawnFailed,
  AcnCandidateIdentityUnavailable,
  AcnCandidateExitedBeforeAdmission,
  AcnCandidateExitedAfterAdmission,
  AcnCandidateAdmissionTimedOut,
  AcnCandidateParentChannelReleaseFailed,
  AcnCandidateOwnershipLost,
  ExactProcessIdentityObservationFailed,
  AcnProcessIdentityObservationTimedOut,
)
export type AcnCandidateFailure = typeof AcnCandidateFailureSchema.Type

export class AcnCandidateBootstrapProcessStopFailed extends Schema.TaggedError<AcnCandidateBootstrapProcessStopFailed>()(
  "AcnCandidateBootstrapProcessStopFailed",
  { pid: ProcessIdSchema, message: Schema.String },
) {}

export class AcnCandidateBootstrapProcessExitUnproven extends Schema.TaggedError<AcnCandidateBootstrapProcessExitUnproven>()(
  "AcnCandidateBootstrapProcessExitUnproven",
  { pid: ProcessIdSchema },
) {}

export class AcnDaemonTargetUnsupported extends Schema.TaggedError<AcnDaemonTargetUnsupported>()(
  "AcnDaemonTargetUnsupported",
  { requested: AcnTargetSchema, supported: AcnTargetSchema },
) {}

export class AcnLaunchOverrideTargetMismatch extends Schema.TaggedError<AcnLaunchOverrideTargetMismatch>()(
  "AcnLaunchOverrideTargetMismatch",
  { requested: AcnTargetSchema, override: AcnTargetSchema },
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

/** The typed control failures the shutdown supervisor can hit while retiring one daemon. */
export const AcnDaemonShutdownControlFailureSchema = Schema.Union(
  AcnOwnerRecordReadUnavailable,
  AcnOwnerRecordInvalid,
  ExactProcessIdentityObservationFailed,
  ProcessGroupObservationFailed,
  ProcessGroupSignalPermissionDenied,
  ProcessGroupSignalFailed,
  ProcessGroupAbsenceUnproven,
)
export type AcnDaemonShutdownControlFailure = typeof AcnDaemonShutdownControlFailureSchema.Type

export class AcnDaemonShutdownFailed extends Schema.TaggedError<AcnDaemonShutdownFailed>()(
  "AcnDaemonShutdownFailed",
  {
    owner: AcnOwnerRecordSchema,
    reason: AcnDaemonShutdownReasonSchema,
    failure: AcnDaemonShutdownControlFailureSchema,
  },
) {}

export class AcnDaemonStartupTimedOut extends Schema.TaggedError<AcnDaemonStartupTimedOut>()(
  "AcnDaemonStartupTimedOut",
  { owner: AcnOwnerRecordSchema },
) {}

export class AcnEnsuranceConvergenceTimedOut extends Schema.TaggedError<AcnEnsuranceConvergenceTimedOut>()(
  "AcnEnsuranceConvergenceTimedOut",
  {},
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
  AcnHealthUnavailable,
  AcnOwnerRecordReadUnavailable,
  AcnOwnerRecordInvalid,
  ExactProcessIdentityObservationFailed,
  ProcessGroupObservationFailed,
  ProcessGroupSignalPermissionDenied,
  ProcessGroupSignalFailed,
  ProcessGroupAbsenceUnproven,
  AcnProcessIdentityObservationTimedOut,
  AcnCandidateSpawnFailed,
  AcnCandidateIdentityUnavailable,
  AcnCandidateExitedBeforeAdmission,
  AcnCandidateExitedAfterAdmission,
  AcnCandidateAdmissionTimedOut,
  AcnCandidateParentChannelReleaseFailed,
  AcnCandidateOwnershipLost,
  AcnCandidateBootstrapProcessStopFailed,
  AcnCandidateBootstrapProcessExitUnproven,
  AcnDaemonShutdownFailed,
  AcnDaemonStartupTimedOut,
  AcnEnsuranceConvergenceTimedOut,
  AcnDaemonTargetUnsupported,
  AcnLaunchOverrideTargetMismatch,
  BinaryNotFound,
  BinaryVersionMismatch,
  BinaryRevisionMismatch,
  DownloadFailed,
  ChecksumMismatch,
)
export type AcnEnsuranceError = typeof AcnEnsuranceError.Type

export type StreamDisplayViewFailure =
  | Subscription.Error<typeof Display.StreamDisplayView>
  | RpcClientError.RpcClientError
  | AcnEnsuranceError

export type WatchFileFailure =
  | Subscription.Error<typeof Files.WatchFile>
  | RpcClientError.RpcClientError
  | AcnEnsuranceError
