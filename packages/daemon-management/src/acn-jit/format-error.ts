import type { AcnEnsuranceError, AcnHealthAttemptFailure } from "./errors"
const shutdownControlFailureMessage = (
  error: Extract<AcnEnsuranceError, { readonly _tag: "AcnDaemonShutdownFailed" }>
): string => {
  const failure = error.failure;
  switch (failure._tag) {
    case "AcnOwnerRecordReadUnavailable":
      return `could not reread the service owner record at ${failure.path}: ${failure.message}`;
    case "AcnOwnerRecordInvalid":
      return `the service owner record at ${failure.path} is invalid: ${failure.message}`;
    case "ExactProcessIdentityObservationFailed":
      return `could not observe exact identity of PID ${failure.pid}: ${failure.message}`;
    case "ProcessGroupObservationFailed":
      return `could not observe its process group: ${failure.message}`;
    case "ProcessGroupSignalPermissionDenied":
      return `permission denied signaling its process group: ${failure.message}`;
    case "ProcessGroupSignalFailed":
      return `could not signal its process group: ${failure.message}`;
    case "ProcessGroupAbsenceUnproven":
      return "its process group remained observable after kill";
  }
};

const formatHealthAttemptFailure = (failure: AcnHealthAttemptFailure): string => {
  switch (failure._tag) {
    case "AcnHealthRequestFailed":
      return `request failed: ${failure.message}`;
    case "AcnHealthAttemptTimedOut":
      return "timed out after 2 seconds";
    case "AcnHealthResponseInvalid":
      return `response was invalid: ${failure.message}`;
  }
};

/** Formats an ensurance failure without imposing a client-specific heading. */
export const formatAcnEnsuranceError = (error: AcnEnsuranceError): string => {
  switch (error._tag) {
    case "AcnEnsuranceFailed":
      return error.reason;
    case "AcnHealthUnavailable":
      return [
        `A live Magnitude service process was observed at http://127.0.0.1:${error.owner.port}/health (PID ${error.owner.pid}), but neither health check produced a valid response.`,
        `Health check 1: ${formatHealthAttemptFailure(error.attempts[0])}`,
        `Health check 2: ${formatHealthAttemptFailure(error.attempts[1])}`,
        "Run `magnitude service start`.",
      ].join("\n");
    case "AcnOwnerRecordReadUnavailable":
      return `Could not read the Magnitude service owner record at ${error.path}: ${error.message}`;
    case "AcnOwnerRecordInvalid":
      return `Invalid Magnitude service owner record at ${error.path}: ${error.message}`;
    case "ExactProcessIdentityObservationFailed":
    case "ProcessGroupObservationFailed":
      return error.message;
    case "ProcessGroupSignalPermissionDenied":
      return `Could not stop the previous Magnitude service: ${error.message}`;
    case "ProcessGroupSignalFailed":
      return `Could not stop the previous Magnitude service: ${error.message}`;
    case "ProcessGroupAbsenceUnproven":
      return "The previous Magnitude service did not stop";
    case "AcnProcessIdentityObservationTimedOut":
      return "Could not verify the running Magnitude service process";
    case "AcnCandidateSpawnFailed":
      return `Could not start the Magnitude service: ${error.message}`;
    case "AcnCandidateIdentityUnavailable":
      return "Magnitude service failed to start because its process could not be verified";
    case "AcnCandidateExitedBeforeAdmission":
    case "AcnCandidateExitedAfterAdmission":
      return error.stderr.trim().length > 0
        ? error.stderr.trim()
        : "The service process exited before becoming ready";
    case "AcnCandidateAdmissionTimedOut":
      return "Magnitude service failed to start in time";
    case "AcnCandidateParentChannelReleaseFailed":
      return `Could not release the parent channel for Magnitude service process ${error.pid}: ${error.message}`;
    case "AcnCandidateOwnershipLost":
      return "Magnitude service stopped unexpectedly during startup";
    case "AcnCandidateBootstrapProcessStopFailed":
      return `Magnitude service startup cleanup failed: ${error.message}`;
    case "AcnCandidateBootstrapProcessExitUnproven":
      return "Magnitude service startup cleanup did not finish";
    case "AcnDaemonShutdownFailed":
      return `Could not stop the previous Magnitude service: ${shutdownControlFailureMessage(error)}`;
    case "AcnDaemonStartupTimedOut":
      return "Magnitude service failed to start in time";
    case "AcnEnsuranceConvergenceTimedOut":
      return "Magnitude service startup did not converge within its deadline";
    case "AcnDaemonTargetUnsupported":
      return `This client supports Magnitude service revision ${error.supported.revision}, not requested revision ${error.requested.revision}`;
    case "AcnLaunchOverrideTargetMismatch":
      return `The Magnitude service launch override targets revision ${error.override.revision}, not requested revision ${error.requested.revision}`;
    case "BinaryNotFound":
      return `Magnitude executable was not found at ${error.path}`;
    case "BinaryVersionMismatch":
      return `Magnitude executable ${error.path} has version ${error.actual}; expected ${error.expected}`;
    case "BinaryRevisionMismatch":
      return `Magnitude executable ${error.path} has service revision ${error.actual}; expected ${error.expected}`;
    case "DownloadFailed":
      return error.reason;
    case "ChecksumMismatch":
      return "Downloaded Magnitude artifact failed integrity verification";
  }
};

