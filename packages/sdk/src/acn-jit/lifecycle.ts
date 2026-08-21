import {
  AcnInstallationPlanSchema,
  AcnInstallationPhaseSchema,
  AcnStartupProgressSchema,
  type AcnInstallationPhase,
  type AcnInstallationPlan,
  type AcnStartupProgress,
  type AcnHealthState,
} from "@magnitudedev/acn-protocol";
import { FSM } from "@magnitudedev/utils";
import { Clock, Duration, Effect, Option, Schema, Stream, SubscriptionRef } from "effect";
import type { AcnEnsuranceError } from "./errors";

export const AcnStartingPhaseSchema = Schema.Union(
  Schema.Literal(
    "PreparingAcn",
    "WaitingForOwner",
    "ResolvingLocalInference",
    "LaunchingLocalInference"
  ),
  Schema.TaggedStruct("PreparingBackend", {
    backend: Schema.Union(
      Schema.TaggedStruct("Cpu", { hardwareLabel: Schema.NonEmptyString }),
      Schema.TaggedStruct("Metal", { hardwareLabel: Schema.NonEmptyString }),
      Schema.TaggedStruct("Cuda", { hardwareLabel: Schema.NonEmptyString }),
      Schema.TaggedStruct("Vulkan", { hardwareLabel: Schema.NonEmptyString })
    ),
  })
);
export type AcnStartingPhase = typeof AcnStartingPhaseSchema.Type;

export const AcnFailureStageSchema = Schema.Literal(
  "InstallDaemon",
  "LaunchDaemon",
  "PrepareLocalInference",
  "Connect"
);
export type AcnFailureStage = typeof AcnFailureStageSchema.Type;

const NormalizedProgressSchema = Schema.Number.pipe(Schema.between(0, 1));

export class ClientAcnChecking extends Schema.TaggedClass<ClientAcnChecking>()(
  "Checking",
  {}
) {}
export class ClientAcnStarting extends Schema.TaggedClass<ClientAcnStarting>()(
  "Starting",
  {
    phase: AcnStartingPhaseSchema,
  }
) {}
export class ClientAcnInstalling extends Schema.TaggedClass<ClientAcnInstalling>()(
  "Installing",
  {
    phase: AcnInstallationPhaseSchema,
    overallProgress: NormalizedProgressSchema,
    detailIsExact: Schema.Boolean,
    detail: Schema.optionalWith(AcnStartupProgressSchema, {
      as: "Option",
      exact: true,
    }),
  }
) {}
export class ClientAcnReady extends Schema.TaggedClass<ClientAcnReady>()(
  "Ready",
  {}
) {}
export class ClientAcnFailed extends Schema.TaggedClass<ClientAcnFailed>()(
  "Failed",
  {
    stage: AcnFailureStageSchema,
    message: Schema.String.pipe(Schema.minLength(1)),
    retryable: Schema.Boolean,
  }
) {}

export const AcnLifecycleStateSchema = Schema.Union(
  ClientAcnChecking,
  ClientAcnStarting,
  ClientAcnInstalling,
  ClientAcnReady,
  ClientAcnFailed
);
export type AcnLifecycleState = typeof AcnLifecycleStateSchema.Type;

export const AcnLifecycleObservationSchema = Schema.Union(
  Schema.TaggedStruct("Starting", {
    phase: AcnStartingPhaseSchema,
  }),
  Schema.TaggedStruct("Installing", {
    phase: AcnInstallationPhaseSchema,
    plan: AcnInstallationPlanSchema,
    progress: Schema.optionalWith(AcnStartupProgressSchema, {
      as: "Option",
      exact: true,
    }),
  })
);
export type AcnLifecycleObservation = typeof AcnLifecycleObservationSchema.Type;

/** Projects authoritative daemon startup state into client presentation state. */
export const acnLifecycleObservationFromHealthState = (
  state: AcnHealthState
): Option.Option<AcnLifecycleObservation> => {
  if (state._tag !== "Starting") return Option.none();
  if (typeof state.activity !== "string") {
    return Option.some(
      state.activity._tag === "Installing"
        ? {
            _tag: "Installing",
            phase: state.activity.phase,
            plan: state.activity.plan,
            progress: state.progress,
          }
        : { _tag: "Starting", phase: state.activity }
    );
  }
  const phase = {
    WaitingForOwnership: "WaitingForOwner",
    Resolving: "ResolvingLocalInference",
    Starting: "LaunchingLocalInference",
  } as const;
  return Option.some({ _tag: "Starting", phase: phase[state.activity] });
};

export interface AcnLifecycle {
  readonly get: Effect.Effect<AcnLifecycleState>;
  readonly changes: Stream.Stream<AcnLifecycleState>;
}

export interface AcnLifecycleOwner extends AcnLifecycle {
  readonly report: (
    observation: AcnLifecycleObservation
  ) => Effect.Effect<void>;
  readonly ready: Effect.Effect<void>;
  readonly fail: (error: AcnEnsuranceError) => Effect.Effect<void>;
}

interface PhaseRange {
  readonly start: number;
  readonly end: number;
}

class ClientAcnInstallingAuthority extends Schema.TaggedClass<ClientAcnInstallingAuthority>()(
  "Installing",
  {
    phase: AcnInstallationPhaseSchema,
    plan: AcnInstallationPlanSchema,
    daemonIncluded: Schema.Boolean,
    startedAtMillis: Schema.Number,
    maximumOverallProgress: NormalizedProgressSchema,
    detail: Schema.optionalWith(AcnStartupProgressSchema, {
      as: "Option",
      exact: true,
    }),
  }
) {}

const ClientAcnLifecycleFsm = FSM.defineFSM(
  {
    Checking: ClientAcnChecking,
    Starting: ClientAcnStarting,
    Installing: ClientAcnInstallingAuthority,
    Ready: ClientAcnReady,
    Failed: ClientAcnFailed,
  },
  {
    Checking: ["Starting", "Installing", "Ready", "Failed"],
    Starting: ["Starting", "Installing", "Ready", "Failed"],
    Installing: ["Starting", "Installing", "Ready", "Failed"],
    Ready: [],
    Failed: ["Starting", "Installing", "Ready", "Failed"],
  } as const
);

type InternalState =
  | ClientAcnChecking
  | ClientAcnStarting
  | ClientAcnInstallingAuthority
  | ClientAcnReady
  | ClientAcnFailed;

const STARTING_MAGNITUDE_SHARE = 0.1;
const DOWNLOAD_SHARE = 1 - STARTING_MAGNITUDE_SHARE;
const STARTING_MAGNITUDE_EXPECTED_MILLIS = 7_500;

const phaseRange = (
  phase: AcnInstallationPhase,
  plan: AcnInstallationPlan,
  daemonIncluded: boolean
): PhaseRange => {
  const daemonBytes = daemonIncluded ? plan.daemonBytes : 0;
  const downloadBytes = daemonBytes + plan.inferenceEngineBytes;
  const daemonEnd =
    downloadBytes === 0 ? 0 : DOWNLOAD_SHARE * (daemonBytes / downloadBytes);
  switch (phase) {
    case "DownloadingDaemon":
      return { start: 0, end: daemonEnd };
    case "DownloadingInferenceEngine":
      return { start: daemonEnd, end: DOWNLOAD_SHARE };
    case "StartingMagnitude":
      return { start: DOWNLOAD_SHARE, end: 1 };
  }
};

const measuredFraction = (
  progress: Option.Option<AcnStartupProgress>
): Option.Option<number> =>
  Option.map(progress, ({ completed, totalBytes }) =>
    Math.max(0, Math.min(1, completed / totalBytes))
  );

const phaseFractionAt = (
  installation: ClientAcnInstallingAuthority,
  now: number
): number =>
  installation.phase === "StartingMagnitude"
    ? Math.min(
        0.999_999,
        1 -
          Math.exp(
            (-Math.log(10) * Math.max(0, now - installation.startedAtMillis)) /
              STARTING_MAGNITUDE_EXPECTED_MILLIS
          )
      )
    : Option.getOrElse(measuredFraction(installation.detail), () => 1);

const overallProgressAt = (
  installation: ClientAcnInstallingAuthority,
  now: number
): number => {
  const range = phaseRange(
    installation.phase,
    installation.plan,
    installation.daemonIncluded
  );
  const observed =
    range.start +
    (range.end - range.start) * phaseFractionAt(installation, now);
  return Math.min(
    0.999_999,
    Math.max(installation.maximumOverallProgress, observed)
  );
};

const renderInstallation = (
  installation: ClientAcnInstallingAuthority,
  now: number
): AcnLifecycleState => {
  return new ClientAcnInstalling({
    phase: installation.phase,
    overallProgress: overallProgressAt(installation, now),
    detailIsExact:
      installation.phase !== "DownloadingInferenceEngine" ||
      installation.plan.inferenceEngineBytesExact,
    detail: installation.detail,
  });
};

const renderState = (
  internal: InternalState
): Effect.Effect<AcnLifecycleState> =>
  internal._tag === "Installing"
    ? Clock.currentTimeMillis.pipe(
        Effect.map((now) => renderInstallation(internal, now))
      )
    : Effect.succeed(internal);

const failureStage = (state: InternalState): AcnFailureStage => {
  if (state._tag === "Starting" && state.phase === "PreparingAcn") {
    return "LaunchDaemon";
  }
  if (state._tag === "Installing") {
    return state.phase === "DownloadingDaemon"
      ? "InstallDaemon"
      : "PrepareLocalInference";
  }
  return "Connect";
};

const ensuranceErrorMessage = (error: AcnEnsuranceError): string => {
  switch (error._tag) {
    case "AcnEnsuranceFailed":
      return error.reason;
    case "AcnOwnerRecordReadUnavailable":
      return `Could not read ACN owner record at ${error.path}: ${error.message}`;
    case "AcnOwnerRecordInvalid":
      return `Invalid ACN owner record at ${error.path}: ${error.message}`;
    case "AcnProcessIdentityObservationTimedOut":
      return `Exact process inspection remained unavailable for PID ${error.pid}`;
    case "AcnCandidateSpawnFailed":
      return `Could not spawn the ACN candidate: ${error.message}`;
    case "AcnCandidateProcessGroupObservationFailed":
      return `Could not observe ACN candidate process group ${error.pid}: ${error.message}`;
    case "AcnCandidateProcessGroupTerminationFailed":
      return `Could not terminate ACN candidate process group ${error.pid}: ${error.message}`;
    case "AcnCandidateProcessGroupTerminationPermissionDenied":
      return `Permission denied terminating ACN candidate process group ${error.pid}: ${error.message}`;
    case "AcnCandidateProcessGroupKillFailed":
      return `Could not kill ACN candidate process group ${error.pid}: ${error.message}`;
    case "AcnCandidateProcessGroupKillPermissionDenied":
      return `Permission denied killing ACN candidate process group ${error.pid}: ${error.message}`;
    case "AcnCandidateProcessGroupAbsenceUnproven":
      return `ACN candidate process group ${error.pid} remained observable after kill`;
    case "AcnCandidateProcessGroupLeaderChanged":
      return `ACN candidate process-group leader ${error.candidate.pid} changed identity before cleanup`;
    case "AcnCandidateBootstrapProcessStopFailed":
      return `Could not stop ACN bootstrap process ${error.pid}: ${error.message}`;
    case "AcnCandidateBootstrapProcessExitUnproven":
      return `Could not prove ACN bootstrap process ${error.pid} exited`;
    case "AcnCandidateExactProcessPidMismatch":
      return `Observed ACN process ${error.observed.pid} does not match spawned candidate PID ${error.candidatePid}`;
    case "AcnCandidateExactProcessAlreadyConfirmed":
      return `ACN candidate ${error.confirmed.pid} was already confirmed as a different exact process`;
    case "AcnCandidateAdmissionBeforeExactProcessConfirmed":
      return `ACN candidate ${error.pid} cannot be admitted before exact process confirmation`;
    case "AcnCandidateParentChannelReleaseFailed":
      return `Could not release the parent channel for ACN candidate ${error.pid}: ${error.message}`;
    case "AcnCandidateAdmissionAlreadyAcknowledged":
      return `Startup admission was already acknowledged for ACN candidate ${error.pid}`;
    case "AcnCandidateLaunchAlreadyAttempted":
      return "An ACN candidate launch was already attempted";
    case "AcnCandidateIdentityUnavailable":
      return `ACN candidate ${error.pid} was not available for exact identity observation`;
    case "AcnCandidateExitedBeforeIdentityObserved":
      return `ACN candidate ${error.pid} exited with code ${error.code} before identity observation${error.stderr ? `:\n${error.stderr}` : ""}`;
    case "AcnCandidateAdmissionAcknowledgementTimedOut":
      return `Startup admission acknowledgement timed out for ACN candidate ${error.pid}`;
    case "AcnCandidateReadyBeforeAdmission":
      return "The ACN candidate reported ready before startup admission";
    case "AcnCandidateReadyInstanceMismatch":
      return `Ready ACN ${error.ready.pid} is not the supervised candidate ${error.candidate.pid}`;
    case "AcnDaemonTargetUnsupported":
      return `This client supports ACN revision ${error.supported.revision}, not requested revision ${error.requested.revision}`;
    case "AcnLaunchOverrideTargetMismatch":
      return `The ACN launch override targets revision ${error.override.revision}, not requested revision ${error.requested.revision}`;
    case "AcnCandidateExitedBeforeAdmissionFailure":
      return `Magnitude daemon ${error.process.pid} exited with code ${error.code} before startup admission${error.stderr ? `:\n${error.stderr}` : ""}`;
    case "AcnCandidateExitedAfterAdmissionFailure":
      return `Magnitude daemon ${error.process.pid} exited with code ${error.code} after admission before it became ready${error.stderr ? `:\n${error.stderr}` : ""}`;
    case "AcnCandidateAdmissionTimedOut":
      return `Magnitude daemon ${error.process.pid} did not complete startup admission`;
    case "AcnCandidateAdmissionLost":
      return `Magnitude daemon ${error.process.pid} lost startup admission acknowledgement`;
    case "AcnCandidateOwnershipLostAfterAdmission":
      return `Magnitude daemon ${error.process.pid} lost ownership after startup admission`;
    case "AcnDaemonStartupTimedOut":
      return `Magnitude daemon ${error.owner.pid} did not become ready within the startup deadline`;
    case "AcnEnsuranceConvergenceTimedOut":
      return "ACN ensurance did not converge within its absolute deadline";
    case "ExactProcessIdentityObservationFailed":
    case "ProcessGroupObservationFailed":
      return error.message;
    case "AcnDaemonOwnerObservationFailed":
    case "AcnDaemonIdentityObservationFailed":
    case "AcnDaemonProcessGroupObservationFailed":
    case "AcnDaemonProcessGroupTerminationPermissionDenied":
    case "AcnDaemonProcessGroupTerminationFailed":
    case "AcnDaemonProcessGroupKillPermissionDenied":
    case "AcnDaemonProcessGroupKillFailed":
    case "AcnDaemonProcessGroupAbsenceUnproven":
      return error.reason;
    case "BinaryNotFound":
      return `Magnitude executable was not found at ${error.path}`;
    case "BinaryVersionMismatch":
      return `Magnitude executable ${error.path} has version ${error.actual}; expected ${error.expected}`;
    case "BinaryRevisionMismatch":
      return `Magnitude executable ${error.path} has ACN revision ${error.actual}; expected ${error.expected}`;
    case "DownloadFailed":
      return error.reason;
    case "ChecksumMismatch":
      return "Downloaded Magnitude artifact failed integrity verification";
  }
};

const nonEmptyFailureMessage = (error: AcnEnsuranceError): string => {
  const message = ensuranceErrorMessage(error).trim();
  return message.length > 0 ? message : "Magnitude is unavailable";
};

export const makeAcnLifecycle = (): Effect.Effect<AcnLifecycleOwner> =>
  Effect.gen(function* () {
    const internal = yield* SubscriptionRef.make<InternalState>(
      new ClientAcnChecking({})
    );
    const transitionLock = yield* Effect.makeSemaphore(1);

    const get = SubscriptionRef.get(internal).pipe(Effect.flatMap(renderState));
    const changes = internal.changes.pipe(
      Stream.flatMap(
        (state) => {
          const current = Stream.fromEffect(renderState(state));
          if (
            state._tag !== "Installing" ||
            state.phase !== "StartingMagnitude"
          ) {
            return current;
          }
          return Stream.merge(
            current,
            Stream.tick(Duration.millis(50)).pipe(
              Stream.mapEffect(() => renderState(state))
            )
          );
        },
        { switch: true }
      )
    );

    const report = (
      observation: AcnLifecycleObservation
    ): Effect.Effect<void> =>
      transitionLock.withPermits(1)(
        Effect.gen(function* () {
          const current = yield* SubscriptionRef.get(internal);
          if (current._tag === "Ready") return;
          if (observation._tag === "Starting") {
            if (
              current._tag === "Installing" &&
              typeof observation.phase === "string"
            )
              return;
            yield* SubscriptionRef.set(
              internal,
              ClientAcnLifecycleFsm.transition(current, "Starting", {
                phase: observation.phase,
              })
            );
            return;
          }

          const now = yield* Clock.currentTimeMillis;
          const daemonIncluded =
            current._tag === "Installing"
              ? current.daemonIncluded
              : observation.phase === "DownloadingDaemon";
          const currentOverall =
            current._tag === "Installing" ? overallProgressAt(current, now) : 0;
          const samePhase =
            current._tag === "Installing" &&
            current.phase === observation.phase;
          yield* SubscriptionRef.set(
            internal,
            ClientAcnLifecycleFsm.transition(current, "Installing", {
              phase: observation.phase,
              plan: observation.plan,
              daemonIncluded,
              startedAtMillis: samePhase ? current.startedAtMillis : now,
              maximumOverallProgress: currentOverall,
              detail: observation.progress,
            })
          );
        })
      );

    return {
      get,
      changes,
      report,
      ready: transitionLock.withPermits(1)(
        Effect.gen(function* () {
          const current = yield* SubscriptionRef.get(internal);
          if (current._tag === "Ready") return;
          yield* SubscriptionRef.set(
            internal,
            ClientAcnLifecycleFsm.transition(current, "Ready", {})
          );
        })
      ),
      fail: (error) =>
        transitionLock.withPermits(1)(
          Effect.gen(function* () {
            const current = yield* SubscriptionRef.get(internal);
            if (current._tag === "Ready") {
              return;
            }
            yield* SubscriptionRef.set(
              internal,
              ClientAcnLifecycleFsm.transition(current, "Failed", {
                stage: failureStage(current),
                message: nonEmptyFailureMessage(error),
                retryable: true,
              })
            );
          })
        ),
    };
  });

export type { AcnInstallationPhase, AcnInstallationPlan, AcnStartupProgress };
