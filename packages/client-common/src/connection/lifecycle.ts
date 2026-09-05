import {
  AcnInstallationPlanSchema,
  AcnInstallationPhaseSchema,
  AcnStartupProgressSchema,
  type AcnInstallationPhase,
  type AcnInstallationPlan,
  type AcnStartupProgress,
  type ConnectionState,
} from "@magnitudedev/sdk";
import { FSM } from "@magnitudedev/utils";
import { Effect, Option, Schema, Stream } from "effect";
import { formatConnectionError, type ConnectionError } from "@magnitudedev/sdk";

export { ServiceStartingPhaseSchema } from "@magnitudedev/sdk";
import {
  ServiceStartingPhaseSchema,
  ServiceStartProgressSchema,
  type ServiceStartProgress,
} from "@magnitudedev/sdk";
export type ServiceStartingPhase = typeof ServiceStartingPhaseSchema.Type;
export { ServiceStartProgressSchema, type ServiceStartProgress };

export const ServiceFailureStageSchema = Schema.Literal(
  "InstallDaemon",
  "LaunchDaemon",
  "PrepareLocalInference",
  "Connect"
);
export type ServiceFailureStage = typeof ServiceFailureStageSchema.Type;

const NormalizedProgressSchema = Schema.Number.pipe(Schema.between(0, 1));

export class ServiceChecking extends Schema.TaggedClass<ServiceChecking>()(
  "Checking",
  {}
) {}
export class ServiceStarting extends Schema.TaggedClass<ServiceStarting>()(
  "Starting",
  {
    phase: ServiceStartingPhaseSchema,
  }
) {}
export class ServiceInstalling extends Schema.TaggedClass<ServiceInstalling>()(
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
export class ServiceReady extends Schema.TaggedClass<ServiceReady>()(
  "Ready",
  {}
) {}
export class ServiceFailed extends Schema.TaggedClass<ServiceFailed>()(
  "Failed",
  {
    stage: ServiceFailureStageSchema,
    message: Schema.String.pipe(Schema.minLength(1)),
    retryable: Schema.Boolean,
  }
) {}

export const ServiceLifecycleStateSchema = Schema.Union(
  ServiceChecking,
  ServiceStarting,
  ServiceInstalling,
  ServiceReady,
  ServiceFailed
);
export type ServiceLifecycleState = typeof ServiceLifecycleStateSchema.Type;

export interface ServiceLifecycle {
  readonly get: Effect.Effect<ServiceLifecycleState>;
  readonly changes: Stream.Stream<ServiceLifecycleState>;
}

interface PhaseRange {
  readonly start: number;
  readonly end: number;
}

class InstallationModel extends Schema.TaggedClass<InstallationModel>()(
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

const LifecycleStates = {
  Checking: ServiceChecking,
  Starting: ServiceStarting,
  Installing: InstallationModel,
  Ready: ServiceReady,
  Failed: ServiceFailed,
};
export const LifecycleModelSchema = Schema.Union(
  ...Object.values(LifecycleStates)
);
const LifecycleFsm = FSM.defineFSM(LifecycleStates, {
  Checking: ["Starting", "Installing", "Ready", "Failed"],
  Starting: ["Starting", "Installing", "Ready", "Failed"],
  Installing: ["Starting", "Installing", "Ready", "Failed"],
  Ready: [],
  Failed: ["Checking", "Starting", "Installing", "Ready", "Failed"],
} as const);

export type LifecycleModel = typeof LifecycleModelSchema.Type;

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
  installation: InstallationModel,
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
  installation: InstallationModel,
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
  installation: InstallationModel,
  now: number
): ServiceLifecycleState => {
  return new ServiceInstalling({
    phase: installation.phase,
    overallProgress: overallProgressAt(installation, now),
    detailIsExact:
      installation.phase !== "DownloadingInferenceEngine" ||
      installation.plan.inferenceEngineBytesExact,
    detail: installation.detail,
  });
};

export const renderLifecycle = (
  internal: LifecycleModel,
  now: number
): ServiceLifecycleState =>
  internal._tag === "Installing" ? renderInstallation(internal, now) : internal;

export const lifecycleIsTimeDependent = (model: LifecycleModel): boolean =>
  model._tag === "Installing" && model.phase === "StartingMagnitude";

const failureStage = (state: LifecycleModel): ServiceFailureStage => {
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

const nonEmptyFailureMessage = (error: ConnectionError): string => {
  const message = formatConnectionError(error).trim();
  return message.length > 0 ? message : "Magnitude is unavailable";
};

/** Pure projection: timestamps come from the presenter, never from the reducer. */
export const reduceLifecycle = (
  current: LifecycleModel,
  state: ConnectionState,
  now: number
): LifecycleModel => {
  if (current._tag === "Ready") return current;
  switch (state._tag) {
    case "Idle":
    case "Closed":
      return current;
    case "Ready":
      return LifecycleFsm.transition(current, "Ready", {});
    case "Failed":
      return LifecycleFsm.transition(current, "Failed", {
        stage: failureStage(current),
        message: nonEmptyFailureMessage(state.error),
        retryable: true,
      });
    case "Connecting": {
      const active =
        current._tag === "Failed"
          ? LifecycleFsm.transition(current, "Checking", {})
          : current;
      if (Option.isNone(state.activity)) return active;
      const observation = state.activity.value;
      if (observation._tag === "Starting") {
        if (
          active._tag === "Installing" &&
          typeof observation.phase === "string"
        )
          return active;
        return LifecycleFsm.transition(active, "Starting", {
          phase: observation.phase,
        });
      }
      const installing = active._tag === "Installing" ? active : undefined;
      return LifecycleFsm.transition(active, "Installing", {
        phase: observation.phase,
        plan: observation.plan,
        daemonIncluded:
          installing !== undefined
            ? installing.daemonIncluded
            : observation.phase === "DownloadingDaemon",
        startedAtMillis:
          installing !== undefined && installing.phase === observation.phase
            ? installing.startedAtMillis
            : now,
        maximumOverallProgress:
          installing !== undefined ? overallProgressAt(installing, now) : 0,
        detail: observation.progress,
      });
    }
  }
};

export type { AcnInstallationPhase, AcnInstallationPlan, AcnStartupProgress };
