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
    "Discovering",
    "WaitingForOwner",
    "LaunchingAcn",
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

/** Stable liveness key: changes only for authoritative phase or measured progress. */
export const acnStartupProgressKey = (state: AcnHealthState): string => {
  if (state._tag !== "Starting") return state._tag;
  const activity =
    typeof state.activity === "string"
      ? state.activity
      : state.activity._tag === "PreparingBackend"
      ? `${state.activity._tag}:${state.activity.backend._tag}:${state.activity.backend.hardwareLabel}`
      : `${state.activity._tag}:${state.activity.phase}:${state.activity.plan.daemonBytes}:${state.activity.plan.inferenceEngineBytes}`;
  return Option.match(state.progress, {
    onNone: () => activity,
    onSome: (progress) =>
      `${activity}:${progress.completed}:${progress.totalBytes}:${
        progress.unit
      }:${Option.getOrUndefined(progress.attempt) ?? ""}`,
  });
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
    Ready: ["Starting"],
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
  if (state._tag === "Starting" && state.phase === "LaunchingAcn") {
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
    case "BinaryNotFound":
      return `Magnitude executable was not found at ${error.path}`;
    case "BinaryVersionMismatch":
      return `Magnitude executable ${error.path} has version ${error.actual}; expected ${error.expected}`;
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
