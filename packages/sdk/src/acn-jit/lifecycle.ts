import {
  AcnInstallationPlanSchema,
  AcnInstallationPhaseSchema,
  AcnStartupProgressSchema,
  type AcnInstallationPhase,
  type AcnInstallationPlan,
  type AcnStartupProgress,
} from "@magnitudedev/acn-protocol";
import { Clock, Effect, Option, Schema, Stream, SubscriptionRef } from "effect";
import type { DaemonError } from "./errors";

export const AcnStartingPhaseSchema = Schema.Union(
  Schema.Literal(
    "Discovering",
    "WaitingForOwner",
    "LaunchingAcn",
    "ResolvingLocalInference",
    "LaunchingLocalInference",
  ),
  Schema.TaggedStruct("PreparingBackend", {
    backend: Schema.Union(
      Schema.TaggedStruct("Cuda", { hardwareLabel: Schema.NonEmptyString }),
    ),
  }),
);
export type AcnStartingPhase = typeof AcnStartingPhaseSchema.Type;

export const AcnFailureStageSchema = Schema.Literal(
  "AcquireAcn",
  "StartAcn",
  "PrepareLocalInference",
  "Connect"
);
export type AcnFailureStage = typeof AcnFailureStageSchema.Type;

const NormalizedProgressSchema = Schema.Number.pipe(Schema.between(0, 1));

export const AcnLifecycleStateSchema = Schema.Union(
  Schema.TaggedStruct("Checking", {}),
  Schema.TaggedStruct("Starting", {
    phase: AcnStartingPhaseSchema,
  }),
  Schema.TaggedStruct("Installing", {
    phase: AcnInstallationPhaseSchema,
    overallProgress: NormalizedProgressSchema,
    detailIsExact: Schema.Boolean,
    detail: Schema.optionalWith(AcnStartupProgressSchema, {
      as: "Option",
      exact: true,
    }),
  }),
  Schema.TaggedStruct("Ready", {
    endpoint: Schema.String.pipe(Schema.minLength(1)),
    version: Schema.String.pipe(Schema.minLength(1)),
  }),
  Schema.TaggedStruct("Failed", {
    stage: AcnFailureStageSchema,
    message: Schema.String.pipe(Schema.minLength(1)),
    retryable: Schema.Boolean,
  })
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
export type AcnLifecycleObservation =
  typeof AcnLifecycleObservationSchema.Type;

export interface AcnLifecycle {
  readonly get: Effect.Effect<AcnLifecycleState>;
  readonly changes: Stream.Stream<AcnLifecycleState>;
}

export interface AcnLifecycleController extends AcnLifecycle {
  readonly report: (
    observation: AcnLifecycleObservation
  ) => Effect.Effect<void>;
  readonly ready: (endpoint: string, version: string) => Effect.Effect<void>;
  readonly fail: (error: DaemonError) => Effect.Effect<void>;
}

interface PhaseRange {
  readonly start: number;
  readonly end: number;
}

interface ActiveInstallation {
  readonly phase: AcnInstallationPhase;
  readonly plan: AcnInstallationPlan;
  readonly daemonIncluded: boolean;
  readonly startedAtMillis: number;
  readonly maximumOverallProgress: number;
  readonly detail: Option.Option<AcnStartupProgress>;
}

type InternalState =
  | {
      readonly _tag: "Visible";
      readonly state: Exclude<
        AcnLifecycleState,
        { readonly _tag: "Installing" }
      >;
    }
  | {
      readonly _tag: "Installing";
      readonly installation: ActiveInstallation;
    };

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
  installation: ActiveInstallation,
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
  installation: ActiveInstallation,
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
  installation: ActiveInstallation,
  now: number
): AcnLifecycleState => ({
  _tag: "Installing",
  phase: installation.phase,
  overallProgress: overallProgressAt(installation, now),
  detailIsExact:
    installation.phase !== "DownloadingInferenceEngine" ||
    installation.plan.inferenceEngineBytesExact,
  detail: installation.detail,
});

const renderState = (
  internal: InternalState
): Effect.Effect<AcnLifecycleState> =>
  internal._tag === "Visible"
    ? Effect.succeed(internal.state)
    : Clock.currentTimeMillis.pipe(
        Effect.map((now) => renderInstallation(internal.installation, now))
      );

const failureStage = (state: InternalState): AcnFailureStage => {
  if (state._tag === "Visible") {
    if (
      state.state._tag === "Starting" &&
      state.state.phase === "LaunchingAcn"
    ) {
      return "StartAcn";
    }
    return "Connect";
  }
  return state.installation.phase === "DownloadingDaemon"
    ? "AcquireAcn"
    : "PrepareLocalInference";
};

const daemonErrorMessage = (error: DaemonError): string => {
  switch (error._tag) {
    case "DaemonSpawnFailed":
      return error.reason;
    case "DaemonCrashed":
      return Option.match(error.diagnostic, {
        onNone: () => `Magnitude exited with code ${error.exitCode}`,
        onSome: (diagnostic) =>
          `Magnitude exited with code ${error.exitCode}: ${diagnostic}`,
      });
    case "BinaryNotFound":
      return `Magnitude executable was not found at ${error.path}`;
    case "BinaryVersionMismatch":
      return `Magnitude executable ${error.path} has version ${error.actual}; expected ${error.expected}`;
    case "RegistrationFileInvalid":
      return `Magnitude registration is invalid: ${error.reason}`;
    case "DownloadFailed":
      return error.reason;
    case "ChecksumMismatch":
      return "Downloaded Magnitude artifact failed integrity verification";
    case "NoDaemon":
      return "No Magnitude service is available";
  }
};

export const makeAcnLifecycle = (): Effect.Effect<AcnLifecycleController> =>
  Effect.gen(function* () {
    const internal = yield* SubscriptionRef.make<InternalState>({
      _tag: "Visible",
      state: { _tag: "Checking" },
    });

    const get = SubscriptionRef.get(internal).pipe(Effect.flatMap(renderState));
    const changes = internal.changes.pipe(
      Stream.flatMap(
        (state) => {
          const current = Stream.fromEffect(renderState(state));
          if (
            state._tag !== "Installing" ||
            state.installation.phase !== "StartingMagnitude"
          ) {
            return current;
          }
          return Stream.merge(
            current,
            Stream.tick("50 millis").pipe(
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
      Effect.gen(function* () {
        const current = yield* SubscriptionRef.get(internal);
        if (observation._tag === "Starting") {
          if (current._tag === "Installing" && typeof observation.phase === "string") return;
          yield* SubscriptionRef.set(internal, {
            _tag: "Visible",
            state: observation,
          });
          return;
        }

        const now = yield* Clock.currentTimeMillis;
        const daemonIncluded =
          current._tag === "Installing"
            ? current.installation.daemonIncluded
            : observation.phase === "DownloadingDaemon";
        const currentOverall =
          current._tag === "Installing"
            ? overallProgressAt(current.installation, now)
            : 0;
        const samePhase =
          current._tag === "Installing" &&
          current.installation.phase === observation.phase;
        yield* SubscriptionRef.set(internal, {
          _tag: "Installing",
          installation: {
            phase: observation.phase,
            plan: observation.plan,
            daemonIncluded,
            startedAtMillis: samePhase
              ? current.installation.startedAtMillis
              : now,
            maximumOverallProgress: currentOverall,
            detail: observation.progress,
          },
        });
      });

    return {
      get,
      changes,
      report,
      ready: (endpoint, version) =>
        SubscriptionRef.set(internal, {
          _tag: "Visible",
          state: { _tag: "Ready", endpoint, version },
        }),
      fail: (error) =>
        Effect.gen(function* () {
          const current = yield* SubscriptionRef.get(internal);
          if (current._tag === "Visible" && current.state._tag === "Ready") {
            return;
          }
          yield* SubscriptionRef.set(internal, {
            _tag: "Visible",
            state: {
              _tag: "Failed",
              stage: failureStage(current),
              message: daemonErrorMessage(error),
              retryable: true,
            },
          });
        }),
    };
  });

export type { AcnInstallationPhase, AcnInstallationPlan, AcnStartupProgress };
