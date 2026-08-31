import { FSM } from "@magnitudedev/utils";
import { Schema } from "effect";
import { AcnIdentitySchema, AcnInstanceIdSchema } from "../acn-identity";
import { AcnRevisionSchema } from "../acn-revision";

const NonEmptyString = Schema.String.pipe(Schema.minLength(1));
const NonNegativeSafeInteger = Schema.Number.pipe(
  Schema.int(),
  Schema.between(0, Number.MAX_SAFE_INTEGER)
);
const PositiveSafeInteger = NonNegativeSafeInteger.pipe(Schema.positive());

export const AcnStartupProgressSchema = Schema.Struct({
  completed: NonNegativeSafeInteger,
  totalBytes: PositiveSafeInteger,
  unit: Schema.Literal("Bytes", "Files"),
  attempt: Schema.optionalWith(PositiveSafeInteger, {
    as: "Option",
    exact: true,
  }),
});
export type AcnStartupProgress = typeof AcnStartupProgressSchema.Type;

export const AcnInstallationPhaseSchema = Schema.Literal(
  "DownloadingDaemon",
  "DownloadingInferenceEngine",
  "StartingMagnitude"
);
export type AcnInstallationPhase = typeof AcnInstallationPhaseSchema.Type;

export const AcnInstallationPlanSchema = Schema.Struct({
  daemonBytes: PositiveSafeInteger,
  inferenceEngineBytes: PositiveSafeInteger,
  inferenceEngineBytesExact: Schema.Boolean,
});
export type AcnInstallationPlan = typeof AcnInstallationPlanSchema.Type;

export const StartupBackendSchema = Schema.Union(
  Schema.TaggedStruct("Cpu", { hardwareLabel: NonEmptyString }),
  Schema.TaggedStruct("Metal", { hardwareLabel: NonEmptyString }),
  Schema.TaggedStruct("Cuda", { hardwareLabel: NonEmptyString }),
  Schema.TaggedStruct("Vulkan", { hardwareLabel: NonEmptyString }),
)
export type StartupBackend = typeof StartupBackendSchema.Type

export const AcnInstallingActivitySchema = Schema.TaggedStruct("Installing", {
  phase: AcnInstallationPhaseSchema,
  plan: AcnInstallationPlanSchema,
});
export type AcnInstallingActivity = typeof AcnInstallingActivitySchema.Type;

export const AcnStartupActivitySchema = Schema.Union(
  Schema.Literal("WaitingForOwnership", "Resolving", "Starting", "DiscoveringModels"),
  Schema.TaggedStruct("PreparingBackend", { backend: StartupBackendSchema }),
  AcnInstallingActivitySchema,
);
export type AcnStartupActivity = typeof AcnStartupActivitySchema.Type;

export const AcnStoppingReasonSchema = Schema.Literal(
  "idle",
  "administrative",
  "ownership-lost",
  "replacement",
  "icn-exited",
  "signal",
  "startup-failed",
  "fatal",
);
export type AcnStoppingReason = typeof AcnStoppingReasonSchema.Type;

export class AcnStarting extends Schema.TaggedClass<AcnStarting>()("Starting", {
  activity: AcnStartupActivitySchema,
  progress: Schema.optionalWith(AcnStartupProgressSchema, {
    as: "Option",
    exact: true,
  }),
}) {}

export class AcnReady extends Schema.TaggedClass<AcnReady>()("Ready", {}) {}

export class AcnStopping extends Schema.TaggedClass<AcnStopping>()("Stopping", {
  reason: AcnStoppingReasonSchema,
  safeDetail: Schema.optionalWith(NonEmptyString, {
    as: "Option",
    exact: true,
  }),
}) {}

export const AcnServiceLifecycleFsm = FSM.defineFSM(
  {
    Starting: AcnStarting,
    Ready: AcnReady,
    Stopping: AcnStopping,
  },
  {
    Starting: ["Ready", "Stopping"],
    Ready: ["Stopping"],
    Stopping: [],
  } as const,
);

export const AcnHealthStateSchema = Schema.Union(AcnStarting, AcnReady, AcnStopping);
export type AcnHealthState = typeof AcnHealthStateSchema.Type;

export const AcnHealthResponseSchema = Schema.Struct({
  service: Schema.Literal("magnitude-acn"),
  version: AcnIdentitySchema,
  revision: AcnRevisionSchema,
  id: AcnInstanceIdSchema,
  pid: PositiveSafeInteger,
  state: AcnHealthStateSchema,
});
export type AcnHealthResponse = typeof AcnHealthResponseSchema.Type;
