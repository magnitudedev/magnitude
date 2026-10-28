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

export const AcnInstallingActivitySchema = Schema.TaggedStruct("Installing", {
  phase: AcnInstallationPhaseSchema,
  plan: AcnInstallationPlanSchema,
});
export type AcnInstallingActivity = typeof AcnInstallingActivitySchema.Type;

export const AcnStartupActivitySchema = Schema.Union(
  Schema.Literal("WaitingForOwnership", "Resolving", "Starting"),
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

/** A step of startup that failed and runs again at `retryAtMillis`, such as an engine download during an outage. */
export const AcnStartupRetrySchema = Schema.Struct({
  message: NonEmptyString,
  retryAtMillis: NonNegativeSafeInteger,
});
export type AcnStartupRetry = typeof AcnStartupRetrySchema.Type;

export class AcnStarting extends Schema.TaggedClass<AcnStarting>()("Starting", {
  activity: AcnStartupActivitySchema,
  progress: Schema.optionalWith(AcnStartupProgressSchema, {
    as: "Option",
    exact: true,
  }),
  retry: Schema.optionalWith(AcnStartupRetrySchema, {
    as: "Option",
    exact: true,
  }),
}) {}

export const AcnHardwareSummarySchema = Schema.Struct({
  /** The active device's backend: a resident model's device, and otherwise the serving device. */
  backend: Schema.Literal("cpu", "metal", "cuda", "vulkan"),
  gpuName: Schema.optionalWith(NonEmptyString, { as: "Option", exact: true }),
  gpuKind: Schema.optionalWith(Schema.Literal("discrete", "integrated"), { as: "Option", exact: true }),
  gpuMemoryBytes: Schema.optionalWith(NonNegativeSafeInteger, { as: "Option", exact: true }),
  gpuCount: NonNegativeSafeInteger,
  gpuUnusedReason: Schema.optionalWith(NonEmptyString, { as: "Option", exact: true }),
  cpuModel: Schema.optionalWith(NonEmptyString, { as: "Option", exact: true }),
  cpuCores: Schema.optionalWith(PositiveSafeInteger, { as: "Option", exact: true }),
  memoryBytes: NonNegativeSafeInteger,
})
export type AcnHardwareSummary = typeof AcnHardwareSummarySchema.Type

export class AcnReady extends Schema.TaggedClass<AcnReady>()("Ready", {
  hardware: Schema.optionalWith(AcnHardwareSummarySchema, { as: "Option", exact: true }),
}) {}

/** `permanent` marks a stop that starting again cannot fix, such as an engine release the host doesn't have. */
export class AcnStopping extends Schema.TaggedClass<AcnStopping>()("Stopping", {
  reason: AcnStoppingReasonSchema,
  safeDetail: Schema.optionalWith(NonEmptyString, {
    as: "Option",
    exact: true,
  }),
  permanent: Schema.optionalWith(Schema.Boolean, {
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

/** Application admission metadata; not a new private coordination prerequisite. */
export const MagnitudeHealthResponseSchema = Schema.Struct({
  ...AcnHealthResponseSchema.fields,
  rpcVersion: PositiveSafeInteger,
});
export type MagnitudeHealthResponse = typeof MagnitudeHealthResponseSchema.Type;
