import { Schema } from "effect";
import { AcnOwnerIdSchema } from "../acn-registry";

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

export const AcnStartupActivitySchema = Schema.Literal(
  "WaitingForOwnership",
  "Resolving",
  "Starting"
);
export type AcnStartupActivity = typeof AcnStartupActivitySchema.Type;

export const AcnHealthStateSchema = Schema.Union(
  Schema.TaggedStruct("Starting", {
    activity: AcnStartupActivitySchema,
    progress: Schema.optionalWith(AcnStartupProgressSchema, {
      as: "Option",
      exact: true,
    }),
  }),
  Schema.TaggedStruct("Installing", {
    phase: AcnInstallationPhaseSchema,
    plan: AcnInstallationPlanSchema,
    progress: Schema.optionalWith(AcnStartupProgressSchema, {
      as: "Option",
      exact: true,
    }),
  }),
  Schema.TaggedStruct("Ready", {}),
  Schema.TaggedStruct("Failed", {
    message: NonEmptyString,
    retryable: Schema.Boolean,
  })
);
export type AcnHealthState = typeof AcnHealthStateSchema.Type;

export const AcnHealthResponseSchema = Schema.Struct({
  service: Schema.Literal("magnitude-acn"),
  version: NonEmptyString,
  id: AcnOwnerIdSchema,
  pid: PositiveSafeInteger,
  state: AcnHealthStateSchema,
});
export type AcnHealthResponse = typeof AcnHealthResponseSchema.Type;
