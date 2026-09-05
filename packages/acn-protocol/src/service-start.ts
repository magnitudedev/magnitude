import { Option, Schema } from "effect"
import {
  AcnInstallationPhaseSchema, AcnInstallationPlanSchema, AcnStartupProgressSchema,
  StartupBackendSchema, type AcnHealthState,
} from "./schemas/acn-health"

export const ServiceStartingPhaseSchema = Schema.Union(
  Schema.Literal("PreparingAcn", "WaitingForOwner", "ResolvingLocalInference", "LaunchingLocalInference"),
  Schema.TaggedStruct("PreparingBackend", { backend: StartupBackendSchema }),
)

/** Startup observations shared by in-process hosts and public health. No ownership data. */
export const ServiceStartProgressSchema = Schema.Union(
  Schema.TaggedStruct("Starting", { phase: ServiceStartingPhaseSchema }),
  Schema.TaggedStruct("Installing", {
    phase: AcnInstallationPhaseSchema,
    plan: AcnInstallationPlanSchema,
    progress: Schema.optionalWith(AcnStartupProgressSchema, { as: "Option", exact: true }),
  }),
)
export type ServiceStartProgress = typeof ServiceStartProgressSchema.Type

export const serviceProgressFromHealth = (state: AcnHealthState): Option.Option<ServiceStartProgress> => {
  if (state._tag !== "Starting") return Option.none()
  if (typeof state.activity !== "string") return Option.some(state.activity._tag === "Installing"
    ? { _tag: "Installing", phase: state.activity.phase, plan: state.activity.plan, progress: state.progress }
    : { _tag: "Starting", phase: state.activity })
  const phases = {
    WaitingForOwnership: "WaitingForOwner",
    Resolving: "ResolvingLocalInference",
    Starting: "LaunchingLocalInference",
  } as const
  return Option.some({ _tag: "Starting", phase: phases[state.activity] })
}
