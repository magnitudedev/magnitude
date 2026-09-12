import { MagnitudeHealthResponseSchema } from "@magnitudedev/acn-protocol"
import { Schema } from "effect"

export class Starting extends Schema.TaggedClass<Starting>()("Starting", {
  attempt: Schema.Int,
  health: Schema.optionalWith(MagnitudeHealthResponseSchema, { as: "Option", exact: true }),
}) {}
export class Ready extends Schema.TaggedClass<Ready>()("Ready", { health: MagnitudeHealthResponseSchema }) {}
export class Failed extends Schema.TaggedClass<Failed>()("Failed", { message: Schema.String }) {}
export class CleanupFailed extends Schema.TaggedClass<CleanupFailed>()("CleanupFailed", { message: Schema.String }) {}
export class Stopping extends Schema.TaggedClass<Stopping>()("Stopping", {}) {}
export class Stopped extends Schema.TaggedClass<Stopped>()("Stopped", {}) {}

export const OwnedServiceState = Schema.Union(Starting, Ready, Failed, CleanupFailed, Stopping, Stopped)
export type OwnedServiceState = typeof OwnedServiceState.Type

export const ApplicationIntent = Schema.Literal("EnsureRunning", "ShowWindow", "Observe", "Retry", "Quit")
export type ApplicationIntent = typeof ApplicationIntent.Type
export const ApplicationRequest = Schema.Struct({ version: Schema.Literal(1), intent: ApplicationIntent })
export const TrayRegistration = Schema.Union(
  Schema.TaggedStruct("Checking", {}), Schema.TaggedStruct("Registered", {}),
  Schema.TaggedStruct("Unavailable", { message: Schema.String }), Schema.TaggedStruct("Closed", {}),
)
export type TrayRegistration = typeof TrayRegistration.Type
export const ApplicationSnapshot = Schema.Struct({
  version: Schema.Literal(1), pid: Schema.Int.pipe(Schema.positive()),
  endpoint: Schema.String, service: OwnedServiceState, tray: TrayRegistration,
})
export type ApplicationSnapshot = typeof ApplicationSnapshot.Type

export const LoginStartupState = Schema.Union(
  Schema.TaggedStruct("Enabled", {}),
  Schema.TaggedStruct("Disabled", {}),
  Schema.TaggedStruct("RequiresApproval", {}),
  Schema.TaggedStruct("Unavailable", { message: Schema.String }),
)
export type LoginStartupState = typeof LoginStartupState.Type
export const LoginStartupAction = Schema.Literal("read", "enable", "disable")
export type LoginStartupAction = typeof LoginStartupAction.Type
export class LoginStartupFailed extends Schema.TaggedError<LoginStartupFailed>()("LoginStartupFailed", { message: Schema.String }) {}
export const ApplicationLoginRequest = Schema.Struct({ version: Schema.Literal(1), login: LoginStartupAction })
export const ApplicationLoginReply = Schema.Union(Schema.TaggedStruct("LoginStartup", { state: LoginStartupState }), LoginStartupFailed)

/** OS-attributed process memory, never catalog estimates or whole-machine used RAM. */
export const ApplicationMemorySample = Schema.Struct({
  bytes: Schema.Int.pipe(Schema.between(0, Number.MAX_SAFE_INTEGER)),
  processCount: Schema.Int.pipe(Schema.between(1, 512)),
  metric: Schema.Literal("PhysicalFootprint", "ProportionalResident", "PrivateWorkingSet"),
})
export const ApplicationMemoryObservation = Schema.Union(
  Schema.TaggedStruct("Measured", {
    ...ApplicationMemorySample.fields,
    measuredAt: Schema.Int.pipe(Schema.between(0, Number.MAX_SAFE_INTEGER)),
  }),
  Schema.TaggedStruct("Unavailable", { message: Schema.String }),
)
export type ApplicationMemoryObservation = typeof ApplicationMemoryObservation.Type

/** Host enclosure identity supplements, but never determines, inference capabilities. */
export const MachineIdentity = Schema.Struct({
  manufacturer: Schema.Trimmed.pipe(Schema.minLength(1), Schema.maxLength(255)),
  model: Schema.Trimmed.pipe(Schema.minLength(1), Schema.maxLength(255)),
})
export const MachineIdentityObservation = Schema.Union(
  Schema.TaggedStruct("Identified", MachineIdentity.fields),
  Schema.TaggedStruct("Unavailable", {}),
)
export type MachineIdentityObservation = typeof MachineIdentityObservation.Type
