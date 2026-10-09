import { MagnitudeHealthResponseSchema } from "@magnitudedev/acn-protocol"
export { MachineFormFactor, MachineIdentity, MachineIdentityObservation } from "@magnitudedev/acn-protocol"
import { Schema } from "effect"
import { ApplicationUpdateState, LoginStartupState } from "@magnitudedev/acn-protocol"
export { ApplicationUpdateState, LoginStartupState } from "@magnitudedev/acn-protocol"

export const AppearancePreference = Schema.Literal("system", "light", "dark")
export type AppearancePreference = typeof AppearancePreference.Type

export const ApplicationUpdateAction = Schema.Literal("status", "check", "download", "install", "discard")
export type ApplicationUpdateAction = typeof ApplicationUpdateAction.Type
export class ApplicationUpdateControlFailed extends Schema.TaggedError<ApplicationUpdateControlFailed>()("ApplicationUpdateControlFailed", { message: Schema.String }) {}
export const ApplicationUpdateRequest = Schema.Struct({ version: Schema.Literal(1), update: ApplicationUpdateAction })
export const ApplicationUpdateReply = Schema.Union(Schema.TaggedStruct("Update", { state: ApplicationUpdateState }), ApplicationUpdateControlFailed)

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

export const ApplicationIntent = Schema.Literal("EnsureRunning", "ShowWindow", "Observe", "Retry", "Quit", "Yield")
export type ApplicationIntent = typeof ApplicationIntent.Type
export const ApplicationRequest = Schema.Struct({ version: Schema.Literal(1), intent: ApplicationIntent })
export const TrayRegistration = Schema.Union(
  Schema.TaggedStruct("Checking", {}), Schema.TaggedStruct("Registered", {}),
  Schema.TaggedStruct("Unavailable", { message: Schema.String }), Schema.TaggedStruct("Closed", {}),
)
export type TrayRegistration = typeof TrayRegistration.Type
export const ApplicationOwner = Schema.Union(
  Schema.TaggedStruct("Desktop", { tray: TrayRegistration }),
  Schema.TaggedStruct("Headless", {}),
)
export type ApplicationOwner = typeof ApplicationOwner.Type
export const ApplicationSnapshot = Schema.Struct({
  version: Schema.Literal(1), pid: Schema.Int.pipe(Schema.positive()),
  endpoint: Schema.String, service: OwnedServiceState, owner: ApplicationOwner,
})
export type ApplicationSnapshot = typeof ApplicationSnapshot.Type

export const LoginStartupAction = Schema.Literal("read", "enable", "disable")
export type LoginStartupAction = typeof LoginStartupAction.Type
export class LoginStartupFailed extends Schema.TaggedError<LoginStartupFailed>()("LoginStartupFailed", { message: Schema.String }) {}
export const ApplicationLoginRequest = Schema.Struct({ version: Schema.Literal(1), login: LoginStartupAction })
export const ApplicationLoginReply = Schema.Union(Schema.TaggedStruct("LoginStartup", { state: LoginStartupState }), LoginStartupFailed)
