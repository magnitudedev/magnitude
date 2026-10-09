import { Schema } from "effect"

/** Application updates belong to the owning application, independently of ACN readiness. */
const Transfer = Schema.Union(
  Schema.TaggedStruct("Idle", {}),
  Schema.TaggedStruct("Available", { version: Schema.String, bytes: Schema.Number }),
  Schema.TaggedStruct("Downloading", { version: Schema.String, completed: Schema.Number, total: Schema.Number }),
  Schema.TaggedStruct("Cancelling", {}),
  Schema.TaggedStruct("Staging", { version: Schema.String }),
  Schema.TaggedStruct("Ready", { version: Schema.String }),
  Schema.TaggedStruct("InstallationFailed", { version: Schema.String, message: Schema.String }),
  Schema.TaggedStruct("Failed", { message: Schema.String }),
  Schema.TaggedStruct("Unavailable", { message: Schema.String }),
  Schema.TaggedStruct("Closed", {}),
)
export const ApplicationUpdateState = Schema.Struct({
  transfer: Transfer,
  check: Schema.Union(
    Schema.TaggedStruct("Idle", {}),
    Schema.TaggedStruct("Checking", {}),
    Schema.TaggedStruct("Succeeded", { at: Schema.Number }),
    Schema.TaggedStruct("Failed", { message: Schema.String }),
  ),
  preference: Schema.Union(
    Schema.TaggedStruct("Known", { autoDownload: Schema.Boolean }),
    Schema.TaggedStruct("Unavailable", { message: Schema.String }),
  ),
})
export type ApplicationUpdateState = typeof ApplicationUpdateState.Type

export const LoginStartupState = Schema.Union(
  Schema.TaggedStruct("Enabled", {}),
  Schema.TaggedStruct("Disabled", {}),
  Schema.TaggedStruct("RequiresApproval", {}),
  Schema.TaggedStruct("Unavailable", { message: Schema.String }),
)
export type LoginStartupState = typeof LoginStartupState.Type

/** The process that owns ACN: the desktop app, or `magnitude serve`. */
export const ApplicationOwnerKind = Schema.Literal("Desktop", "Headless")
export type ApplicationOwnerKind = typeof ApplicationOwnerKind.Type
export const OwnerCapability = Schema.Literal("Updates", "LaunchAtLogin", "RestartService", "Quit")
export type OwnerCapability = typeof OwnerCapability.Type

/** Work in progress on the service. The owner installs an update only when both counts are zero. */
export const ServiceActivity = Schema.Struct({
  /** Sessions whose agents are working on a turn. */
  workingSessions: Schema.NonNegativeInt,
  /** Inference requests from harnesses and other devices whose responses are still streaming. */
  inferenceRequests: Schema.NonNegativeInt,
})
export type ServiceActivity = typeof ServiceActivity.Type

/** What the owner reports about itself; it sends a new state whenever any part changes. */
export const ApplicationOwnerState = Schema.Struct({
  owner: ApplicationOwnerKind,
  capabilities: Schema.Array(OwnerCapability),
  updates: Schema.optionalWith(ApplicationUpdateState, { as: "Option", exact: true }),
  loginStartup: Schema.optionalWith(LoginStartupState, { as: "Option", exact: true }),
})
export type ApplicationOwnerState = typeof ApplicationOwnerState.Type

export const OwnerRequest = Schema.Union(
  Schema.TaggedStruct("CheckUpdate", {}),
  Schema.TaggedStruct("DownloadUpdate", {}),
  Schema.TaggedStruct("DiscardUpdate", {}),
  /** Installs a prepared update and restarts the application. */
  Schema.TaggedStruct("InstallUpdate", {}),
  Schema.TaggedStruct("SetAutoDownload", { enabled: Schema.Boolean }),
  Schema.TaggedStruct("SetLoginStartup", { enabled: Schema.Boolean }),
  /** Restarts the whole application, which applies settings read at startup. */
  Schema.TaggedStruct("RestartService", {}),
  Schema.TaggedStruct("Quit", {}),
)
export type OwnerRequest = typeof OwnerRequest.Type

export const OwnerReply = Schema.Union(
  Schema.TaggedStruct("Done", {}),
  Schema.TaggedStruct("Unsupported", {}),
  Schema.TaggedStruct("Failed", { message: Schema.String }),
)
export type OwnerReply = typeof OwnerReply.Type

export class ApplicationOwnerUnavailable extends Schema.TaggedError<ApplicationOwnerUnavailable>()("ApplicationOwnerUnavailable", {
  message: Schema.String,
}) {}
export class OwnerRequestUnsupported extends Schema.TaggedError<OwnerRequestUnsupported>()("OwnerRequestUnsupported", {
  request: Schema.String,
}) {}
export class OwnerRequestFailed extends Schema.TaggedError<OwnerRequestFailed>()("OwnerRequestFailed", {
  message: Schema.String,
}) {}
