import { Schema } from "effect"

const optional = <A, I, R>(schema: Schema.Schema<A, I, R>) => Schema.optionalWith(schema, { as: "Option", exact: true })

/** Where downloaded models are stored. The engine reads the folder once when the service starts. */
export const ModelStorageSettings = Schema.Struct({
  active: Schema.String,
  path: Schema.String,
  source: Schema.Literal("Default", "Configured"),
  defaultPath: Schema.String,
  warning: optional(Schema.String),
})
export type ModelStorageSettings = typeof ModelStorageSettings.Type

export const ModelStorageChange = Schema.Struct({ path: optional(Schema.String) })
export type ModelStorageChange = typeof ModelStorageChange.Type

export const DirectoryEntry = Schema.Struct({ name: Schema.String, path: Schema.String })
export const DirectoryListing = Schema.Struct({
  path: Schema.String,
  parent: optional(Schema.String),
  directories: Schema.Array(DirectoryEntry),
})
export type DirectoryListing = typeof DirectoryListing.Type

export const NetworkInterfaceAddress = Schema.Struct({
  name: Schema.String,
  address: Schema.String,
  kind: Schema.Literal("lan", "tailscale", "virtual"),
})
export type NetworkInterfaceAddress = typeof NetworkInterfaceAddress.Type

export const NetworkAccessSettings = Schema.Struct({
  enabled: Schema.Boolean,
  bind: optional(Schema.String),
  requireApiKey: Schema.Boolean,
  apiKey: optional(Schema.String),
  interfaces: Schema.Array(NetworkInterfaceAddress),
  port: Schema.Int,
  pending: Schema.Boolean,
  warning: optional(Schema.String),
})
export type NetworkAccessSettings = typeof NetworkAccessSettings.Type

/** All interfaces, or one IP address of the machine running Magnitude. */
export const NetworkBind = Schema.Union(
  Schema.TaggedStruct("AllInterfaces", {}),
  Schema.TaggedStruct("Address", { address: Schema.String }),
)
export type NetworkBind = typeof NetworkBind.Type
export const NetworkAccessChange = Schema.Struct({
  enabled: optional(Schema.Boolean),
  bind: optional(NetworkBind),
  requireApiKey: optional(Schema.Boolean),
})
export type NetworkAccessChange = typeof NetworkAccessChange.Type

/** Host enclosure identity supplements, but never determines, inference capabilities. */
export const MachineFormFactor = Schema.Literal("Portable", "Desktop", "AllInOne", "MiniPc", "Server", "Unknown")
export type MachineFormFactor = typeof MachineFormFactor.Type
const FirmwareLabel = Schema.Trimmed.pipe(Schema.minLength(1), Schema.maxLength(255))
export const MachineIdentity = Schema.Struct({
  manufacturer: Schema.Trimmed.pipe(Schema.minLength(1), Schema.maxLength(255)),
  model: Schema.Trimmed.pipe(Schema.minLength(1), Schema.maxLength(255)),
  family: optional(FirmwareLabel),
  version: optional(FirmwareLabel),
  formFactor: MachineFormFactor,
})
export const MachineIdentityObservation = Schema.Union(
  Schema.TaggedStruct("Identified", MachineIdentity.fields),
  Schema.TaggedStruct("Unavailable", { formFactor: MachineFormFactor }),
)
export type MachineIdentityObservation = typeof MachineIdentityObservation.Type

/** The machine running Magnitude: its OS for commands run there, and its enclosure. */
export const ServerMachine = Schema.Struct({
  platform: Schema.String,
  identity: MachineIdentityObservation,
})
export type ServerMachine = typeof ServerMachine.Type

export class ServerSettingsFailed extends Schema.TaggedError<ServerSettingsFailed>()("ServerSettingsFailed", {
  message: Schema.String,
}) {}
