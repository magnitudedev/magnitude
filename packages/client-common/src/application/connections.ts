import { Schema } from "effect"
import { ProviderModelIdSchema } from "@magnitudedev/sdk"
import { HarnessIdSchema } from "../harness-connections/service"

export const DesktopConnectRequest = Schema.Struct({
  harness: HarnessIdSchema,
  model: Schema.optionalWith(ProviderModelIdSchema, { as: "Option", exact: true }),
})
export type DesktopConnectRequest = typeof DesktopConnectRequest.Type

export const ConnectionInspection = Schema.Union(
  Schema.TaggedStruct("Connected", {}),
  Schema.TaggedStruct("Disconnected", { reason: Schema.String }),
  Schema.TaggedStruct("Unavailable", { reason: Schema.String }),
)
export type ConnectionInspection = typeof ConnectionInspection.Type
export const DesktopHarnessConnection = Schema.Struct({
  id: HarnessIdSchema,
  name: Schema.String,
  installed: Schema.Boolean,
  managed: Schema.Boolean,
  inspection: ConnectionInspection,
  configurationFiles: Schema.Array(Schema.String),
  plugin: Schema.optionalWith(Schema.Struct({ name: Schema.String, source: Schema.String }), { as: "Option", exact: true }),
})
export type DesktopHarnessConnection = typeof DesktopHarnessConnection.Type

export const DesktopConnectionsSnapshot = Schema.Union(
  Schema.TaggedStruct("Ready", { connections: Schema.Array(DesktopHarnessConnection) }),
  Schema.TaggedStruct("Unavailable", { message: Schema.String }),
)
export type DesktopConnectionsSnapshot = typeof DesktopConnectionsSnapshot.Type
