import { Schema } from "effect"

export const DesktopPage = Schema.Literal("discover", "catalog", "models", "connections", "usage", "status", "settings")
export type DesktopPage = typeof DesktopPage.Type
export const DesktopAction = Schema.Union(Schema.TaggedStruct("Navigate", { page: DesktopPage }), Schema.TaggedStruct("StopModel", {}))
export const ModelTrayPresentation = Schema.Struct({ label: Schema.String, canStop: Schema.Boolean })
export const DesktopApplicationInfo = Schema.Struct({ version: Schema.String })

export { DesktopConnectRequest, DesktopConnectionsSnapshot } from "./connections"
export { DesktopUpdateState } from "./update"
export { HarnessIdSchema } from "../harness-connections/service"
