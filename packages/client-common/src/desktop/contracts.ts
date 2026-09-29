import { Schema } from "effect"

export { DesktopConnectRequest, DesktopConnectionsSnapshot } from "./connections"
export { HarnessIdSchema } from "../harness-connections/service"

export const DesktopPage = Schema.Literal("discover", "catalog", "models", "connections", "usage", "status", "settings")
export type DesktopPage = typeof DesktopPage.Type
export const DesktopAction = Schema.Union(Schema.TaggedStruct("Navigate", { page: DesktopPage }), Schema.TaggedStruct("StopModel", {}))
/** What the tray row shows beside an active model's phase. */
export const ModelTrayDetail = Schema.Union(
  /** Work with no measure of its own: a spinner. */
  Schema.TaggedStruct("Working", {}),
  Schema.TaggedStruct("Progress", { fraction: Schema.Number.pipe(Schema.finite(), Schema.between(0, 1)) }),
  /** The memory a loaded model holds. */
  Schema.TaggedStruct("Memory", { text: Schema.String }),
)
export const ModelTrayStatus = Schema.Struct({ model: Schema.String, phase: Schema.String, detail: ModelTrayDetail })
/**
 * The tray's model line: `label` as plain text, and for an active model its `status` for a live
 * native row where the platform has one.
 */
export const ModelTrayPresentation = Schema.Struct({
  label: Schema.String,
  status: Schema.optionalWith(ModelTrayStatus, { as: "Option", exact: true }),
  canStop: Schema.Boolean,
})
export const DesktopApplicationInfo = Schema.Struct({ version: Schema.String })
