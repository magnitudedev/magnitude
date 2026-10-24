import { Schema } from "effect"
import { HuggingFaceArtifactSelectorSchema, HuggingFaceRepositoryIdSchema } from "@magnitudedev/sdk/desktop-host"

export const ApplicationPage = Schema.Literal("discover", "catalog", "models", "chat", "connections", "usage", "status", "settings")
export type ApplicationPage = typeof ApplicationPage.Type
/** A message raised outside the renderer, shown in the app while its window exists. */
export const HostNotice = Schema.Struct({ title: Schema.String, description: Schema.String })
export type HostNotice = typeof HostNotice.Type
export const HostAction = Schema.Union(
  Schema.TaggedStruct("Navigate", { page: ApplicationPage }),
  Schema.TaggedStruct("StopModel", {}),
  Schema.TaggedStruct("ShowNotice", { notice: HostNotice }),
  Schema.TaggedStruct("QuitFailed", {}),
  Schema.TaggedStruct("OpenHuggingFaceModel", {
    repository: HuggingFaceRepositoryIdSchema,
    file: Schema.optionalWith(HuggingFaceArtifactSelectorSchema, { as: "Option", exact: true }),
  }),
)
export type HostAction = typeof HostAction.Type
/** What the Add model dialog opens on: a Hugging Face repository (optionally one file in it), or a search. */
export const AddModelRequest = Schema.Union(
  Schema.TaggedStruct("HuggingFace", {
    repository: HuggingFaceRepositoryIdSchema,
    file: Schema.optionalWith(HuggingFaceArtifactSelectorSchema, { as: "Option", exact: true }),
  }),
  Schema.TaggedStruct("Search", { query: Schema.String }),
)
export type AddModelRequest = typeof AddModelRequest.Type
export const QuitFailureDecision = Schema.Literal("KeepOpen", "RetryQuit", "ForceQuit")
export type QuitFailureDecision = typeof QuitFailureDecision.Type
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
