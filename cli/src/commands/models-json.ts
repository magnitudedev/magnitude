import { formatLocalModelDisplayName } from "@magnitudedev/client-common"
import {
  ModelIdSchema, IntegrationModelIdSchema, type LocalModel,
  JsonInstallationStateSchema, JsonResidencyStateSchema, JsonLocalModelSchema,
  ModelsStatusJsonDataSchema, type JsonLocalModel, type ModelsStatusJsonData,
} from "@magnitudedev/sdk"
import { Schema } from "effect"

export { JsonInstallationStateSchema, JsonResidencyStateSchema, JsonLocalModelSchema, ModelsStatusJsonDataSchema }
export { ModelsLoadJsonDataSchema, ModelsStopJsonDataSchema } from "@magnitudedev/sdk"
export type { JsonLocalModel, ModelsStatusJsonData }

export const modelsLoadJsonData = (modelId: typeof ModelIdSchema.Type) => ({ modelId: IntegrationModelIdSchema.make(modelId) })

export const modelsStopJsonData = () => ({})

export type ModelsStatusResult =
  | { readonly _tag: "Initializing" }
  | { readonly _tag: "List"; readonly models: readonly LocalModel[] }
  | { readonly _tag: "Detail"; readonly model: LocalModel }

export const modelsForStatus = (models: readonly LocalModel[]): LocalModel[] => models
  .filter((model) => model._tag === "Discovered" || model.acquisitionState._tag !== "NotInstalled")
  .sort((left, right) => formatLocalModelDisplayName(left).localeCompare(formatLocalModelDisplayName(right))
    || left.modelId.localeCompare(right.modelId))

const installationState = (model: LocalModel): typeof JsonInstallationStateSchema.Type => {
  if (model._tag === "Discovered") return model.state._tag === "Ready" ? "installed" : "unavailable"
  switch (model.acquisitionState._tag) {
    case "NotInstalled": return "not_installed"
    case "Installing": return "installing"
    case "InstallFailed": return "unavailable"
    case "Installed":
    case "UpdateAvailable":
    case "Updating":
    case "UpdateFailed": return "installed"
    case "Removing": return "removing"
    case "RemoveFailed": return "unavailable"
  }
}

const residencyState = (model: LocalModel): typeof JsonResidencyStateSchema.Type | undefined => {
  const state = model._tag === "Discovered"
    ? model.state._tag === "Ready" ? model.state.residencyState : undefined
    : "residencyState" in model.acquisitionState ? model.acquisitionState.residencyState : undefined
  if (state === undefined) return undefined
  switch (state._tag) {
    case "Unloaded": return "unloaded"
    case "Requested":
    case "Loading": return "loading"
    case "Ready": return "ready"
    case "Stopping": return "stopping"
    case "Failed": return "failed"
  }
}

export const localModelJson = (model: LocalModel): JsonLocalModel => {
  const residency = residencyState(model)
  return Schema.decodeUnknownSync(JsonLocalModelSchema)({
    modelId: model.modelId,
    displayName: formatLocalModelDisplayName(model),
    installation: installationState(model),
    ...(residency === undefined ? {} : { residency }),
  })
}

export const modelsStatusJsonData = (result: ModelsStatusResult): ModelsStatusJsonData => {
  switch (result._tag) {
    case "Initializing": return { state: "initializing", models: [] }
    case "List": return { state: "ready", models: modelsForStatus(result.models).map(localModelJson) }
    case "Detail": return { state: "ready", models: [localModelJson(result.model)] }
  }
}
