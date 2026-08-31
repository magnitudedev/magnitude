import { Option } from "effect"
import { describe, expect, it } from "vitest"
import {
  PRIMARY_SLOT_ID,
  HttpsUrlSchema,
  ProviderIdSchema,
  ProviderModelIdSchema,
  type LocalModel,
  type ModelTransferProgress,
  type ProviderModelCatalogEntry,
} from "@magnitudedev/sdk"
import {
  buildModelsMenuEntries,
  catalogInspectorActionLabel,
  catalogInspectorActions,
  catalogStatus,
  catalogLocalModels,
  huggingFaceRepositoryUrls,
  localModelInstalledStatus,
  localModelReadinessFailureMessage,
  localModelReadinessStatus,
  modelsMenuEntryIsSelected,
  modelsMenuEntryIsEligible,
  modelsMenuOrderingAtOpen,
  modelsMenuStatusPresentation,
  modelsMenuSelectionAction,
  providerDisabledStatus,
  resolveRootNavigationDirection,
} from "./container"
import {
  LOCAL_PROVIDER_ID,
  makeModel,
  makeCatalogOnlyModel,
  makeInstalledCatalogModel,
  makeView,
  TEST_MODEL_ID,
  withDoesNotFitAssessment,
} from "../local-inference/test-fixtures"

const testProgress: ModelTransferProgress = {
  stage: "downloading",
  completedBytes: 1,
  totalBytes: 4,
  bytesPerSecond: Option.none(),
}

type CatalogLocalModel = Extract<LocalModel, { readonly _tag: "Catalog" }>

const installedFieldsOf = (model: CatalogLocalModel) => {
  if (model.acquisitionState._tag !== "Installed") {
    throw new Error("expected an installed fixture")
  }
  return model.acquisitionState
}

const withUpdateAvailable = (model: CatalogLocalModel): CatalogLocalModel => ({
  ...model,
  acquisitionState: { ...installedFieldsOf(model), _tag: "UpdateAvailable" },
})

const withUpdating = (model: CatalogLocalModel): CatalogLocalModel => ({
  ...model,
  acquisitionState: { ...installedFieldsOf(model), _tag: "Updating", progress: testProgress },
})

describe("unified models menu projection", () => {
  it("handles unmodified lateral and tab navigation", () => {
    expect(resolveRootNavigationDirection({ name: "left", ctrl: false, meta: false, option: false, shift: false })).toBe(-1)
    expect(resolveRootNavigationDirection({ name: "tab", ctrl: false, meta: false, option: false, shift: true })).toBe(-1)
    expect(resolveRootNavigationDirection({ name: "right", ctrl: true, meta: false, option: false, shift: false })).toBeNull()
  })

  it("renders every provider-disabled reason", () => {
    expect(providerDisabledStatus("insufficient_resources")).toBe("Insufficient resources")
    expect(providerDisabledStatus("provider_unavailable")).toBe("Provider unavailable")
    expect(providerDisabledStatus("model_unavailable")).toBe("Model unavailable")
    expect(providerDisabledStatus("installation_unavailable")).toBe("Installation missing")
    expect(providerDisabledStatus("incompatible_runtime")).toBe("Incompatible runtime")
    expect(providerDisabledStatus("invalid_configuration")).toBe("Invalid configuration")
  })

  it("presents low free memory with a violet warning treatment", () => {
    expect(modelsMenuStatusPresentation("Low free memory")).toEqual({
      label: "! Low free memory",
      tone: "warning",
    })
  })

  it("creates one local entry for one canonical installed model", () => {
    const view = makeView()
    const entries = buildModelsMenuEntries(
      view.models.models,
      [],
      [],
    )
    expect(entries).toHaveLength(1)
    expect(entries[0]?._tag).toBe("Local")
  })

  it("excludes every uninstalled row from Models", () => {
    const installed = makeModel()
    const catalogOnly = makeCatalogOnlyModel()
    const entries = buildModelsMenuEntries(
      [installed, catalogOnly],
      [],
      [],
    )
    expect(entries).toHaveLength(1)
    expect(entries[0]).toMatchObject({ _tag: "Local", model: installed })
  })

  it("includes only fitting catalog rows in Catalog", () => {
    const catalogFit = makeCatalogOnlyModel()
    const nonCatalogFit = makeModel()
    const models = [
      catalogFit,
      nonCatalogFit,
      withDoesNotFitAssessment(makeCatalogOnlyModel()),
    ]

    expect(catalogLocalModels(models)).toEqual([catalogFit])
  })

  it("prefers authoritative download progress once it is visible", () => {
    expect(catalogStatus({
      ...makeCatalogOnlyModel(),
      acquisitionState: { _tag: "Installing", progress: testProgress },
    })).toBe("Downloading 25%")
  })

  it("shows exact catalog drift as an available update without hiding the installed model", () => {
    expect(catalogStatus(withUpdateAvailable(makeInstalledCatalogModel()))).toBe("Update available")
  })

  it("preserves the established detail actions and labels", () => {
    const available = makeCatalogOnlyModel()
    const installed = makeInstalledCatalogModel()
    const update = withUpdateAvailable(makeInstalledCatalogModel())
    const selectedSlot = makeView().slots.slots.primary
    const downloading = makeCatalogOnlyModel({
      acquisitionState: { _tag: "Installing", progress: testProgress },
    })

    expect(catalogInspectorActions(available)).toEqual(["primary"])
    expect(catalogInspectorActions(installed)).toEqual(["select", "remove"])
    expect(catalogInspectorActions(update)).toEqual(["select", "primary", "remove"])
    expect(catalogInspectorActions(installed, true, selectedSlot)).toEqual(["stop", "remove"])
    expect(catalogInspectorActions(update, true, selectedSlot)).toEqual(["stop", "primary", "remove"])
    expect(catalogInspectorActions(downloading)).toEqual(["cancel"])
    expect(catalogInspectorActionLabel("primary", available)).toBe("Download")
    expect(catalogInspectorActionLabel("primary", update)).toBe("Update")
    expect(catalogInspectorActionLabel("select", installed)).toBe("Select model")
    expect(catalogInspectorActionLabel("cancel", downloading)).toBe("Cancel download")
    expect(catalogInspectorActionLabel("stop", installed, selectedSlot)).toBe("Stop model")
  })

  it("lists catalog source repositories without exposing their packages", () => {
    const model = makeCatalogOnlyModel({
      presentation: {
        ...makeCatalogOnlyModel().presentation,
        sourceUrls: [
          HttpsUrlSchema.make("https://huggingface.co/publisher/target"),
          HttpsUrlSchema.make("https://huggingface.co/publisher/draft"),
        ],
      },
    })

    expect(huggingFaceRepositoryUrls(model)).toEqual([
      "https://huggingface.co/publisher/target",
      "https://huggingface.co/publisher/draft",
    ])
  })

  it("lists a discovered Hugging Face repository", () => {
    const model = makeCatalogOnlyModel({
      presentation: {
        ...makeCatalogOnlyModel().presentation,
        sourceUrls: [HttpsUrlSchema.make("https://huggingface.co/publisher/target")],
      },
    })

    expect(huggingFaceRepositoryUrls(model)).toEqual([
      "https://huggingface.co/publisher/target",
    ])
  })

  it("uses embedded local availability for selection and identity", () => {
    const view = makeView()
    const [entry] = buildModelsMenuEntries(
      view.models.models,
      [],
      [],
    )
    if (entry === undefined) throw new Error("entry missing")
    expect(modelsMenuSelectionAction(entry)).toMatchObject({
      _tag: "Some",
      value: { _tag: "AssignLocal", providerModelId: TEST_MODEL_ID },
    })
    expect(modelsMenuEntryIsSelected(entry, Option.some({
      providerId: LOCAL_PROVIDER_ID,
      providerModelId: TEST_MODEL_ID,
    }))).toBe(true)
  })

  it("keeps eligibility tied to the selection captured when the menu opened", () => {
    const local = makeModel()
    if (local.state.servingState._tag !== "Assessed") throw new Error("fixture must be assessed")
    const provider = {
      providerId: ProviderIdSchema.make("test"),
      displayName: "Test",
      kind: "Hosted" as const,
      authentication: "NotRequired" as const,
      availability: { _tag: "Available" as const },
    }
    const unavailable = {
      providerId: provider.providerId,
      providerModelId: ProviderModelIdSchema.make("unavailable"),
      modelFamilyId: Option.none(),
      displayName: "Unavailable",
      variantLabel: Option.none(),
      supportedSlots: [PRIMARY_SLOT_ID],
      contextWindow: 1,
      maxOutputTokens: 1,
      memory: Option.none(),
      capabilities: local.state.servingState.capabilities,
      availability: { _tag: "Disabled" as const, reason: "model_unavailable" as const },
      pricing: Option.none(),
    } satisfies ProviderModelCatalogEntry
    const entry = {
      _tag: "Provider" as const,
      id: "test:unavailable",
      model: unavailable,
      provider,
    }

    expect(modelsMenuEntryIsEligible(entry, Option.some(unavailable))).toBe(true)
    expect(modelsMenuEntryIsEligible(entry, Option.none())).toBe(false)
  })

  it("waits for complete ordering inputs before capturing the selected model", () => {
    const view = makeView()
    const selected = Option.some({
      providerId: LOCAL_PROVIDER_ID,
      providerModelId: TEST_MODEL_ID,
    })

    expect(Option.isNone(modelsMenuOrderingAtOpen(
      false,
      true,
      Option.some(view.slots),
      selected,
    ))).toBe(true)

    const ordering = Option.getOrThrow(modelsMenuOrderingAtOpen(
      true,
      true,
      Option.some(view.slots),
      selected,
    ))
    expect(Option.getOrThrow(ordering.selectedModel).providerModelId).toBe(TEST_MODEL_ID)

    expect(modelsMenuOrderingAtOpen(true, true, Option.none(), Option.none())).toMatchObject({
      _tag: "Some",
      value: { recentModelKeys: [], favoriteKeys: new Set() },
    })
  })

  it("reports installation origin and stable assessment failures", () => {
    expect(localModelInstalledStatus(makeModel())).toContain("Installed")
    const model = withDoesNotFitAssessment(makeModel())
    expect(localModelReadinessStatus(model)).toBe("Doesn’t fit")
  })

  it("preserves the exact discovered-model failure for the details view", () => {
    const model = makeModel()
    const failed: LocalModel = {
      ...model,
      state: {
        ...model.state,
        servingState: {
          _tag: "Failed",
          profile: model.state.servingState._tag === "Assessed"
            ? model.state.servingState.assessment.profile
            : { contextLength: 32_768 },
          failure: {
            code: "invalid_artifact",
            message: "The GGUF artifact does not contain a complete model.",
            retryable: false,
          },
        },
      },
    }

    expect(localModelReadinessFailureMessage(failed)).toBe(
      "The GGUF artifact does not contain a complete model.",
    )
  })

  it("renders catalog upgrade state without inferring artifact differences", () => {
    expect(localModelReadinessStatus(withUpdateAvailable(makeInstalledCatalogModel()))).toBe("Update available")
    const model = withUpdating(makeInstalledCatalogModel())
    expect(localModelReadinessStatus(model)).toBe("Updating")
    expect(catalogStatus(model)).toBe("Updating 25%")
  })
})
