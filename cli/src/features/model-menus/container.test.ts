import { Option } from "effect"
import { describe, expect, it } from "vitest"
import {
  ModelDownloadIdSchema,
  PRIMARY_SLOT_ID,
  ProviderIdSchema,
  ProviderModelIdSchema,
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
  makeStandaloneBundle,
  makeView,
  TEST_MODEL_ID,
  withDoesNotFitAssessment,
} from "../local-inference/test-fixtures"

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
    const nonCatalogFit = makeModel({
      acquisitionState: { _tag: "NotInstalled", completedBytes: 0, totalBytes: 1 },
    })
    const models = [
      catalogFit,
      nonCatalogFit,
      withDoesNotFitAssessment(makeCatalogOnlyModel()),
    ]

    expect(catalogLocalModels(models)).toEqual([catalogFit])
  })

  it("shows download admission before mirrored acquisition begins", () => {
    expect(catalogStatus(makeCatalogOnlyModel(), {
      _tag: "Starting",
      operation: "Install",
    })).toBe("Starting download…")
    expect(catalogStatus(makeModel({ upgradeState: { _tag: "Available" } }), {
      _tag: "Starting",
      operation: "Update",
    })).toBe("Starting update…")
  })

  it("prefers authoritative download progress once it is visible", () => {
    expect(catalogStatus({
      ...makeCatalogOnlyModel(),
      acquisitionState: {
        _tag: "Downloading",
        downloadId: ModelDownloadIdSchema.make("download-a"),
        stage: "downloading",
        completedBytes: 1,
        totalBytes: 4,
        bytesPerSecond: Option.none(),
      },
    }, {
      _tag: "Transferring",
      operation: "Install",
      downloadId: ModelDownloadIdSchema.make("download-a"),
      stage: "downloading",
      completedBytes: 1,
      totalBytes: 4,
      bytesPerSecond: Option.none(),
    })).toBe("Downloading 25%")
  })

  it("shows exact catalog drift as an available update without hiding the installed model", () => {
    expect(catalogStatus({
      ...makeModel(),
      upgradeState: { _tag: "Available" },
    })).toBe("Update available")
  })

  it("preserves the established detail actions and labels", () => {
    const available = makeCatalogOnlyModel()
    const installed = makeModel()
    const update = makeModel({ upgradeState: { _tag: "Available" } })
    const selectedSlot = makeView().slots.slots.primary
    const downloading = makeCatalogOnlyModel({
      acquisitionState: {
        _tag: "Downloading",
        downloadId: ModelDownloadIdSchema.make("download-a"),
        stage: "downloading",
        completedBytes: 1,
        totalBytes: 4,
        bytesPerSecond: Option.none(),
      },
    })

    expect(catalogInspectorActions(available, { _tag: "Idle" })).toEqual(["primary"])
    expect(catalogInspectorActions(installed, { _tag: "Idle" })).toEqual(["select", "uninstall"])
    expect(catalogInspectorActions(update, { _tag: "Idle" })).toEqual(["select", "primary", "uninstall"])
    expect(catalogInspectorActions(installed, { _tag: "Idle" }, true, selectedSlot)).toEqual(["stop", "uninstall"])
    expect(catalogInspectorActions(update, { _tag: "Idle" }, true, selectedSlot)).toEqual(["stop", "primary", "uninstall"])
    expect(catalogInspectorActions(downloading, { _tag: "Idle" })).toEqual(["cancel"])
    expect(catalogInspectorActions(available, {
      _tag: "Starting",
      operation: "Install",
    })).toEqual([])
    expect(catalogInspectorActionLabel("primary", available)).toBe("Download (16.0 GiB)")
    expect(catalogInspectorActionLabel("primary", update)).toBe("Update")
    expect(catalogInspectorActionLabel("select", installed)).toBe("Select model")
    expect(catalogInspectorActionLabel("cancel", downloading)).toBe("Cancel download")
    expect(catalogInspectorActionLabel("stop", installed, selectedSlot)).toBe("Stop model")
  })

  it("lists target and separate-draft repositories without treating the draft source as a package", () => {
    const target = makeStandaloneBundle("package_target")
    const draft = makeStandaloneBundle("package_draft")
    if (target._tag !== "Standalone" || draft._tag !== "Standalone") {
      throw new Error("test bundles must contain standalone packages")
    }
    const model = makeCatalogOnlyModel({
      bundle: {
        _tag: "SpeculativeDecoding",
        target: {
          ...target.package,
          source: { _tag: "HuggingFace", repository: "publisher/target", revision: "a".repeat(40) },
        },
        draftSource: {
          _tag: "Separate",
          draft: {
            ...draft.package,
            source: { _tag: "HuggingFace", repository: "publisher/draft", revision: "b".repeat(40) },
          },
        },
        method: { _tag: "DSpark" },
      },
    })

    expect(huggingFaceRepositoryUrls(model)).toEqual([
      "https://huggingface.co/publisher/target",
      "https://huggingface.co/publisher/draft",
    ])
  })

  it("lists only the target repository for embedded speculative decoding", () => {
    const target = makeStandaloneBundle("package_target")
    if (target._tag !== "Standalone") throw new Error("test bundle must contain a package")
    const model = makeCatalogOnlyModel({
      bundle: {
        _tag: "SpeculativeDecoding",
        target: {
          ...target.package,
          source: { _tag: "HuggingFace", repository: "publisher/target", revision: "a".repeat(40) },
        },
        draftSource: { _tag: "Embedded" },
        method: { _tag: "Mtp" },
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
    if (local.servingState._tag !== "Assessed") throw new Error("fixture must be assessed")
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
      capabilities: local.servingState.capabilities,
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

  it("renders catalog upgrade state without inferring artifact differences", () => {
    expect(localModelReadinessStatus(makeModel({
      upgradeState: { _tag: "Available" },
    }))).toBe("Update available")
    const model = makeModel({
      upgradeState: {
        _tag: "Upgrading",
        downloadId: ModelDownloadIdSchema.make("download-upgrade"),
        stage: "downloading",
        completedBytes: 1,
        totalBytes: 4,
        bytesPerSecond: Option.none(),
      },
    })
    expect(localModelReadinessStatus(model)).toBe("Updating")
    expect(catalogStatus(model, {
      _tag: "Transferring",
      operation: "Update",
      downloadId: ModelDownloadIdSchema.make("download-upgrade"),
      stage: "downloading",
      completedBytes: 1,
      totalBytes: 4,
      bytesPerSecond: Option.none(),
    })).toBe("Updating 25%")
  })
})
