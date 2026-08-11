import { Option } from "effect"
import { describe, expect, it } from "vitest"
import {
  buildModelsMenuEntries,
  catalogLocalModels,
  localModelInstalledStatus,
  localModelReadinessStatus,
  modelsMenuEntryIsSelected,
  modelsMenuStatusPresentation,
  modelsMenuSelectionAction,
  providerDisabledStatus,
  resolveRootNavigationDirection,
} from "./container"
import {
  LOCAL_PROVIDER_ID,
  makeModel,
  makeCatalogOnlyModel,
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

  it("reports installation origin and stable assessment failures", () => {
    expect(localModelInstalledStatus(makeModel())).toContain("Installed")
    const model = withDoesNotFitAssessment(makeModel())
    expect(localModelReadinessStatus(model)).toBe("Doesn’t fit")
  })
})
