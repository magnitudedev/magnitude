import { describe, expect, test } from "vitest"
import type { KeyEvent } from "@opentui/core"
import { Option } from "effect"
import {
  PRIMARY_SLOT_ID,
  SECONDARY_SLOT_ID,
  ProviderIdSchema,
  ProviderModelCatalogLifecycle,
  ProviderModelIdSchema,
  ReasoningEffortSchema,
} from "@magnitudedev/sdk"
import {
  buildModelsMenuEntries,
  huggingFaceRepositoryUrls,
  localModelInstalledStatus,
  localModelReadinessStatus,
  modelsMenuEntryIsSelected,
  modelsMenuSelectionAction,
  providerDisabledStatus,
  resolveRootNavigationDirection,
  scrollCatalogCandidateIntoView,
} from "./container"
import {
  makeCatalogCandidate,
  makeCatalogOnlyModel,
  makeModel,
  makeView,
  TEST_MODEL_ID,
} from "../local-inference/test-fixtures"
import { buildInstalledLocalModelChoices } from "../local-inference/view-model"

const key = (
  name: string,
  overrides: Partial<Pick<KeyEvent, "ctrl" | "meta" | "option" | "shift">> = {},
): Pick<KeyEvent, "name" | "ctrl" | "meta" | "option" | "shift"> => ({
  name,
  ctrl: false,
  meta: false,
  option: false,
  shift: false,
  ...overrides,
})

describe("model menu root navigation", () => {
  test("resolves lateral navigation without depending on the nested view", () => {
    expect(resolveRootNavigationDirection(key("left"))).toBe(-1)
    expect(resolveRootNavigationDirection(key("right"))).toBe(1)
  })

  test("resolves forward and reverse tab navigation", () => {
    expect(resolveRootNavigationDirection(key("tab"))).toBe(1)
    expect(resolveRootNavigationDirection(key("tab", { shift: true }))).toBe(-1)
  })

  test("leaves modified navigation keys unhandled", () => {
    expect(resolveRootNavigationDirection(key("left", { ctrl: true }))).toBeNull()
    expect(resolveRootNavigationDirection(key("right", { meta: true }))).toBeNull()
    expect(resolveRootNavigationDirection(key("tab", { option: true }))).toBeNull()
  })

  test("leaves unrelated keys to the active view", () => {
    expect(resolveRootNavigationDirection(key("escape"))).toBeNull()
    expect(resolveRootNavigationDirection(key("up"))).toBeNull()
  })
})

describe("installed model status", () => {
  test("identifies models installed from the Hugging Face cache", () => {
    expect(localModelInstalledStatus({
      _tag: "Downloaded",
      installedBytes: 1,
      origins: ["HuggingFaceCache"],
    })).toBe("Installed (HF)")
  })

  test("uses the standard installed label for Magnitude-managed models", () => {
    expect(localModelInstalledStatus({
      _tag: "Downloaded",
      installedBytes: 1,
      origins: ["Magnitude"],
    })).toBe("Installed")
  })

  test("does not render native inspection diagnostics as model-list status", () => {
    const model = {
      ...makeModel(),
      readiness: {
        _tag: "Failed" as const,
        failure: {
          code: "template_inspection_failed",
          message: "template inspection failed for /Users/example/.cache/model.gguf: native stderr",
          retryable: false,
        },
      },
    }

    expect(localModelReadinessStatus(model)).toBe("Error")
  })
})

describe("catalog keyboard navigation", () => {
  test("reveals the candidate selected by the keyboard cursor", () => {
    const revealed: string[] = []

    scrollCatalogCandidateIntoView({
      scrollChildIntoView: (id) => { revealed.push(id) },
    }, "qwen-config")

    expect(revealed).toEqual(["catalog-candidate:qwen-config"])
  })

  test("does nothing before the catalog scrollbox is mounted", () => {
    expect(() => scrollCatalogCandidateIntoView(null, "qwen-config")).not.toThrow()
  })
})

describe("catalog repository links", () => {
  test("derives unique Hugging Face repository URLs from package sources", () => {
    const candidate = makeCatalogCandidate({
      sources: [
        {
          source: {
            _tag: "HuggingFace",
            repository: "LiquidAI/LFM2.5-2.6B-GGUF",
            revision: "revision",
          },
          files: [],
        },
        {
          source: {
            _tag: "HuggingFace",
            repository: "LiquidAI/LFM2.5-2.6B-GGUF",
            revision: "revision",
          },
          files: [],
        },
        {
          source: { _tag: "Local", path: "/models/liquid" },
          files: [],
        },
      ],
    })

    expect(huggingFaceRepositoryUrls(candidate)).toEqual([
      "https://huggingface.co/LiquidAI/LFM2.5-2.6B-GGUF",
    ])
  })

  test("returns no repository URL for local-only packages", () => {
    const candidate = makeCatalogCandidate({
      sources: [{
        source: { _tag: "Local", path: "/models/local-only" },
        files: [],
      }],
    })

    expect(huggingFaceRepositoryUrls(candidate)).toEqual([])
  })
})

describe("models menu entries", () => {
  test("renders every disabled reason without a generic fallback", () => {
    expect([
      providerDisabledStatus("insufficient_resources"),
      providerDisabledStatus("provider_unavailable"),
      providerDisabledStatus("model_unavailable"),
      providerDisabledStatus("installation_unavailable"),
      providerDisabledStatus("incompatible_runtime"),
      providerDisabledStatus("invalid_configuration"),
    ]).toEqual([
      "Insufficient resources",
      "Provider unavailable",
      "Model unavailable",
      "Installation missing",
      "Incompatible runtime",
      "Invalid configuration",
    ])
  })

  test("does not put an uninstalled catalog model in Models", () => {
    expect(buildModelsMenuEntries([], [], [], [])).toEqual([])
  })

  test("keeps an installed model visible when its assessment does not fit", () => {
    const fitting = makeModel()
    if (fitting.readiness._tag !== "Assessed") throw new Error("fixture is not assessed")
    const model = {
      ...fitting,
      readiness: {
        ...fitting.readiness,
        assessment: {
          _tag: "DoesNotFit",
          assessmentId: "assessment-test" as never,
          environmentId: "environment-test" as never,
          memory: [],
          deficitBytes: 1,
          limitingResource: "system_memory",
        } as const,
      },
    }

    expect(buildModelsMenuEntries([], [model], [], [])).toMatchObject([
      { _tag: "LocalStatus", model },
    ])
  })

  test("keeps an unavailable installed offering visible but not selectable", () => {
    const available = makeModel()
    if (available.readiness._tag !== "Assessed") throw new Error("fixture is not assessed")
    const model = {
      ...available,
      readiness: {
        ...available.readiness,
        offering: Option.map(available.readiness.offering, (offering) => ({
          ...offering,
          availability: { _tag: "Disabled" as const, reason: "incompatible_runtime" as const },
        })),
      },
    }
    const choices = buildInstalledLocalModelChoices({
      inventory: { _tag: "Ready" },
      models: [model],
      downloads: [],
      recommendations: { _tag: "Loading", progress: [] },
    })
    const entries = buildModelsMenuEntries(choices, [model], [], [])

    expect(entries).toMatchObject([{ _tag: "Local" }])
    expect(modelsMenuSelectionAction(entries[0]!)).toEqual(Option.none())
  })

  test("keeps one stable local row while assessment completes", () => {
    const fitting = makeModel()
    const assessing = {
      ...fitting,
      readiness: { _tag: "Assessing" as const },
    }
    const fittingEntries = buildModelsMenuEntries(
      buildInstalledLocalModelChoices({
        inventory: { _tag: "Ready" },
        models: [fitting],
        downloads: [],
        recommendations: { _tag: "Loading", progress: [] },
      }),
      [fitting],
      [],
      [],
    )
    const assessingEntries = buildModelsMenuEntries([], [assessing], [], [])

    expect(assessingEntries).toHaveLength(1)
    expect(fittingEntries).toHaveLength(1)
    expect(assessingEntries[0]).toMatchObject({ _tag: "LocalStatus" })
    expect(fittingEntries[0]).toMatchObject({ _tag: "Local" })
    expect(assessingEntries[0]?.id).toBe(fittingEntries[0]?.id)
  })

  test("joins custom provider metadata into its model entry", () => {
    const providerId = ProviderIdSchema.make("custom:openrouter")
    const entries = buildModelsMenuEntries([], [], [{
      providerId,
      providerModelId: ProviderModelIdSchema.make("z-ai/glm-5.2"),
      modelFamilyId: Option.none(),
      displayName: "GLM 5.2",
      supportedSlots: [PRIMARY_SLOT_ID, SECONDARY_SLOT_ID],
      contextWindow: 1048576,
      maxOutputTokens: 128000,
      memory: Option.none(),
      capabilities: {
        vision: false,
        tools: true,
        structuredOutput: false,
        reasoning: {
          supported: true,
          efforts: [ReasoningEffortSchema.make("high")],
          defaultEffort: Option.some(ReasoningEffortSchema.make("high")),
        },
      },
      availability: { _tag: "Available" },
      pricing: Option.none(),
    }], [{
      providerId,
      displayName: "OpenRouter",
      kind: "Custom",
      authentication: "Authenticated",
      availability: { _tag: "Available" },
    }])

    expect(entries).toHaveLength(1)
    expect(entries[0]).toMatchObject({
      _tag: "Provider",
      provider: { displayName: "OpenRouter", kind: "Custom" },
      model: { displayName: "GLM 5.2" },
    })
  })

  test("keeps an installed catalog entry visible while its provider offering appears", () => {
    const withoutOffering = makeView({
      ready: false,
      models: [makeCatalogOnlyModel()],
      catalogCandidates: [makeCatalogCandidate({
        download: { _tag: "Downloaded", installedBytes: 16, origins: ["Magnitude"] },
        availability: { _tag: "Available" },
      })],
    })
    const withOffering = makeView({
      ready: false,
      catalogCandidates: [makeCatalogCandidate({
        download: { _tag: "Downloaded", installedBytes: 16, origins: ["Magnitude"] },
        availability: { _tag: "Available" },
      })],
    })
    const entriesFor = (view: ReturnType<typeof makeView>) => buildModelsMenuEntries(
      buildInstalledLocalModelChoices(view.models, view.slots),
      [],
      ProviderModelCatalogLifecycle.match(view.catalog, {
        Loading: () => [],
        Ready: ({ models }) => models,
        Refreshing: ({ models }) => models,
        Degraded: ({ models }) => models,
        Unavailable: () => [],
      }),
      ProviderModelCatalogLifecycle.match(view.catalog, {
        Loading: () => [],
        Ready: ({ providers }) => providers,
        Refreshing: ({ providers }) => providers,
        Degraded: ({ providers }) => providers,
        Unavailable: ({ providers }) => providers,
      }),
    )

    const before = entriesFor(withoutOffering)
    const after = entriesFor(withOffering)
    expect(before).toHaveLength(1)
    expect(after).toHaveLength(1)
    expect(Option.getOrThrow(modelsMenuSelectionAction(before[0]!))).toMatchObject({
      _tag: "InstallConfiguration",
    })
    expect(after[0]?._tag).toBe("Local")
    expect(modelsMenuEntryIsSelected(after[0]!, Option.none())).toBe(false)
    expect(modelsMenuEntryIsSelected(
      after[0]!,
      Option.some({ providerId: ProviderIdSchema.make("local"), providerModelId: TEST_MODEL_ID }),
    )).toBe(true)
    expect(Option.getOrThrow(modelsMenuSelectionAction(after[0]!))).toMatchObject({
      _tag: "AssignOffering",
      providerModel: { providerModelId: TEST_MODEL_ID },
    })
  })
})
