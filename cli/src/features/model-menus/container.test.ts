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
  modelsMenuEntryIsSelected,
  modelsMenuSelectionAction,
  resolveRootNavigationDirection,
  scrollCatalogCandidateIntoView,
} from "./container"
import {
  makeCatalogCandidate,
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
  test("joins custom provider metadata into its model entry", () => {
    const providerId = ProviderIdSchema.make("custom:openrouter")
    const entries = buildModelsMenuEntries([], [{
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

  test("represents an installed target once before and after offering creation", () => {
    const withoutOffering = makeView({
      ready: false,
      models: [makeModel({ offerings: [] })],
      catalogCandidates: [makeCatalogCandidate({
        download: { _tag: "Downloaded", installedBytes: 16 },
        availability: { _tag: "Available" },
      })],
    })
    const withOffering = makeView({
      ready: false,
      catalogCandidates: [makeCatalogCandidate({
        download: { _tag: "Downloaded", installedBytes: 16 },
        availability: { _tag: "Available" },
      })],
    })
    const entriesFor = (view: ReturnType<typeof makeView>) => buildModelsMenuEntries(
      buildInstalledLocalModelChoices(view.models, view.catalog, view.slots),
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
    expect(before[0]?.id).toBe(after[0]?.id)
    expect(before[0]?._tag).toBe("Local")
    expect(after[0]?._tag).toBe("Local")
    expect(modelsMenuEntryIsSelected(before[0]!, Option.none())).toBe(false)
    expect(modelsMenuEntryIsSelected(after[0]!, Option.none())).toBe(false)
    expect(modelsMenuEntryIsSelected(
      after[0]!,
      Option.some({ providerId: ProviderIdSchema.make("local"), providerModelId: TEST_MODEL_ID }),
    )).toBe(true)
    expect(Option.getOrThrow(modelsMenuSelectionAction(before[0]!))).toMatchObject({
      _tag: "CreateOffering",
      configurationId: makeCatalogCandidate().configurationId,
    })
    expect(Option.getOrThrow(modelsMenuSelectionAction(after[0]!))).toMatchObject({
      _tag: "AssignOffering",
      providerModel: { providerModelId: TEST_MODEL_ID },
    })
  })
})
