import { createElement } from "react"
import { renderToStaticMarkup } from "react-dom/server"
import { Option } from "effect"
import { describe, expect, it } from "vitest"
import {
  ModelSlotConfiguredLocal,
  ModelInstanceIdSchema,
  ModelServingConfigurationIdSchema,
  PRIMARY_SLOT_ID,
  ProviderIdSchema,
  ProviderModelIdSchema,
  ReasoningEffortSchema,
  type DisplayRootStatus,
  type ModelResidency,
} from "@magnitudedev/sdk"
import { InlineWorkActivity } from "./inline-work-activity"

const working = (
  detail: Extract<DisplayRootStatus, { readonly _tag: "Working" }>["detail"],
): DisplayRootStatus => ({
  _tag: "Working",
  chainStartedAt: Date.now() - 5_000,
  detail,
  activeChildCount: 0,
})

const selection = {
  providerId: ProviderIdSchema.make("local"),
  providerModelId: ProviderModelIdSchema.make("test-configuration"),
  reasoningEffort: ReasoningEffortSchema.make("none"),
}

const requestedModel = new ModelSlotConfiguredLocal({
  slotId: PRIMARY_SLOT_ID,
  selection,
  descriptor: {
    providerId: selection.providerId,
    providerModelId: selection.providerModelId,
    displayName: "Qwen Test",
    variantLabel: Option.none(),
  },
  availability: { _tag: "Available" },
  residency: { _tag: "Requested" },
  actions: ["Stop"],
})

const loadingModel = (progress: Option.Option<number>) => new ModelSlotConfiguredLocal({
  ...requestedModel,
  residency: {
    _tag: "Loading",
    instanceId: ModelInstanceIdSchema.make("instance"),
    configurationId: ModelServingConfigurationIdSchema.make("configuration"),
    stage: "loading",
    progress,
    plannedAllocation: Option.none(),
  } satisfies ModelResidency,
})

const render = (props: React.ComponentProps<typeof InlineWorkActivity>) =>
  renderToStaticMarkup(createElement(InlineWorkActivity, props))

describe("InlineWorkActivity", () => {
  it("gives authoritative model loading precedence over waiting", () => {
    const html = render({
      rootStatus: working({ _tag: "WaitingForModel", turnStartedAt: Date.now() }),
      modelLoadActivity: requestedModel,
      modelName: "Qwen Test",
    })
    expect(html).toContain("Loading Qwen Test")
    expect(html).not.toContain("Waiting for model")
    expect(html).toContain('role="status"')
  })

  it("renders indeterminate loading without fabricating zero percent", () => {
    const html = render({
      rootStatus: working({ _tag: "WaitingForModel", turnStartedAt: Date.now() }),
      modelLoadActivity: loadingModel(Option.none()),
      modelName: "Qwen Test",
    })
    expect(html).toContain("Loading Qwen Test")
    expect(html).not.toContain(">0%</span>")
    expect(html).not.toContain('role="progressbar"')
    expect(html).not.toContain("animate-shimmer rounded-full")
  })

  it("renders each authoritative progress sample", () => {
    const zero = render({
      rootStatus: working({ _tag: "WaitingForModel", turnStartedAt: Date.now() }),
      modelLoadActivity: loadingModel(Option.some(0)),
      modelName: "Qwen Test",
    })
    const advanced = render({
      rootStatus: working({ _tag: "WaitingForModel", turnStartedAt: Date.now() }),
      modelLoadActivity: loadingModel(Option.some(0.47)),
      modelName: "Qwen Test",
    })
    expect(zero).toContain("0%")
    expect(zero).toContain('aria-valuenow="0"')
    expect(advanced).toContain("47%")
    expect(advanced).toContain('aria-valuenow="47"')
  })

  it("retires slot loading activity when the conversation is no longer working", () => {
    const html = render({
      rootStatus: {
        _tag: "Worked",
        lastProductiveMs: 5_000,
      },
      modelLoadActivity: loadingModel(Option.some(0)),
      modelName: "Qwen Test",
    })
    expect(html).toBe("")
  })

  it("renders effective uncached prefill detail without a progress bar", () => {
    const html = render({
      rootStatus: working({
        _tag: "Prefill",
        totalTokens: 1_200,
        completedTokens: 700,
        cachedTokens: 200,
      }),
      modelLoadActivity: null,
      modelName: null,
    })
    expect(html).toContain("Preparing context")
    expect(html).toContain("500 / 1.0k tokens")
    expect(html).toContain("200 cached")
    expect(html).not.toContain('role="progressbar"')
  })

  it("suppresses generic work when the chronological entry owns activity", () => {
    const html = render({
      rootStatus: working({ _tag: "NoDetail" }),
      modelLoadActivity: null,
      modelName: null,
      suppressGeneric: true,
    })
    expect(html).toBe("")
  })
})
