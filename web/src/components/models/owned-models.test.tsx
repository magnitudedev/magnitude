import { renderToStaticMarkup } from "react-dom/server"
import { Option, Schema } from "effect"
import { describe, expect, it } from "vitest"
import { CatalogFormModelIdSchema, CatalogSupportSchema, LocalModelMutationFailed, QuantNameSchema, type LocalModel } from "@magnitudedev/sdk"
import type { LocalModelCommandStatus } from "@magnitudedev/client-common"
import { makeSetupModel, streamedReadyResidency } from "@magnitudedev/client-common/fixtures/model"
import { modelLabs } from "../model-logo"
import { ownedModel } from "./data"
import { modelRowAction, OwnedModels } from "./models-screen"
import { matchingModels, YourModels } from "./your-models"

/** Whether any button or link sits inside a button or an element acting as one. */
const nestsControls = (html: string) => /role="button"/.test(html) || /<button\b(?:(?!<\/button>).)*<(?:button|a)\b/s.test(html)

const baseModel = makeSetupModel(true)
const model = ownedModel(baseModel, "Device")
const EMPTY = <p>No models are downloaded yet.</p>
const list = <YourModels models={[model]} action={() => null} onExploreQuants={() => {}} onRemove={() => {}} onDrafter={() => {}} />
const noopActions = { loadPending: false, onLoad: () => {}, onStop: () => {}, onCancel: () => {}, onRetryInstall: () => {} }

const replacementModel: LocalModel = {
  ...makeSetupModel(false),
  modelId: CatalogFormModelIdSchema.make("qwen3.8-27b:gguf:iq2"),
  presentation: { ...baseModel.presentation, displayName: "Qwen3.8 27B" },
  catalogData: {
    ...baseModel.catalogData,
    quant: { ...baseModel.catalogData.quant, name: QuantNameSchema.make("IQ2_M") },
  },
}

const deprecatedModel: LocalModel = {
  ...baseModel,
  modelId: CatalogFormModelIdSchema.make("qwen3.5-9b:gguf:q4"),
  presentation: { ...baseModel.presentation, displayName: "Qwen3.5 9B" },
  catalogData: {
    ...baseModel.catalogData,
    intelligence: Option.none(),
    support: Schema.decodeUnknownSync(CatalogSupportSchema)({
      _tag: "Deprecated",
      since: "2026-11-18",
      replacement: replacementModel.modelId,
      reason: "Superseded by Qwen3.8 27B at 2-bit, which is far smarter at a similar size.",
    }) as Extract<typeof CatalogSupportSchema.Type, { readonly _tag: "Deprecated" }>,
  },
}

const disabledModel: LocalModel = {
  ...baseModel,
  modelId: CatalogFormModelIdSchema.make("disabled-model:gguf:q4"),
  presentation: { ...baseModel.presentation, displayName: "Disabled Model" },
  catalogData: {
    ...baseModel.catalogData,
    intelligence: Option.none(),
    support: { _tag: "Disabled", reason: "Full serving path is unfinished." },
  },
}

describe("the user's models", () => {
  it("shows the empty state, then Hidden, while every model is hidden, with no search or filters and no 'No models match.'", () => {
    const html = renderToStaticMarkup(<OwnedModels list={null} empty={EMPTY} hidden={[model]} unrunnable={[]} onUnhide={() => {}} />)
    expect(html).toContain("No models are downloaded yet.")
    expect(html).toContain("Hidden (1)")
    expect(html).not.toContain("No models match.")
    expect(html).not.toContain("Find in your models")
  })

  it("shows the list in the empty state's place once a model is visible, with Hidden still below", () => {
    const html = renderToStaticMarkup(<OwnedModels list={list} empty={EMPTY} hidden={[model]} unrunnable={[]} onUnhide={() => {}} />)
    expect(html).not.toContain("No models are downloaded yet.")
    expect(html).toContain("1 downloaded model")
    expect(html).toContain("Find in your models")
    expect(html).toContain("Hidden (1)")
    expect(html).not.toContain("No models match.")
  })

  it("opens a model from its own button, with Load and More beside it rather than inside it", () => {
    const withActions = <YourModels models={[model]} action={() => <button type="button">Load</button>} onExploreQuants={() => {}} onRemove={() => {}} onDrafter={() => {}} />
    const html = renderToStaticMarkup(withActions)
    expect(html).toContain('aria-expanded="false"')
    expect(nestsControls(html)).toBe(false)
  })

  it("matches none of the visible models when the filters or search exclude them all, which is when 'No models match.' shows", () => {
    const otherLab = modelLabs.find(lab => lab.name !== model.lab?.name)!
    expect(matchingModels([model], { lab: otherLab, usOnly: false }, "")).toEqual([])
    expect(matchingModels([model], { lab: null, usOnly: false }, "no such model")).toEqual([])
    expect(matchingModels([model], { lab: null, usOnly: false }, "")).toEqual([model])
  })

  it("hides Load behind a single Measuring status while standing measurements are still pending", () => {
    const base = makeSetupModel(true)
    const pending: LocalModel = {
      ...base,
      servingState: base.servingState._tag === "Assessed" && base.servingState.assessment._tag === "Fits"
        ? {
            ...base.servingState,
            assessment: {
              ...base.servingState.assessment,
              residency: {
                _tag: "Resident" as const,
                performance: { _tag: "Pending" as const, measurement: { _tag: "Measuring" as const, progress: Option.none() } },
              },
            },
          }
        : base.servingState,
    }
    const pendingView = ownedModel(pending, "Device")
    expect(pendingView.quant.fits).toBe(true)
    expect(pendingView.quant.measuring).toBe(true)
    const rowHtml = renderToStaticMarkup(<YourModels
      models={[pendingView]}
      action={view => modelRowAction(view, pending, noopActions)}
      onExploreQuants={() => {}} onRemove={() => {}} onDrafter={() => {}}
    />)
    expect(rowHtml).not.toContain(">Load<")
    expect(rowHtml.match(/Measuring…/g)).toHaveLength(1)
  })

  it("shows Load once standing measurements settle, and keeps Stop when already running", () => {
    const settled = makeSetupModel(true)
    const settledView = ownedModel(settled, "Device")
    const settledHtml = renderToStaticMarkup(<>{modelRowAction(settledView, settled, noopActions)}</>)
    expect(settledHtml).toContain("Load")
    expect(settledHtml).not.toContain("Measuring…")

    const running: LocalModel = {
      ...settled,
      acquisitionState: settled.acquisitionState._tag === "Installed"
        ? { ...settled.acquisitionState, residencyState: streamedReadyResidency(["metal:0"]) }
        : settled.acquisitionState,
    }
    const runningHtml = renderToStaticMarkup(<>{modelRowAction(ownedModel(running, "Device"), running, noopActions)}</>)
    expect(runningHtml).toContain("Running")
    expect(runningHtml).toContain("Stop")
  })

  it("surfaces a deprecated owned catalog model with its badge and replacement, and omits the broken Load button", () => {
    const deprecated = ownedModel(deprecatedModel, "Device", [deprecatedModel, replacementModel])
    const html = renderToStaticMarkup(
      <YourModels
        models={[deprecated]}
        action={view => modelRowAction(view, deprecatedModel, noopActions)}
        onExploreQuants={() => {}}
        onRemove={() => {}}
        onDrafter={() => {}}
      />,
    )
    expect(html).toContain("Deprecated")
    expect(html).toContain("No longer supported")
    expect(html).toContain("Superseded by Qwen3.8 27B at 2-bit, which is far smarter at a similar size.")
    expect(html).toContain("Switch to Qwen3.8 27B (IQ2_M).")
    expect(html).not.toMatch(/>Load<\/button>/)
    expect(nestsControls(html)).toBe(false)
  })

  it("surfaces a disabled owned catalog model with its reason and omits Load", () => {
    const disabled = ownedModel(disabledModel, "Device", [disabledModel])
    const html = renderToStaticMarkup(
      <YourModels
        models={[disabled]}
        action={view => modelRowAction(view, disabledModel, noopActions)}
        onExploreQuants={() => {}}
        onRemove={() => {}}
        onDrafter={() => {}}
      />,
    )
    expect(html).toContain("Disabled")
    expect(html).toContain("Full serving path is unfinished.")
    expect(html).not.toMatch(/>Load<\/button>/)
  })

  it("surfaces a rejected Load click on the row when residency stays unloaded", () => {
    const failedLoad: LocalModelCommandStatus = {
      pending: false,
      pendingOperations: [],
      failures: [{
        operation: "load",
        rejection: Option.some(new LocalModelMutationFailed({
          code: "memory_shortage",
          message: "PRIVATE requires 18710917797 bytes and 17334222848 bytes are available",
          retryable: true,
        })),
      }],
    }
    const failed = ownedModel(baseModel, "Device", [baseModel], failedLoad)
    expect(failed.failure?.detail).toContain("PRIVATE")
    const html = renderToStaticMarkup(
      <YourModels
        models={[failed]}
        action={view => modelRowAction(view, baseModel, noopActions)}
        onExploreQuants={() => {}}
        onRemove={() => {}}
        onDrafter={() => {}}
      />,
    )
    expect(html).toContain('role="alert"')
    expect(html).toContain("Not enough memory right now.")
    expect(html).toContain("Quit apps you aren’t using, or choose a smaller model.")
    expect(html).toContain("Couldn’t load")
    expect(html).toContain(">Load</button>")
    expect(html).not.toContain("PRIVATE")
    expect(nestsControls(html)).toBe(false)
  })

  it("surfaces a non-retryable or transport Load rejection and shows Loading while a load command is pending", () => {
    const transportFailed: LocalModelCommandStatus = {
      pending: false,
      pendingOperations: [],
      failures: [{ operation: "load", rejection: Option.none() }],
    }
    const failed = ownedModel(baseModel, "Device", [baseModel], transportFailed)
    const failedHtml = renderToStaticMarkup(
      <YourModels
        models={[failed]}
        action={view => modelRowAction(view, baseModel, noopActions)}
        onExploreQuants={() => {}}
        onRemove={() => {}}
        onDrafter={() => {}}
      />,
    )
    expect(failedHtml).toContain('role="alert"')
    expect(failedHtml).toContain("This model couldn’t load.")
    expect(failedHtml).toContain(">Load</button>")

    const pendingHtml = renderToStaticMarkup(
      <YourModels
        models={[model]}
        action={view => modelRowAction(view, baseModel, { ...noopActions, loadPending: true })}
        onExploreQuants={() => {}}
        onRemove={() => {}}
        onDrafter={() => {}}
      />,
    )
    expect(pendingHtml).toContain("Loading…")
    expect(pendingHtml).not.toContain(">Load</button>")
  })

  it("shows Not supported with the engine reason instead of Doesn’t fit for unsupported downloaded models", () => {
    const assessed = baseModel.servingState._tag === "Assessed" ? baseModel.servingState : (() => { throw new Error("expected assessed") })()
    const unsupported: LocalModel = {
      _tag: "Custom",
      modelId: "hf:magnitudedev/d1-2B-GGUF/d1-2B-Q4_K_M.gguf" as never,
      origin: { _tag: "Added", repository: "magnitudedev/d1-2B-GGUF" as never, commit: "1a75d474b9b3b91f150e391a42ef96cf134401e9" },
      presentation: {
        displayName: "d1 2B",
        variantLabel: "Q4_K_M" as never,
        sourceLabel: "Added",
        variantOf: Option.none(),
        developer: Option.none(),
        license: Option.none(),
        sourceUrls: [],
      },
      drafter: Option.none(),
      drafterPending: false,
      drafterChoice: { _tag: "Auto" },
      recommendedDrafter: Option.none(),
      acquisitionState: baseModel.acquisitionState as never,
      servingState: {
        ...assessed,
        assessment: {
          _tag: "Unsupported",
          environmentId: assessed.assessment.environmentId,
          profile: Option.none(),
          reasons: [{ _tag: "UnrecognizedMetadata", key: "lfm3.attention.causal" }],
          detail: 'unknown metadata ["lfm3.attention.causal"]',
        },
      },
      storageBytes: 1_480_000_000,
    }
    const view = ownedModel(unsupported, "Unified")
    expect(view.unsupported).toBe("Has parts Magnitude doesn't recognize (lfm3.attention.causal)")
    const html = renderToStaticMarkup(
      <YourModels
        models={[view, model]}
        action={item => modelRowAction(item, item.modelId === view.modelId ? unsupported : baseModel, noopActions)}
        onExploreQuants={() => {}}
        onRemove={() => {}}
        onDrafter={() => {}}
      />,
    )
    expect(html).toContain("2 downloaded models")
    expect(html).toContain("Not supported")
    expect(html).toContain("Has parts Magnitude doesn&#x27;t recognize (lfm3.attention.causal)")
    expect(html).not.toContain("Doesn’t fit")
  })
})
