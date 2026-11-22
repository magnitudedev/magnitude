import { renderToStaticMarkup } from "react-dom/server"
import { Option } from "effect"
import { describe, expect, it } from "vitest"
import type { LocalModel } from "@magnitudedev/sdk"
import { makeSetupModel, streamedReadyResidency } from "@magnitudedev/client-common/fixtures/model"
import { modelLabs } from "../model-logo"
import { ownedModel } from "./data"
import { OwnedModelAction, OwnedModels } from "./models-screen"
import { matchingModels, YourModels } from "./your-models"

/** Whether any button or link sits inside a button or an element acting as one. */
const nestsControls = (html: string) => /role="button"/.test(html) || /<button\b(?:(?!<\/button>).)*<(?:button|a)\b/s.test(html)

const model = ownedModel(makeSetupModel(true), "Device")
const EMPTY = <p>No models are downloaded yet.</p>
const list = <YourModels models={[model]} action={() => null} onExploreQuants={() => {}} onRemove={() => {}} onDrafter={() => {}} />

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
      action={view => <OwnedModelAction model={pending} view={view} onLoad={() => {}} onStop={() => {}} onCancel={() => {}} onRetry={() => {}} />}
      onExploreQuants={() => {}} onRemove={() => {}} onDrafter={() => {}}
    />)
    expect(rowHtml).not.toContain(">Load<")
    expect(rowHtml.match(/Measuring…/g)).toHaveLength(1)
  })

  it("shows Load once standing measurements settle, and keeps Stop when already running", () => {
    const settled = makeSetupModel(true)
    const settledView = ownedModel(settled, "Device")
    const settledHtml = renderToStaticMarkup(<OwnedModelAction model={settled} view={settledView} onLoad={() => {}} onStop={() => {}} onCancel={() => {}} onRetry={() => {}} />)
    expect(settledHtml).toContain("Load")
    expect(settledHtml).not.toContain("Measuring…")

    const running: LocalModel = {
      ...settled,
      acquisitionState: settled.acquisitionState._tag === "Installed"
        ? { ...settled.acquisitionState, residencyState: streamedReadyResidency(["metal:0"]) }
        : settled.acquisitionState,
    }
    const runningHtml = renderToStaticMarkup(<OwnedModelAction model={running} view={ownedModel(running, "Device")} onLoad={() => {}} onStop={() => {}} onCancel={() => {}} onRetry={() => {}} />)
    expect(runningHtml).toContain("Running")
    expect(runningHtml).toContain("Stop")
  })
})
