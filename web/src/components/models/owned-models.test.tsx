import { renderToStaticMarkup } from "react-dom/server"
import { describe, expect, it } from "vitest"
import { makeSetupModel } from "@magnitudedev/client-common/fixtures/model"
import { modelLabs } from "../model-logo"
import { ownedModel } from "./data"
import { OwnedModels } from "./models-screen"
import { matchingModels, YourModels } from "./your-models"

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

  it("matches none of the visible models when the filters or search exclude them all, which is when 'No models match.' shows", () => {
    const otherLab = modelLabs.find(lab => lab.name !== model.lab?.name)!
    expect(matchingModels([model], { lab: otherLab, usOnly: false }, "")).toEqual([])
    expect(matchingModels([model], { lab: null, usOnly: false }, "no such model")).toEqual([])
    expect(matchingModels([model], { lab: null, usOnly: false }, "")).toEqual([model])
  })
})
