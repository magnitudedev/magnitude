import { renderToStaticMarkup } from "react-dom/server"
import { describe, expect, it } from "vitest"
import type { CatalogLocalModel } from "@magnitudedev/sdk"
import { LoadModelButton, loadRefusal } from "./model-fit"

const assessed = (assessment: object) => ({ servingState: { _tag: "Assessed", assessment } }) as unknown as Pick<CatalogLocalModel, "servingState">
const doesNotFit = assessed({ _tag: "DoesNotFit", deficitBytes: 1_825_361_000 })
const fits = assessed({ _tag: "Fits" })

describe("Load model action", () => {
  it("is disabled with the shortfall when the model does not fit", () => {
    const html = renderToStaticMarkup(<LoadModelButton model={doesNotFit} pending={false} onLoad={() => {}} />)
    expect(html).toContain("Load model")
    expect(html).toMatch(/<button[^>]*disabled=""/)
    expect(html).toContain("needs 1.7 GB more memory than this computer can provide")
    expect(loadRefusal(doesNotFit)).toContain("1.7 GB")
  })
  it("is enabled when the model fits or is still being assessed", () => {
    for (const model of [fits, { servingState: { _tag: "Assessing" } } as unknown as Pick<CatalogLocalModel, "servingState">]) {
      const html = renderToStaticMarkup(<LoadModelButton model={model} pending={false} onLoad={() => {}} />)
      expect(html).not.toMatch(/disabled=""/)
      expect(loadRefusal(model)).toBeNull()
    }
  })
  it("is disabled with the engine's reasons when Magnitude can't run the model", () => {
    const unsupported = assessed({ _tag: "Unsupported", reasons: [{ _tag: "Architecture", architecture: "deepseek4" }], detail: "PRIVATE" })
    expect(loadRefusal(unsupported)).toBe("Magnitude can’t run this model. No kernels for the deepseek4 architecture.")
    expect(renderToStaticMarkup(<LoadModelButton model={unsupported} pending={false} onLoad={() => {}} />)).toMatch(/<button[^>]*disabled=""/)
  })
})
