import { renderToStaticMarkup } from "react-dom/server"
import { describe, expect, it } from "vitest"
import { ModelRadar } from "./discovery-visuals"
import { catalogModel } from "../../test/models/fixtures"

describe("model radar", () => {
  it("puts the memory breakdown icon beside the fit label", () => {
    const html = renderToStaticMarkup(<ModelRadar model={catalogModel()} />)
    expect(html).toMatch(/>Fit<button[^>]*aria-label="Memory breakdown"/)
    expect(html).toContain("(24 GB)")
  })
})
