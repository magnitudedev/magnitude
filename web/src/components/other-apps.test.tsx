import { renderToStaticMarkup } from "react-dom/server"
import { expect, it } from "vitest"
import { OtherApps } from "./other-apps"
import { exampleRequest } from "./example-request"

const baseUrl = "http://127.0.0.1:10100/inference/v1"
const render = (model: string | undefined) => renderToStaticMarkup(
  <OtherApps origin="http://127.0.0.1:10100" model={model} platform="darwin" onOpenSettings={() => {}} />,
)

it("shows the local base URL and a Settings link with the example request collapsed", () => {
  const html = render("qwen3-32b")
  expect(html).toContain(baseUrl)
  expect(html).toContain("https://docs.magnitude.dev/integrations/other-agents")
  expect(html).toContain("network access in")
  expect(html).toContain("/inference/anthropic")
  expect(html).toContain('aria-expanded="false"')
  expect(html).not.toContain("chat/completions")
})

it("quotes the example for the user's shell", () => {
  expect(exampleRequest(baseUrl, "it's", "darwin")).toContain(`'{"model":"it'\\''s"`)
  expect(exampleRequest(baseUrl, "it's", "win32")).toMatch(/^Invoke-RestMethod .* -Body '\{"model":"it''s"/)
})
