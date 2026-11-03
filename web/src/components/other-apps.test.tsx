import { renderToStaticMarkup } from "react-dom/server"
import { expect, it } from "vitest"
import { OtherApps } from "./other-apps"
import { chatCompletionExample, decisionsExample, decisionsPython, exampleRequest, systemOneExample, systemOnePython } from "./example-request"

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

it("points to the address and key in Settings once network access is on", () => {
  const html = renderToStaticMarkup(<OtherApps origin="http://127.0.0.1:10100" model="m" platform="linux" networkAccess onOpenSettings={() => {}} />)
  expect(html).toContain("Other devices use the address and API key under Network access in")
  expect(html).not.toContain("turn on network access")
})

it("quotes the example for the user's shell", () => {
  expect(exampleRequest(baseUrl, chatCompletionExample("it's"), "darwin")).toContain(`'{"model":"it'\\''s"`)
  expect(exampleRequest(baseUrl, chatCompletionExample("it's"), "win32")).toMatch(/^Invoke-RestMethod .* -Body '\{"model":"it''s"/)
})

it("tells another device to send the key and points it to Settings for it", () => {
  const html = renderToStaticMarkup(<OtherApps origin="http://192.168.1.20:10100" model="m" platform="win32" remote onOpenSettings={() => {}} />)
  expect(html).toContain("http://192.168.1.20:10100/inference/v1")
  expect(html).toContain("Send the Network access key")
  expect(html).not.toContain("Any API key works")
  expect(html).toContain("Copy the key from Network access in")
})

it("adds the key header to the example only from another device", () => {
  expect(exampleRequest(baseUrl, chatCompletionExample("m"), "darwin", true)).toContain(`-H "Authorization: Bearer YOUR_KEY"`)
  expect(exampleRequest(baseUrl, chatCompletionExample("m"), "win32", true)).toContain(`-Headers @{ Authorization = "Bearer YOUR_KEY" }`)
  expect(exampleRequest(baseUrl, chatCompletionExample("m"), "darwin")).not.toContain("Authorization")
})

it("asks the same decision in both decision formats", () => {
  const systemOne = systemOneExample("elephant-1")
  const decisions = decisionsExample("elephant-1")
  expect(Object.keys(systemOne.body.questions)).toEqual(decisions.body.questions.map(question => question.name))
  expect(exampleRequest(baseUrl, systemOne, "darwin")).toContain(`curl ${baseUrl}/systemone`)
  expect(exampleRequest(baseUrl, decisions, "win32")).toMatch(/^Invoke-RestMethod .*\/decisions -Method Post/)
})

it("points the TypeSafe SDK at the API root and the OpenAI SDK at the base URL", () => {
  expect(systemOnePython("http://127.0.0.1:10100/inference", "m")).toContain(`base_url="http://127.0.0.1:10100/inference"`)
  expect(decisionsPython(baseUrl, "m", true)).toContain(`OpenAI(api_key="YOUR_KEY", base_url="${baseUrl}")`)
})
