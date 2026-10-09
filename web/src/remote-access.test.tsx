import { renderToStaticMarkup } from "react-dom/server"
import { RegistryProvider } from "@effect-atom/atom-react"
import { expect, it } from "vitest"
import { SignIn } from "./remote-access"

const render = (keyConfigured: boolean) => renderToStaticMarkup(<RegistryProvider><SignIn origin="http://192.168.1.20:10100" keyConfigured={keyConfigured} onSignedIn={() => {}} /></RegistryProvider>)

it("asks for the Network access key in a password field", () => {
  const html = render(true)
  expect(html).toContain("Network access key")
  expect(html).toContain('type="password"')
  expect(html).toContain('autoComplete="current-password"')
})

it("explains how to create a key when the server has none", () => {
  const html = render(false)
  expect(html).toContain("No key is set")
  expect(html).not.toContain('type="password"')
})
