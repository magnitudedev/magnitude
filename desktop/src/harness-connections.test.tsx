import { renderToStaticMarkup } from "react-dom/server"
import { expect, it } from "vitest"
import { Option, Schema } from "effect"
import { DesktopHarnessConnection } from "@magnitudedev/client-common"
import { ProviderModelIdSchema } from "@magnitudedev/sdk"
import { TooltipProvider } from "../../web/src/components/ui/tooltip"
import { HarnessConnections, HarnessCommand } from "./harness-connections"

const missing = Schema.decodeUnknownSync(DesktopHarnessConnection)({
  id: "openclaw", name: "OpenClaw", installed: false, managed: true,
  inspection: { _tag: "Disconnected", reason: "Old configuration is missing" },
  configurationFiles: ["/private/old-config.json"], plugin: { name: "old plugin", source: "old-source" },
})
const installed: DesktopHarnessConnection = {
  ...missing, id: Schema.decodeUnknownSync(DesktopHarnessConnection.fields.id)("pi"), name: "Pi", installed: true, managed: false, plugin: Option.none(),
}
const render = (connections: readonly DesktopHarnessConnection[], busy = false) => renderToStaticMarkup(
  <HarnessConnections connections={connections} busy={busy} canConnect={true} onConnect={() => {}} onDisconnect={() => {}} models={[]} defaultModel={undefined} platform="darwin" />,
)
it("puts installed harnesses first without changing the observed array", () => {
  const rows = [missing, installed]
  const html = render(rows)
  expect(html.indexOf('aria-label="Pi"')).toBeLessThan(html.indexOf('aria-label="OpenClaw"'))
  expect(rows[0]).toBe(missing)
  expect(html).not.toContain("Detected on your machine")
  expect(html).toContain("Not connected")
})
it("hides stale configuration and all controls for uninstalled harnesses", () => {
  const html = render([missing])
  expect(html).toContain('href="https://docs.openclaw.ai/install" target="_blank"')
  expect(html).toContain("Not installed")
  for (const text of ["<button", "<details", "old-config", "old plugin", "Old configuration", "Detect", "Disconnect"]) expect(html).not.toContain(text)
})
it("keeps connected and unverifiable states distinct from installation", () => {
  expect(render([{ ...installed, inspection: { _tag: "Connected" } }])).toContain("Connected")
  const html = render([{ ...installed, managed: true, inspection: { _tag: "Unavailable", reason: "Permission denied" } }], true)
  for (const text of ["Could not verify connection", "Permission denied", "Configuration files to check", "/private/old-config.json", "Connect", 'disabled=""']) expect(html).toContain(text)
})
it("sorts connected installations first and only shows verified configuration paths", () => {
  const connected: DesktopHarnessConnection = { ...installed, id: Schema.decodeUnknownSync(DesktopHarnessConnection.fields.id)("codex"), name: "Codex", inspection: { _tag: "Connected" } }
  const rows = [installed, missing, connected]
  const html = render(rows)
  expect(html.indexOf('aria-label="Codex"')).toBeLessThan(html.indexOf('aria-label="Pi"'))
  expect(html.indexOf('aria-label="Pi"')).toBeLessThan(html.indexOf('aria-label="OpenClaw"'))
  expect(rows).toEqual([installed, missing, connected])
  expect(render([connected])).toContain("bg-green-600")
  expect(render([connected])).toContain("/private/old-config.json")
  expect(render([installed])).toContain("bg-slate-400")
  expect(render([installed])).not.toContain("/private/old-config.json")
})

it("offers disconnect for detected external configuration", () => {
  expect(render([{ ...installed, managed: false, inspection: { _tag: "Connected" } }])).toContain("Disconnect")
  expect(render([installed])).not.toContain("Disconnect")
})

it("shows a themed warning and one repair action for damaged connections", () => {
  const html = render([{ ...installed, managed: true, inspection: { _tag: "Disconnected", reason: "Magnitude skill is missing or has changed" } }])
  for (const text of ["Connection needs repair", "bg-orange-500", "text-orange-600", "Repair connection"]) expect(html).toContain(text)
  expect(html.match(/<button/g)).toHaveLength(1)
  for (const text of ["Reconnect", "Disconnect", "Magnitude skill", "Remove configuration"]) expect(html).not.toContain(text)
  expect(html).not.toContain("Not connected")
})

it("shows model commands only for connected installations", () => {
  const model = Schema.decodeUnknownSync(ProviderModelIdSchema)("test-model-Q4")
  const html = renderToStaticMarkup(<HarnessConnections connections={[{ ...installed, inspection: { _tag: "Connected" } }]} busy={false} canConnect={true} onConnect={() => {}} onDisconnect={() => {}} models={[{ id: model, label: "Test model (Q4)" }]} defaultModel={model} platform="darwin" />)
  expect(html).toContain('aria-label="Pi model"')
  expect(html).toContain("pi --provider magnitude --model test-model-Q4")
  expect(html).toContain('aria-label="Copy Pi command"')
  expect(html).not.toContain(">Launch")
  expect(render([installed])).not.toContain("Copy Pi command")
})

it("uses an available fallback instead of a removed default", () => {
  const model = Schema.decodeUnknownSync(ProviderModelIdSchema)("available-Q8")
  const html = renderToStaticMarkup(<TooltipProvider><HarnessCommand harness={installed.id} name="Pi" models={[{ id: model, label: "Available (Q8)" }]} defaultModel={Schema.decodeUnknownSync(ProviderModelIdSchema)("removed-Q4")} platform="win32" /></TooltipProvider>)
  expect(html).toContain("--model available-Q8")
  expect(html).not.toContain("removed-Q4")
})

it("shows download guidance without a copy action when no model is available", () => {
  const html = render([{ ...installed, inspection: { _tag: "Connected" } }])
  expect(html).toContain("Download a compatible model")
  expect(html).not.toContain("Copy Pi command")
  expect(html).not.toContain("--model")
})

it("keeps OpenClaw model selection separate from its terminal command", () => {
  const model = Schema.decodeUnknownSync(ProviderModelIdSchema)("gemma-4-e2b-it-qat:gguf:q4")
  const html = renderToStaticMarkup(<TooltipProvider><HarnessCommand harness={missing.id} name="OpenClaw" models={[{ id: model, label: "Gemma (Q4)" }]} defaultModel={model} platform="darwin" /></TooltipProvider>)
  expect(html).toContain("openclaw tui")
  expect(html).toContain("/model magnitude/gemma-4-e2b-it-qat:gguf:q4")
  expect(html).not.toContain("--model")
  expect(html).toContain("inside it")
})
