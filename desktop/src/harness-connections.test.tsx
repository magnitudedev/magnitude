import { renderToStaticMarkup } from "react-dom/server"
import { expect, it } from "vitest"
import { Option, Schema } from "effect"
import { DesktopHarnessConnection } from "@magnitudedev/client-common"
import { HarnessConnections } from "./harness-connections"

const missing = Schema.decodeUnknownSync(DesktopHarnessConnection)({
  id: "openclaw", name: "OpenClaw", installed: false, managed: true,
  inspection: { _tag: "Disconnected", reason: "Old configuration is missing" },
  configurationFiles: ["/private/old-config.json"], plugin: { name: "old plugin", source: "old-source" },
})
const installed: DesktopHarnessConnection = {
  ...missing, id: Schema.decodeUnknownSync(DesktopHarnessConnection.fields.id)("pi"), name: "Pi", installed: true, managed: false, plugin: Option.none(),
}
const render = (connections: readonly DesktopHarnessConnection[], busy = false) => renderToStaticMarkup(
  <HarnessConnections connections={connections} busy={busy} canConnect={true} onConnect={() => {}} onDisconnect={() => {}} />,
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
  for (const text of ["Could not verify connection", "Permission denied", "Connect", 'disabled=""']) expect(html).toContain(text)
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
