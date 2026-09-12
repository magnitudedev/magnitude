import { describe, expect, it, vi } from "vitest"
import { buildTrayMenu } from "./tray-menu"

const actions = () => ({ open: vi.fn(), stopModel: vi.fn(), quit: vi.fn() })
const model = { label: "Bonsai · Loaded", canStop: true }
describe("native tray menu", () => {
  it.each(["Starting", "Failed", "CleanupFailed", "Stopping", "Stopped"] as const)("keeps Open, Status, and Quit available during %s without stale model actions", service => {
    const menu = buildTrayMenu({ service, model, setup: "Required" }, actions())
    const labels = menu.flatMap(item => "label" in item && typeof item.label === "string" ? [item.label] : [])
    expect(labels).toContain("Model status unavailable")
    expect(labels).toContain("Open Magnitude")
    expect(labels).toContain("Status")
    expect(labels).toContain("Quit Magnitude")
    expect(labels).not.toContain("Stop Model")
    expect(labels).not.toContain(model.label)
  })
  it("routes every actionable item to the common application actions", () => {
    const callbacks = actions()
    const menu = buildTrayMenu({ service: "Ready", model, setup: "Complete" }, callbacks)
    for (const item of menu) if ("click" in item) item.click?.()
    expect(callbacks.open.mock.calls).toEqual([[], ["discover"], ["status"]])
    expect(callbacks.stopModel).toHaveBeenCalledOnce()
    expect(callbacks.quit).toHaveBeenCalledOnce()
  })
  it("shows first-use guidance only from an observed incomplete flag", () => {
    for (const setup of ["Required", "Complete", "Unavailable"] as const) {
      const menu = buildTrayMenu({ service: "Ready", model: { label: "No model loaded", canStop: false }, setup }, actions())
      const labels = menu.flatMap(item => "label" in item && typeof item.label === "string" ? [item.label] : [])
      expect(labels.some(label => label.startsWith("Setup needed"))).toBe(setup === "Required")
      expect(labels).not.toContain("Stop Model")
    }
  })
})
