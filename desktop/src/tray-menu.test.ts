import { Option } from "effect"
import { describe, expect, it, vi } from "vitest"
import { buildTrayMenu, MODEL_STATUS_ITEM } from "./tray-menu"

const actions = () => ({ open: vi.fn(), stopModel: vi.fn(), quit: vi.fn(), restartUpdate: vi.fn() })
const model = {
  label: "Bonsai · Loaded · 18.4 GB",
  status: Option.some({ model: "Bonsai", phase: "Loaded", detail: { _tag: "Memory" as const, text: "18.4 GB" } }),
  canStop: true,
}
describe("native tray menu", () => {
  it.each(["Starting", "Failed", "CleanupFailed", "Stopping", "Stopped"] as const)("keeps Open, Status, and Quit available during %s without stale model actions", service => {
    const menu = buildTrayMenu({ service, model, updateReady: false }, actions())
    const labels = menu.flatMap(item => "label" in item && typeof item.label === "string" ? [item.label] : [])
    expect(labels).toContain("Model status unavailable")
    expect(labels).toContain("Open Magnitude")
    expect(labels).toContain("Status")
    expect(labels).toContain("Quit Magnitude")
    expect(labels).not.toContain("Stop Model")
    expect(labels).not.toContain(model.label)
    expect(menu.some(item => "id" in item && item.id === MODEL_STATUS_ITEM)).toBe(false)
  })
  it("marks the model line for the live row only while a model is active", () => {
    const active = buildTrayMenu({ service: "Ready", model, updateReady: false }, actions())
    expect(active.findIndex(item => "id" in item && item.id === MODEL_STATUS_ITEM)).toBe(1)
    const idle = buildTrayMenu({ service: "Ready", model: { label: "No model loaded", status: Option.none(), canStop: false }, updateReady: false }, actions())
    expect(idle.some(item => "id" in item && item.id === MODEL_STATUS_ITEM)).toBe(false)
  })
  it("routes every actionable item to the common application actions", () => {
    const callbacks = actions()
    const menu = buildTrayMenu({ service: "Ready", model, updateReady: true }, callbacks)
    for (const item of menu) if ("click" in item) item.click?.()
    expect(callbacks.open.mock.calls).toEqual([[], ["discover"], ["status"]])
    expect(callbacks.stopModel).toHaveBeenCalledOnce()
    expect(callbacks.quit).toHaveBeenCalledOnce()
    expect(callbacks.restartUpdate).toHaveBeenCalledOnce()
  })
})
