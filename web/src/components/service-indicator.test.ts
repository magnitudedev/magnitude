import { describe, expect, it } from "vitest"
import { serviceIndicator } from "./service-indicator"

describe("the status card's service mark", () => {
  it("shows no running dot beside No model loaded, only beside an active model", () => {
    expect(serviceIndicator({ ready: true, unavailable: false, modelActive: false })).toBeNull()
    expect(serviceIndicator({ ready: true, unavailable: false, modelActive: true })).toBe("ready")
  })

  it("still shows the service starting or failing whatever the model", () => {
    expect(serviceIndicator({ ready: false, unavailable: false, modelActive: false })).toBe("starting")
    expect(serviceIndicator({ ready: false, unavailable: true, modelActive: false })).toBe("failed")
  })
})
