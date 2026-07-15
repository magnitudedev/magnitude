import { describe, expect, it, vi } from "vitest"
import {
  allowProviderMessageSend,
  hasExplicitModelSlots,
  NO_PROVIDERS_CONFIGURED_MESSAGE,
} from "./provider-send-guard"

describe("hasExplicitModelSlots", () => {
  it("recognizes both persisted local model slots", () => {
    expect(hasExplicitModelSlots({
      primary: { providerId: "llamacpp", providerModelId: "/models/qwen.gguf" },
      secondary: { providerId: "llamacpp", providerModelId: "/models/qwen.gguf" },
    })).toBe(true)
  })

  it("rejects an incomplete slot configuration", () => {
    expect(hasExplicitModelSlots({
      primary: { providerId: "llamacpp", providerModelId: "/models/qwen.gguf" },
      secondary: {},
    })).toBe(false)
  })
})

describe("allowProviderMessageSend", () => {
  it("blocks the send and reports the missing provider", () => {
    const showToast = vi.fn()

    expect(allowProviderMessageSend(false, showToast)).toBe(false)
    expect(showToast).toHaveBeenCalledWith(NO_PROVIDERS_CONFIGURED_MESSAGE)
  })

  it("allows the send without showing an error", () => {
    const showToast = vi.fn()

    expect(allowProviderMessageSend(true, showToast)).toBe(true)
    expect(showToast).not.toHaveBeenCalled()
  })
})
