import { describe, expect, it, vi } from "vitest"
import {
  allowProviderMessageSend,
  NO_PROVIDERS_CONFIGURED_MESSAGE,
} from "./provider-send-guard"

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
