import { describe, expect, it, vi } from "vitest"
import {
  allowModelMessageSend,
  NO_MODEL_SELECTED_MESSAGE,
} from "./provider-send-guard"

describe("allowModelMessageSend", () => {
  it("blocks the send and reports the unassigned model slot", () => {
    const showToast = vi.fn()

    expect(allowModelMessageSend(false, showToast)).toBe(false)
    expect(showToast).toHaveBeenCalledWith(NO_MODEL_SELECTED_MESSAGE)
  })

  it("allows the send without showing an error", () => {
    const showToast = vi.fn()

    expect(allowModelMessageSend(true, showToast)).toBe(true)
    expect(showToast).not.toHaveBeenCalled()
  })
})
