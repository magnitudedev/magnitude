import { describe, expect, it } from "vitest"
import { Registry } from "@effect-atom/atom-react"
import {
  composerAttachmentsAtom,
  composerTextAtom,
  messageHistoryAtom,
  selectedCwdAtom,
  sessionCreateOptionsAtom,
} from "./session-atoms"

describe("registry-lifetime client state", () => {
  it("declares state that must survive consumer gaps as keep-alive", () => {
    expect([
      selectedCwdAtom,
      messageHistoryAtom,
      composerTextAtom,
      composerAttachmentsAtom,
      sessionCreateOptionsAtom,
    ].every((atom) => atom.keepAlive)).toBe(true)
  })

  it("retains writes without consumers", async () => {
    const registry = Registry.make({
      defaultIdleTTL: 1,
      timeoutResolution: 1,
    })

    registry.set(selectedCwdAtom, "/workspace")
    await new Promise((resolve) => setTimeout(resolve, 10))

    expect(registry.get(selectedCwdAtom)).toBe("/workspace")
    registry.dispose()
  })
})
