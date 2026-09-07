import { act } from "react"
import { testRender } from "@opentui/react/test-utils"
import { HarnessIdSchema } from "@magnitudedev/client-common"
import { describe, expect, it } from "vitest"
import { SetupFrame, SetupHostContext } from "./setup-frame"

describe("shared hosted setup frame", () => {
  it.each([120, 80, 50])("retains normal labels without a host at width %s", async width => {
    const view = await testRender(<SetupFrame width={width} stage="choose"><text>Shared content</text></SetupFrame>, { width, height: 60 })
    try {
      await act(view.renderOnce)
      const frame = view.captureCharFrame()
      expect(frame).toContain("Select harness")
      expect(frame).toContain("Shared content")
      expect(frame).not.toContain("Package:")
    } finally { await act(async () => view.renderer.destroy()) }
  })
  it.each([false, true])("omits the setup disclosure (development %s)", async developmentService => {
    const host = {
      id: HarnessIdSchema.make("pi"), name: "Pi", availability: "Installed" as const,
      selectable: true, connected: false, developmentService,
      companion: { name: "Magnitude", source: "npm:@magnitudedev/pi-extension@1.0.0", securityNotice: "Runs with user permissions." },
    }
    const view = await testRender(
      <SetupHostContext.Provider value={host}><SetupFrame width={120} stage="choose"><text>Shared content</text></SetupFrame></SetupHostContext.Provider>,
      { width: 120, height: 40 },
    )
    try {
      await act(view.renderOnce)
      const frame = view.captureCharFrame().replace(/\s+/g, " ")
      expect(frame).toContain("Connect Pi")
      expect(frame).not.toContain("Select harness")
      expect(frame).not.toContain("Setup connects")
      expect(frame).not.toContain("installs the Magnitude skill")
      expect(frame).not.toContain(host.companion.source)
      expect(frame).not.toContain("Runs with user permissions.")
      expect(frame).not.toContain("login startup")
      expect(frame).toContain("Shared content")
    } finally { await act(async () => view.renderer.destroy()) }
  })
})
