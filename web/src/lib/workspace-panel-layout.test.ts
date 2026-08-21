import { describe, expect, it } from "vitest"
import {
  WORKSPACE_CHAT_MINIMUM_WIDTH,
  WORKSPACE_PANEL_MAXIMUM_WIDTH,
  WORKSPACE_PANEL_MINIMUM_WIDTH,
  workspacePanelMaximumWidthForViewport,
  workspacePanelWidthForViewport,
} from "./workspace-panel-layout"

describe("workspace panel sizing", () => {
  it("uses all available width on narrow viewports", () => {
    expect(workspacePanelWidthForViewport(600, WORKSPACE_PANEL_MINIMUM_WIDTH, 640)).toBe(640)
    expect(workspacePanelWidthForViewport(600, WORKSPACE_PANEL_MINIMUM_WIDTH, 400)).toBe(400)
  })

  it("clamps the shared preference to panel bounds", () => {
    expect(workspacePanelWidthForViewport(100, WORKSPACE_PANEL_MINIMUM_WIDTH, 1_200)).toBe(WORKSPACE_PANEL_MINIMUM_WIDTH)
    expect(workspacePanelMaximumWidthForViewport(2_000)).toBe(WORKSPACE_PANEL_MAXIMUM_WIDTH)
  })

  it("preserves minimum chat and occupied sidebar space", () => {
    const occupied = WORKSPACE_CHAT_MINIMUM_WIDTH + 260
    expect(workspacePanelMaximumWidthForViewport(1_200, occupied)).toBe(580)
    expect(workspacePanelWidthForViewport(600, WORKSPACE_PANEL_MINIMUM_WIDTH, 1_200, occupied)).toBe(580)
  })
})
