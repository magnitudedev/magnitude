import { describe, expect, it } from "vitest"

import {
  WORKSPACE_BROWSER_MINIMUM_WIDTH,
  WORKSPACE_CHAT_MINIMUM_WIDTH,
  WORKSPACE_DOCUMENT_WIDTH,
  WORKSPACE_FILES_MINIMUM_WIDTH,
  WORKSPACE_PANEL_MAXIMUM_WIDTH,
  workspaceDocumentWidthForViewport,
  workspacePanelMaximumWidthForViewport,
  workspacePanelWidthForViewport,
} from "./workspace-panel-layout"

describe("workspace panel sizing", () => {
  it("uses the full viewport at the narrow breakpoint", () => {
    expect(workspacePanelWidthForViewport(WORKSPACE_DOCUMENT_WIDTH, WORKSPACE_FILES_MINIMUM_WIDTH, 640)).toBe(640)
    expect(workspacePanelWidthForViewport(320, WORKSPACE_FILES_MINIMUM_WIDTH, 400)).toBe(400)
  })

  it("clamps file and browser widths to their own minimum", () => {
    expect(workspacePanelWidthForViewport(100, WORKSPACE_FILES_MINIMUM_WIDTH, 1_200)).toBe(WORKSPACE_FILES_MINIMUM_WIDTH)
    expect(workspacePanelWidthForViewport(100, WORKSPACE_BROWSER_MINIMUM_WIDTH, 1_200)).toBe(WORKSPACE_BROWSER_MINIMUM_WIDTH)
  })

  it("limits the panel to sixty-five percent of a medium viewport", () => {
    expect(workspacePanelMaximumWidthForViewport(1_000)).toBe(650)
    expect(workspacePanelWidthForViewport(700, WORKSPACE_BROWSER_MINIMUM_WIDTH, 1_000)).toBe(650)
  })

  it("allows the configured maximum on wide viewports", () => {
    expect(workspacePanelMaximumWidthForViewport(2_000)).toBe(WORKSPACE_PANEL_MAXIMUM_WIDTH)
  })

  it("reserves room for the chat and an expanded left sidebar", () => {
    const occupied = WORKSPACE_CHAT_MINIMUM_WIDTH + 260
    expect(workspacePanelMaximumWidthForViewport(1_200, occupied)).toBe(500)
    expect(workspacePanelWidthForViewport(700, WORKSPACE_BROWSER_MINIMUM_WIDTH, 1_200, occupied)).toBe(500)
    expect(workspacePanelWidthForViewport(700, WORKSPACE_BROWSER_MINIMUM_WIDTH, 1_200, WORKSPACE_CHAT_MINIMUM_WIDTH)).toBe(700)
  })

  it("never makes an open document narrower than the file tree", () => {
    expect(workspaceDocumentWidthForViewport(600, WORKSPACE_DOCUMENT_WIDTH, 1_600)).toBe(600)
    expect(workspaceDocumentWidthForViewport(320, WORKSPACE_DOCUMENT_WIDTH, 1_600)).toBe(WORKSPACE_DOCUMENT_WIDTH)
  })
})
