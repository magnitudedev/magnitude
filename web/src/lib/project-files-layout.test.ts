import { describe, expect, it } from "vitest"

import {
  PROJECT_FILES_MAXIMUM_WIDTH,
  PROJECT_FILES_MINIMUM_WIDTH,
  PROJECT_FILES_DOCUMENT_WIDTH,
  projectFilesDocumentWidthForViewport,
  projectFilesMaximumWidthForViewport,
  projectFilesWidthForViewport,
} from "./project-files-layout"

describe("project files panel sizing", () => {
  it("uses the full viewport at the narrow breakpoint", () => {
    expect(projectFilesWidthForViewport(PROJECT_FILES_DOCUMENT_WIDTH, 640)).toBe(640)
    expect(projectFilesWidthForViewport(320, 400)).toBe(400)
  })

  it("clamps preferred widths to the panel bounds", () => {
    expect(projectFilesWidthForViewport(100, 1200)).toBe(PROJECT_FILES_MINIMUM_WIDTH)
    expect(projectFilesWidthForViewport(1_000, 2_000)).toBe(PROJECT_FILES_MAXIMUM_WIDTH)
  })

  it("limits the panel to sixty-five percent of a medium viewport", () => {
    expect(projectFilesMaximumWidthForViewport(1_000)).toBe(650)
    expect(projectFilesWidthForViewport(700, 1_000)).toBe(650)
  })

  it("allows the full configured maximum on wide viewports", () => {
    expect(projectFilesMaximumWidthForViewport(1_600)).toBe(PROJECT_FILES_MAXIMUM_WIDTH)
  })

  it("never makes an open document narrower than the browser", () => {
    expect(projectFilesDocumentWidthForViewport(600, PROJECT_FILES_DOCUMENT_WIDTH, 1_600)).toBe(600)
    expect(projectFilesDocumentWidthForViewport(320, PROJECT_FILES_DOCUMENT_WIDTH, 1_600)).toBe(PROJECT_FILES_DOCUMENT_WIDTH)
  })
})
