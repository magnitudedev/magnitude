import { clampResizableValue } from "./resizable"

export const PROJECT_FILES_BROWSER_WIDTH = 320
export const PROJECT_FILES_DOCUMENT_WIDTH = 500
export const PROJECT_FILES_MINIMUM_WIDTH = 280
export const PROJECT_FILES_MAXIMUM_WIDTH = 800
export const PROJECT_FILES_FULL_WIDTH_BREAKPOINT = 640

export type ProjectFilesPanelMode = "browser" | "document"

export const projectFilesMaximumWidthForViewport = (viewportWidth: number): number =>
  clampResizableValue(
    viewportWidth * 0.65,
    PROJECT_FILES_MINIMUM_WIDTH,
    PROJECT_FILES_MAXIMUM_WIDTH,
  )

export const projectFilesWidthForViewport = (
  preferredWidth: number,
  viewportWidth: number,
): number => viewportWidth <= PROJECT_FILES_FULL_WIDTH_BREAKPOINT
  ? Math.max(0, viewportWidth)
  : clampResizableValue(
      preferredWidth,
      PROJECT_FILES_MINIMUM_WIDTH,
      projectFilesMaximumWidthForViewport(viewportWidth),
    )

export const projectFilesDocumentWidthForViewport = (
  browserPreferredWidth: number,
  documentPreferredWidth: number,
  viewportWidth: number,
): number => Math.max(
  projectFilesWidthForViewport(browserPreferredWidth, viewportWidth),
  projectFilesWidthForViewport(documentPreferredWidth, viewportWidth),
)
