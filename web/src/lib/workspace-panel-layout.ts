import { clampResizableValue } from "./resizable"

export const WORKSPACE_FILES_WIDTH = 320
export const WORKSPACE_DOCUMENT_WIDTH = 500
export const WORKSPACE_BROWSER_WIDTH = 700
export const WORKSPACE_FILES_MINIMUM_WIDTH = 280
export const WORKSPACE_BROWSER_MINIMUM_WIDTH = 480
export const WORKSPACE_PANEL_MAXIMUM_WIDTH = 1_000
export const WORKSPACE_PANEL_FULL_WIDTH_BREAKPOINT = 640
export const WORKSPACE_CHAT_MINIMUM_WIDTH = 440

export type WorkspacePanelWidthMode = "filesTree" | "document" | "browser"

export const workspacePanelMaximumWidthForViewport = (
  viewportWidth: number,
  occupiedWidth = 0,
): number => clampResizableValue(
  Math.min(viewportWidth * 0.65, viewportWidth - occupiedWidth),
  WORKSPACE_FILES_MINIMUM_WIDTH,
  WORKSPACE_PANEL_MAXIMUM_WIDTH,
)

export const workspacePanelWidthForViewport = (
  preferredWidth: number,
  minimumWidth: number,
  viewportWidth: number,
  occupiedWidth = 0,
): number => viewportWidth <= WORKSPACE_PANEL_FULL_WIDTH_BREAKPOINT
  ? Math.max(0, viewportWidth)
  : clampResizableValue(
      preferredWidth,
      minimumWidth,
      Math.max(minimumWidth, workspacePanelMaximumWidthForViewport(viewportWidth, occupiedWidth)),
    )

export const workspaceDocumentWidthForViewport = (
  filesPreferredWidth: number,
  documentPreferredWidth: number,
  viewportWidth: number,
  occupiedWidth = 0,
): number => Math.max(
  workspacePanelWidthForViewport(
    filesPreferredWidth,
    WORKSPACE_FILES_MINIMUM_WIDTH,
    viewportWidth,
    occupiedWidth,
  ),
  workspacePanelWidthForViewport(
    documentPreferredWidth,
    WORKSPACE_FILES_MINIMUM_WIDTH,
    viewportWidth,
    occupiedWidth,
  ),
)
