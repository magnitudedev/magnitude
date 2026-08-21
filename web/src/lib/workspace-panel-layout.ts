import { clampResizableValue } from "./resizable"

export const WORKSPACE_PANEL_WIDTH = 600
export const WORKSPACE_PANEL_MINIMUM_WIDTH = 420
export const WORKSPACE_TREE_WIDTH = 200
export const WORKSPACE_TREE_MINIMUM_WIDTH = 160
export const WORKSPACE_TREE_MAXIMUM_WIDTH = 320
export const WORKSPACE_PANEL_MAXIMUM_WIDTH = 1_000
export const WORKSPACE_PANEL_FULL_WIDTH_BREAKPOINT = 640
export const WORKSPACE_CHAT_MINIMUM_WIDTH = 360

export const workspacePanelMaximumWidthForViewport = (
  viewportWidth: number,
  occupiedWidth = 0,
): number => clampResizableValue(
  Math.min(viewportWidth * 0.65, viewportWidth - occupiedWidth),
  WORKSPACE_PANEL_MINIMUM_WIDTH,
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
