import { Schema } from "effect"
import type { BrowserTabId } from "@magnitudedev/client-common"
import type { ProjectId, RelativePath } from "@magnitudedev/sdk"
import { WORKSPACE_PANEL_WIDTH, WORKSPACE_TREE_WIDTH } from "./workspace-panel-layout"

export const WorkspaceTabIdSchema = Schema.NonEmptyString.pipe(
  Schema.brand("WorkspaceTabId"),
)
export type WorkspaceTabId = typeof WorkspaceTabIdSchema.Type

export const makeWorkspaceTabId = (): WorkspaceTabId =>
  WorkspaceTabIdSchema.make(crypto.randomUUID())

export interface WorkspaceFileDocument {
  readonly path: RelativePath
}

export interface WorkspaceFileTab {
  readonly id: WorkspaceTabId
  readonly kind: "file"
  readonly projectId: ProjectId
  readonly document: WorkspaceFileDocument | null
}

export interface WorkspaceBrowserTab {
  readonly id: WorkspaceTabId
  readonly kind: "browser"
  readonly browserTabId: BrowserTabId
}

export type WorkspaceTab = WorkspaceFileTab | WorkspaceBrowserTab

export interface WorkspacePresentation {
  readonly tabs: readonly WorkspaceTab[]
  readonly activeTabId: WorkspaceTabId | null
  readonly treeOpen: boolean
  readonly panelWidth: number
  readonly treeWidth: number
}

export const initialWorkspacePresentation: WorkspacePresentation = {
  tabs: [],
  activeTabId: null,
  treeOpen: false,
  panelWidth: WORKSPACE_PANEL_WIDTH,
  treeWidth: WORKSPACE_TREE_WIDTH,
}

const activateFallback = (
  tabs: readonly WorkspaceTab[],
  closedIndex: number,
): WorkspaceTabId | null => tabs[Math.min(closedIndex, tabs.length - 1)]?.id ?? null

export const addEmptyFileTab = (
  state: WorkspacePresentation,
  id: WorkspaceTabId,
  projectId: ProjectId,
): WorkspacePresentation => ({
  ...state,
  tabs: [...state.tabs, { id, kind: "file", projectId, document: null }],
  activeTabId: id,
  treeOpen: true,
})

export const addBrowserTab = (
  state: WorkspacePresentation,
  id: WorkspaceTabId,
  browserTabId: BrowserTabId,
): WorkspacePresentation => {
  const existing = state.tabs.find(
    (tab): tab is WorkspaceBrowserTab => tab.kind === "browser" && tab.browserTabId === browserTabId,
  )
  if (existing !== undefined) return { ...state, activeTabId: existing.id }
  return {
    ...state,
    tabs: [...state.tabs, { id, kind: "browser", browserTabId }],
    activeTabId: id,
  }
}

export const openWorkspaceFile = (
  state: WorkspacePresentation,
  id: WorkspaceTabId,
  projectId: ProjectId,
  path: RelativePath,
): WorkspacePresentation => {
  const activeIndex = state.tabs.findIndex((tab) => tab.id === state.activeTabId)
  const active = activeIndex < 0 ? undefined : state.tabs[activeIndex]
  if (active?.kind === "file" && active.projectId === projectId) {
    if (active.document?.path === path) return state
    const tabs = [...state.tabs]
    tabs[activeIndex] = { ...active, document: { path } }
    return { ...state, tabs }
  }

  const existing = state.tabs.find(
    (tab): tab is WorkspaceFileTab => tab.kind === "file"
      && tab.projectId === projectId
      && tab.document?.path === path,
  )
  if (existing !== undefined) return { ...state, activeTabId: existing.id }

  return {
    ...state,
    tabs: [...state.tabs, {
      id,
      kind: "file",
      projectId,
      document: { path },
    }],
    activeTabId: id,
  }
}

export const activateWorkspaceTab = (
  state: WorkspacePresentation,
  id: WorkspaceTabId,
): WorkspacePresentation => state.tabs.some((tab) => tab.id === id)
  ? { ...state, activeTabId: id }
  : state

export const closeWorkspaceTab = (
  state: WorkspacePresentation,
  id: WorkspaceTabId,
): WorkspacePresentation => {
  const index = state.tabs.findIndex((tab) => tab.id === id)
  if (index < 0) return state
  const tabs = state.tabs.filter((tab) => tab.id !== id)
  return {
    ...state,
    tabs,
    activeTabId: state.activeTabId === id
      ? activateFallback(tabs, index)
      : state.activeTabId,
  }
}

export const reconcileBrowserTabs = (
  state: WorkspacePresentation,
  browserTabIds: readonly BrowserTabId[],
  newWorkspaceId: (browserTabId: BrowserTabId) => WorkspaceTabId,
): WorkspacePresentation => {
  const available = new Set(browserTabIds)
  const tabs = state.tabs.filter(
    (tab) => tab.kind !== "browser" || available.has(tab.browserTabId),
  )
  const represented = new Set(tabs.flatMap((tab) => tab.kind === "browser" ? [tab.browserTabId] : []))
  for (const browserTabId of browserTabIds) {
    if (!represented.has(browserTabId)) {
      tabs.push({ id: newWorkspaceId(browserTabId), kind: "browser", browserTabId })
    }
  }
  const activeStillExists = tabs.some((tab) => tab.id === state.activeTabId)
  return {
    ...state,
    tabs,
    activeTabId: activeStillExists ? state.activeTabId : tabs.at(-1)?.id ?? null,
  }
}

export const changeWorkspaceProject = (
  state: WorkspacePresentation,
  projectId: ProjectId | null,
): WorkspacePresentation => {
  const tabs = state.tabs.filter((tab) => tab.kind === "browser" || tab.projectId === projectId)
  const activeStillExists = tabs.some((tab) => tab.id === state.activeTabId)
  return {
    ...state,
    tabs,
    activeTabId: activeStillExists ? state.activeTabId : tabs.at(-1)?.id ?? null,
    treeOpen: projectId === null ? false : state.treeOpen,
  }
}

export const replaceWorkspaceFilePath = (
  state: WorkspacePresentation,
  projectId: ProjectId,
  translate: (path: RelativePath) => RelativePath,
): WorkspacePresentation => ({
  ...state,
  tabs: state.tabs.map((tab) => tab.kind === "file" && tab.projectId === projectId && tab.document !== null
    ? { ...tab, document: { ...tab.document, path: translate(tab.document.path) } }
    : tab),
})
