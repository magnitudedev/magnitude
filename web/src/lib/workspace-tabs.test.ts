import { describe, expect, it } from "vitest"
import { BrowserTabIdSchema } from "@magnitudedev/client-common"
import { RelativePathSchema, type ProjectId } from "@magnitudedev/sdk"
import {
  WorkspaceTabIdSchema,
  activateWorkspaceTab,
  addBrowserTab,
  addEmptyFileTab,
  changeWorkspaceProject,
  closeWorkspaceTab,
  initialWorkspacePresentation,
  openWorkspaceFile,
  reconcileBrowserTabs,
  replaceWorkspaceFilePath,
} from "./workspace-tabs"

const projectA = "project-a" as ProjectId
const projectB = "project-b" as ProjectId
const pathA = RelativePathSchema.make("src/a.ts")
const pathB = RelativePathSchema.make("src/b.ts")
const workspaceId = (value: string) => WorkspaceTabIdSchema.make(value)
const browserId = (value: string) => BrowserTabIdSchema.make(value)

describe("workspace tabs", () => {
  it("creates an empty file tab with the tree open and fills it on first selection", () => {
    const empty = addEmptyFileTab(initialWorkspacePresentation, workspaceId("file-a"), projectA)
    const opened = openWorkspaceFile(empty, workspaceId("unused"), projectA, pathA)
    expect(opened.treeOpen).toBe(true)
    expect(opened.tabs).toEqual([{
      id: workspaceId("file-a"),
      kind: "file",
      projectId: projectA,
      document: { path: pathA },
    }])
  })

  it("replaces the active file tab instead of opening or activating another tab", () => {
    const first = openWorkspaceFile(initialWorkspacePresentation, workspaceId("a"), projectA, pathA)
    const secondEmpty = addEmptyFileTab(first, workspaceId("b"), projectA)
    const second = openWorkspaceFile(secondEmpty, workspaceId("unused-b"), projectA, pathB)
    const activeFirst = activateWorkspaceTab(second, workspaceId("a"))
    const replaced = openWorkspaceFile(activeFirst, workspaceId("unused-a"), projectA, pathB)
    expect(replaced.tabs).toEqual([
      { id: workspaceId("a"), kind: "file", projectId: projectA, document: { path: pathB } },
      { id: workspaceId("b"), kind: "file", projectId: projectA, document: { path: pathB } },
    ])
    expect(replaced.activeTabId).toBe(workspaceId("a"))
  })

  it("selects the right neighbor, then the left, when active tabs close", () => {
    const a = openWorkspaceFile(initialWorkspacePresentation, workspaceId("a"), projectA, pathA)
    const b = openWorkspaceFile(addEmptyFileTab(a, workspaceId("b"), projectA), workspaceId("unused"), projectA, pathB)
    const browser = addBrowserTab(b, workspaceId("c"), browserId("browser-c"))
    const activeB = activateWorkspaceTab(browser, workspaceId("b"))
    const closedB = closeWorkspaceTab(activeB, workspaceId("b"))
    expect(closedB.activeTabId).toBe(workspaceId("c"))
    expect(closeWorkspaceTab(closedB, workspaceId("c")).activeTabId).toBe(workspaceId("a"))
  })

  it("represents closing the final tab as a genuine empty workspace", () => {
    const only = openWorkspaceFile(initialWorkspacePresentation, workspaceId("only"), projectA, pathA)
    expect(closeWorkspaceTab(only, workspaceId("only"))).toMatchObject({ tabs: [], activeTabId: null })
  })

  it("reconciles browser membership without disturbing mixed order", () => {
    const file = openWorkspaceFile(initialWorkspacePresentation, workspaceId("file"), projectA, pathA)
    const withBrowser = addBrowserTab(file, workspaceId("browser-a"), browserId("browser-a"))
    const reconciled = reconcileBrowserTabs(
      withBrowser,
      [browserId("browser-b")],
      (id) => workspaceId(`workspace-${id}`),
    )
    expect(reconciled.tabs.map((tab) => tab.kind === "file" ? "file" : tab.browserTabId)).toEqual([
      "file",
      browserId("browser-b"),
    ])
  })

  it("adopts a popup once and preserves an existing native tab's workspace identity", () => {
    const initial = addBrowserTab(
      initialWorkspacePresentation,
      workspaceId("browser-workspace-a"),
      browserId("browser-a"),
    )
    const reconciled = reconcileBrowserTabs(
      initial,
      [browserId("browser-a"), browserId("popup")],
      (id) => workspaceId(`workspace-${id}`),
    )
    const repeated = reconcileBrowserTabs(
      reconciled,
      [browserId("browser-a"), browserId("popup")],
      (id) => workspaceId(`duplicate-${id}`),
    )
    expect(repeated.tabs).toEqual([
      { id: workspaceId("browser-workspace-a"), kind: "browser", browserTabId: browserId("browser-a") },
      { id: workspaceId("workspace-popup"), kind: "browser", browserTabId: browserId("popup") },
    ])
  })

  it("removes prior-project file tabs while retaining browser tabs", () => {
    const file = openWorkspaceFile(initialWorkspacePresentation, workspaceId("file"), projectA, pathA)
    const mixed = addBrowserTab(file, workspaceId("browser"), browserId("browser"))
    const changed = changeWorkspaceProject(mixed, projectB)
    expect(changed.tabs).toEqual([{
      id: workspaceId("browser"),
      kind: "browser",
      browserTabId: browserId("browser"),
    }])
  })

  it("translates every open document beneath an acknowledged move", () => {
    const first = openWorkspaceFile(initialWorkspacePresentation, workspaceId("a"), projectA, pathA)
    const second = openWorkspaceFile(first, workspaceId("b"), projectB, pathB)
    const translated = replaceWorkspaceFilePath(second, projectA, (path) =>
      RelativePathSchema.make(`moved/${path}`))
    expect(translated.tabs).toEqual([
      {
        id: workspaceId("a"),
        kind: "file",
        projectId: projectA,
        document: { path: RelativePathSchema.make("moved/src/a.ts") },
      },
      {
        id: workspaceId("b"),
        kind: "file",
        projectId: projectB,
        document: { path: pathB },
      },
    ])
  })
})
