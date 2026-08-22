import {
  useCallback,
  useMemo,
  useRef,
  useState,
  useSyncExternalStore,
  type MouseEvent,
  type ReactNode,
} from "react"
import { FilePlus2, FileText, FolderTree, Globe2, PanelRight, Plus, X } from "lucide-react"
import { Atom, useAtomMount, useAtomSet, useAtomValue } from "@effect-atom/atom-react"
import { Effect, Option } from "effect"
import {
  type BrowserTabId,
  type BrowserWorkspaceState,
  type EmbeddedBrowserCapability,
} from "@magnitudedev/client-common"
import type { ProjectId, RelativePath } from "@magnitudedev/sdk"
import { BrowserContent } from "@/components/browser-panel"
import { ProjectFileContent, ProjectTreeDock } from "@/components/project-files/panel"
import { Button } from "@/components/ui/button"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import { ResizableEdge } from "@/components/ui/resizable-edge"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { ActionTooltip } from "@/components/ui/tooltip"
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog"
import { notify } from "@/lib/notifications"
import {
  WORKSPACE_CHAT_MINIMUM_WIDTH,
  WORKSPACE_PANEL_FULL_WIDTH_BREAKPOINT,
  WORKSPACE_PANEL_MINIMUM_WIDTH,
  WORKSPACE_TREE_MAXIMUM_WIDTH,
  WORKSPACE_TREE_MINIMUM_WIDTH,
  workspacePanelMaximumWidthForViewport,
  workspacePanelWidthForViewport,
} from "@/lib/workspace-panel-layout"
import {
  activateWorkspaceTab,
  addBrowserTab,
  addEmptyFileTab,
  changeWorkspaceProject,
  closeWorkspaceTab,
  makeWorkspaceTabId,
  openWorkspaceFile,
  reconcileBrowserTabs,
  type WorkspaceFileTab,
  type WorkspaceTab,
  type WorkspaceTabId,
} from "@/lib/workspace-tabs"
import {
  projectFileDraftsAtom,
  sidebarCollapsedAtom,
  sidebarWidthAtom,
  workspacePanelEnteringAtom,
  workspacePanelOpenAtom,
  workspacePresentationAtom,
} from "@/state/web-atoms"

const windowWidth = () => typeof window === "undefined" ? 1_440 : window.innerWidth
const subscribeWindow = (listener: () => void) => {
  window.addEventListener("resize", listener)
  return () => window.removeEventListener("resize", listener)
}
const emptyBrowserState: BrowserWorkspaceState = {
  revision: 0,
  focusLocationRevision: 0,
  activeTabId: Option.none(),
  tabs: [],
  permissionRequest: null,
  downloads: [],
}
const emptySubscribe = () => () => undefined
const emptySnapshot = () => emptyBrowserState

const runCommand = <A,>(command: Promise<A>, failure: string, onSuccess?: (value: A) => void): void => {
  void command.then(onSuccess).catch((cause: unknown) => {
    console.error(`[workspace] ${failure}`, cause)
    notify("error", failure)
  })
}

type PendingDiscard =
  | { readonly kind: "close"; readonly tab: WorkspaceFileTab }
  | { readonly kind: "open"; readonly tab: WorkspaceFileTab; readonly path: RelativePath }

function WorkspaceTabView({
  tab,
  active,
  browserState,
  dirty,
  onClose,
}: {
  readonly tab: WorkspaceTab
  readonly active: boolean
  readonly browserState: BrowserWorkspaceState
  readonly dirty: boolean
  readonly onClose: (event: MouseEvent<HTMLButtonElement>) => void
}): ReactNode {
  const browserTab = tab.kind === "browser"
    ? browserState.tabs.find((candidate) => candidate.id === tab.browserTabId)
    : undefined
  const title = tab.kind === "file"
    ? tab.document?.path.split("/").at(-1) ?? "Select a file"
    : browserTab?.title ?? "Browser"
  return (
    <div className={`group flex h-8 min-w-20 max-w-40 items-center rounded-md font-sans text-xs transition-colors ${active ? "bg-slate-200 text-slate-950 shadow-sm dark:bg-slate-750 dark:text-white" : "bg-slate-100 text-slate-600 hover:bg-slate-150 hover:text-slate-900 dark:bg-slate-800 dark:text-slate-300 dark:hover:bg-slate-750 dark:hover:text-white"}`}>
      <TabsTrigger
        value={tab.id}
        data-workspace-tab-kind={tab.kind}
        title={tab.kind === "file" ? tab.document?.path ?? "Select a file" : title}
        className="h-full min-w-0 flex-1 justify-start gap-1.5 border-0 bg-transparent! px-2 text-inherit! shadow-none hover:bg-transparent! hover:text-inherit! after:hidden data-active:bg-transparent! dark:bg-transparent! dark:text-inherit! dark:hover:bg-transparent! dark:hover:text-inherit! dark:data-active:bg-transparent!"
      >
        {tab.kind === "file" ? <FileText size={14} />
          : browserTab?.phase === "loading" ? <span className="size-3.5 animate-spin rounded-full border-2 border-slate-300 border-t-blue-500 motion-reduce:animate-none dark:border-slate-600 dark:border-t-blue-400" />
            : browserTab?.faviconUrl ? <img src={browserTab.faviconUrl} alt="" className="size-3.5 shrink-0" />
              : <Globe2 size={14} />}
        <span className="min-w-0 flex-1 truncate text-left">{title}</span>
        {dirty ? <span className="size-1.5 shrink-0 rounded-full bg-blue-500" aria-label="Unsaved changes" /> : null}
      </TabsTrigger>
      <button type="button" aria-label={`Close ${title}`} onClick={onClose} className={`mr-1 flex size-5 shrink-0 items-center justify-center rounded hover:bg-slate-200 focus:opacity-100 group-hover:opacity-100 dark:hover:bg-slate-700 ${active ? "opacity-100" : "opacity-0"}`}><X size={12} /></button>
    </div>
  )
}

export function WorkspacePanel({
  projectId,
  browser,
}: {
  readonly projectId: ProjectId | null
  readonly browser: EmbeddedBrowserCapability | undefined
}): ReactNode {
  const panelRef = useRef<HTMLDivElement>(null)
  const closingRef = useRef(false)
  const [resizing, setResizing] = useState(false)
  const [pendingDiscard, setPendingDiscard] = useState<PendingDiscard | null>(null)
  const workspace = useAtomValue(workspacePresentationAtom)
  const setWorkspace = useAtomSet(workspacePresentationAtom)
  const drafts = useAtomValue(projectFileDraftsAtom)
  const setDrafts = useAtomSet(projectFileDraftsAtom)
  const setOpen = useAtomSet(workspacePanelOpenAtom)
  const entering = useAtomValue(workspacePanelEnteringAtom)
  const setEntering = useAtomSet(workspacePanelEnteringAtom)
  const sidebarCollapsed = useAtomValue(sidebarCollapsedAtom)
  const sidebarWidth = useAtomValue(sidebarWidthAtom)
  const viewportWidth = useSyncExternalStore(subscribeWindow, windowWidth, windowWidth)
  const browserState = useSyncExternalStore(
    browser?.subscribe ?? emptySubscribe,
    browser?.getSnapshot ?? emptySnapshot,
    browser?.getSnapshot ?? emptySnapshot,
  )
  const activeTab = workspace.tabs.find((tab) => tab.id === workspace.activeTabId) ?? null
  const activeBrowserTab = activeTab?.kind === "browser"
    ? browserState.tabs.find((tab) => tab.id === activeTab.browserTabId) ?? null
    : null
  const occupiedWidth = WORKSPACE_CHAT_MINIMUM_WIDTH + (sidebarCollapsed ? 0 : sidebarWidth)
  const fullWidth = viewportWidth <= WORKSPACE_PANEL_FULL_WIDTH_BREAKPOINT
  const maximumWidth = workspacePanelMaximumWidthForViewport(viewportWidth, occupiedWidth)
  const panelWidth = workspacePanelWidthForViewport(
    workspace.panelWidth,
    WORKSPACE_PANEL_MINIMUM_WIDTH,
    viewportWidth,
    occupiedWidth,
  )
  const treeMaximum = Math.min(WORKSPACE_TREE_MAXIMUM_WIDTH, Math.max(WORKSPACE_TREE_MINIMUM_WIDTH, panelWidth - 240))
  const treeWidth = Math.min(workspace.treeWidth, treeMaximum)

  const projectLifecycleAtom = useMemo(
    () => Atom.make(Effect.sync(() => setWorkspace((current) => changeWorkspaceProject(current, projectId)))),
    [projectId, setWorkspace],
  )
  useAtomMount(projectLifecycleAtom)

  const fileTabIds = workspace.tabs.flatMap((tab) => tab.kind === "file" ? [tab.id] : []).join("\0")
  const draftLifecycleAtom = useMemo(
    () => Atom.make(Effect.sync(() => setDrafts((current) => {
      const openIds = new Set(fileTabIds === "" ? [] : fileTabIds.split("\0"))
      const retained = Object.entries(current).filter(([id]) => openIds.has(id))
      return retained.length === Object.keys(current).length
        ? current
        : Object.fromEntries(retained)
    }))),
    [fileTabIds, setDrafts],
  )
  useAtomMount(draftLifecycleAtom)

  const browserIds = browserState.tabs.map((tab) => tab.id).join("\0")
  const browserLifecycleAtom = useMemo(
    () => Atom.make(Effect.sync(() => {
      setWorkspace((current) => {
        const reconciled = reconcileBrowserTabs(
          current,
          browserState.tabs.map((tab) => tab.id),
          () => makeWorkspaceTabId(),
        )
        const browserActiveId = Option.getOrNull(browserState.activeTabId)
        const currentActive = reconciled.tabs.find((tab) => tab.id === reconciled.activeTabId)
        if (currentActive?.kind !== "browser" || browserActiveId === null || currentActive.browserTabId === browserActiveId) return reconciled
        const matching = reconciled.tabs.find((tab) => tab.kind === "browser" && tab.browserTabId === browserActiveId)
        return matching === undefined ? reconciled : activateWorkspaceTab(reconciled, matching.id)
      })
    })),
    [browserIds, browserState.activeTabId, setWorkspace],
  )
  useAtomMount(browserLifecycleAtom)

  const collapse = useCallback(() => {
    if (closingRef.current) return
    setEntering(false)
    void browser?.setViewport(null)
    const panel = panelRef.current
    if (panel === null || window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
      setOpen(false)
      return
    }
    closingRef.current = true
    const width = panel.getBoundingClientRect().width
    const animation = panel.animate([{ width: `${width}px` }, { width: "0px" }], { duration: 150, easing: "ease-out", fill: "forwards" })
    void animation.finished.then(() => setOpen(false), () => setOpen(false))
  }, [browser, setEntering, setOpen])

  const activate = useCallback((tab: WorkspaceTab) => {
    setWorkspace((current) => activateWorkspaceTab(current, tab.id))
    if (tab.kind === "browser" && browser !== undefined) runCommand(browser.activateTab(tab.browserTabId), "Could not switch tabs.")
    else void browser?.setViewport(null)
  }, [browser, setWorkspace])

  const closeTab = useCallback((tab: WorkspaceTab) => {
    if (tab.kind === "file") {
      if (drafts[tab.id] !== undefined) {
        setPendingDiscard({ kind: "close", tab })
        return
      }
      setWorkspace((current) => closeWorkspaceTab(current, tab.id))
      return
    }
    if (browser === undefined) return
    runCommand(browser.closeTab(tab.browserTabId), "Could not close this tab.", () => {
      setWorkspace((current) => closeWorkspaceTab(current, tab.id))
    })
  }, [browser, drafts, setWorkspace])

  const openFile = useCallback((path: RelativePath) => {
    if (projectId === null) return
    if (
      activeTab?.kind === "file"
      && activeTab.document?.path !== path
      && drafts[activeTab.id] !== undefined
    ) {
      setPendingDiscard({ kind: "open", tab: activeTab, path })
      return
    }
    setWorkspace((current) => openWorkspaceFile(current, makeWorkspaceTabId(), projectId, path))
  }, [activeTab, drafts, projectId, setWorkspace])

  return (
    <div
      ref={panelRef}
      style={{ width: fullWidth ? "100%" : panelWidth }}
      onAnimationEnd={(event) => {
        if (event.currentTarget === event.target) setEntering(false)
      }}
      className={`relative flex shrink-0 overflow-visible ${entering && !resizing ? "animate-[workspace-panel-open_150ms_ease-out]" : ""}`}
    >
      {!fullWidth ? (
        <ResizableEdge
          side="left"
          placement="outside"
          value={panelWidth}
          minimum={WORKSPACE_PANEL_MINIMUM_WIDTH}
          maximum={maximumWidth}
          onValueChange={(width) => setWorkspace((current) => ({ ...current, panelWidth: width }))}
          onDraggingChange={setResizing}
          label="Resize workspace"
        />
      ) : null}
      <aside aria-label="Workspace" className="flex min-w-0 flex-1 flex-col overflow-hidden border-l border-slate-200 bg-white dark:border-slate-800 dark:bg-slate-850">
      <Tabs value={workspace.activeTabId ?? ""} onValueChange={(id) => {
        const tab = workspace.tabs.find((candidate) => candidate.id === id)
        if (tab !== undefined) activate(tab)
      }} className="flex min-h-0 flex-1 gap-0">
      <header className="flex h-11 shrink-0 select-none items-end gap-1 border-b border-slate-200 px-2 pt-1 dark:border-slate-800 [-webkit-app-region:drag]">
        <ActionTooltip label="Collapse sidebar" side="bottom" disabled={entering} trigger={(
          <Button variant="ghost" size="icon-sm" onClick={collapse} className="mb-1 [-webkit-app-region:no-drag]" aria-label="Collapse sidebar"><PanelRight size={18} /></Button>
        )} />
        <TabsList activateOnFocus aria-label="Workspace tabs" variant="line" className="h-9 min-w-0 max-w-full flex-1 justify-start gap-0 overflow-hidden p-0 [-webkit-app-region:no-drag]">
            <div className="flex h-9 min-w-0 max-w-[calc(100%-2.25rem)] shrink items-center gap-1 overflow-x-auto [scrollbar-width:none] [&::-webkit-scrollbar]:hidden">
              {workspace.tabs.map((tab) => (
                <WorkspaceTabView
                  key={tab.id}
                  tab={tab}
                  active={tab.id === workspace.activeTabId}
                  browserState={browserState}
                  dirty={tab.kind === "file" && drafts[tab.id] !== undefined}
                  onClose={(event) => {
                    event.stopPropagation()
                    closeTab(tab)
                  }}
                />
              ))}
            </div>
            <DropdownMenu>
              <DropdownMenuTrigger render={<Button variant="ghost" size="icon-sm" className="ml-1 shrink-0" aria-label="New workspace tab" />}><Plus size={15} /></DropdownMenuTrigger>
              <DropdownMenuContent align="start" className="w-44">
                <DropdownMenuItem disabled={projectId === null} onClick={() => {
                  if (projectId !== null) setWorkspace((current) => addEmptyFileTab(current, makeWorkspaceTabId(), projectId))
                }}><FilePlus2 />File</DropdownMenuItem>
                {browser !== undefined ? <DropdownMenuItem onClick={() => runCommand(browser.createTab(), "Could not create a browser tab.", (browserTabId) => {
                  setWorkspace((current) => addBrowserTab(current, makeWorkspaceTabId(), browserTabId))
                })}><Globe2 />Browser</DropdownMenuItem> : null}
              </DropdownMenuContent>
            </DropdownMenu>
          </TabsList>
        <ActionTooltip label={workspace.treeOpen ? "Hide project files" : "Show project files"} side="bottom" trigger={(
          <Button variant="ghost" size="icon-sm" className="mb-1 aria-pressed:bg-slate-200 aria-pressed:text-slate-900 dark:aria-pressed:bg-slate-700 dark:aria-pressed:text-white [-webkit-app-region:no-drag]" disabled={projectId === null} aria-label={workspace.treeOpen ? "Hide project files" : "Show project files"} aria-pressed={workspace.treeOpen} onClick={() => setWorkspace((current) => ({ ...current, treeOpen: !current.treeOpen }))}><FolderTree size={17} /></Button>
        )} />
        <Button variant="ghost" size="icon-sm" className="mb-1 [-webkit-app-region:no-drag]" onClick={collapse} aria-label="Close sidebar"><X size={16} /></Button>
      </header>
      <div className="flex min-h-0 flex-1">
        {activeTab === null ? (
          <div className="flex min-w-0 flex-1 flex-col">
            <div className="flex min-h-0 flex-1 flex-col items-center justify-center gap-3 px-6 text-center">
              <div className="font-mono text-base font-semibold text-slate-900 dark:text-slate-100">Open a workspace tab</div>
              <div className="font-sans text-sm text-slate-500 dark:text-slate-400">Use the plus button to open a file or browser.</div>
            </div>
          </div>
        ) : (
          <TabsContent value={activeTab.id} className="flex min-w-0 flex-1 flex-col">
            {activeTab.kind === "file" ? (
              <ProjectFileContent tab={activeTab} onDeleted={() => setWorkspace((current) => closeWorkspaceTab(current, activeTab.id))} />
            ) : browser !== undefined && activeBrowserTab !== null ? (
              <BrowserContent browser={browser} state={browserState} activeTab={activeBrowserTab} />
            ) : (
              <div className="flex min-h-0 flex-1 items-center justify-center font-sans text-sm text-slate-500">This browser tab is no longer available.</div>
            )}
          </TabsContent>
        )}
        {workspace.treeOpen && projectId !== null ? (
          <div style={{ width: treeWidth }} className="relative flex shrink-0 border-l border-slate-200 dark:border-slate-800">
            <ResizableEdge
              side="left"
              value={treeWidth}
              minimum={WORKSPACE_TREE_MINIMUM_WIDTH}
              maximum={treeMaximum}
              onValueChange={(width) => setWorkspace((current) => ({ ...current, treeWidth: width }))}
              onDraggingChange={setResizing}
              label="Resize project files"
            />
            <ProjectTreeDock
              projectId={projectId}
              selectedPath={activeTab?.kind === "file" ? activeTab.document?.path ?? null : null}
              onOpenFile={openFile}
            />
          </div>
        ) : null}
      </div>
      </Tabs>
      <AlertDialog open={pendingDiscard !== null} onOpenChange={(open) => {
        if (!open) setPendingDiscard(null)
      }}>
        <AlertDialogContent className="max-w-[400px]">
          <AlertDialogHeader>
            <AlertDialogTitle>Discard unsaved changes?</AlertDialogTitle>
            <AlertDialogDescription>Your edits to {pendingDiscard?.tab.document?.path ?? "this file"} have not been saved.</AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Keep editing</AlertDialogCancel>
            <AlertDialogAction variant="destructive" onClick={() => {
              if (pendingDiscard === null) return
              setDrafts((current) => {
                const { [pendingDiscard.tab.id]: _removed, ...rest } = current
                return rest
              })
              setWorkspace((current) => pendingDiscard.kind === "close"
                ? closeWorkspaceTab(current, pendingDiscard.tab.id)
                : openWorkspaceFile(
                    activateWorkspaceTab(current, pendingDiscard.tab.id),
                    makeWorkspaceTabId(),
                    pendingDiscard.tab.projectId,
                    pendingDiscard.path,
                  ))
              setPendingDiscard(null)
            }}>Discard changes</AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
      </aside>
    </div>
  )
}
