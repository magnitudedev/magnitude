import { useCallback, useMemo, useRef, useState, useSyncExternalStore, lazy, Suspense, type ReactNode } from "react"
import {
  Tree,
  type CursorProps,
  type DragPreviewProps,
  type NodeRendererProps,
} from "react-arborist"
import { CircleAlert, ChevronDown, ChevronRight, File, Folder, FolderOpen, ArrowLeft, Ellipsis, PanelRight, Trash2, X } from "lucide-react"
import { Cause, Effect, Option } from "effect"
import { Atom, Result, useAtomMount, useAtomSet, useAtomValue } from "@effect-atom/atom-react"
import {
  usePlatform,
  useProjectDirectoryTree,
  useProjectDirectoryRefresh,
  useProjectEntryMove,
  useProjectFileDelete,
  useProjectFile,
  useProjectFileSave,
  useProjectFilesWatch,
} from "@magnitudedev/client-common"
import { RelativePathSchema, type ProjectDirectoryEntry, type ProjectId, type RelativePath } from "@magnitudedev/sdk"
import { Button } from "@/components/ui/button"
import { Spinner } from "@/components/ui/spinner"
import { ResizableEdge } from "@/components/ui/resizable-edge"
import { ActionTooltip } from "@/components/ui/tooltip"
import { WorkspacePanelHeader } from "@/components/workspace-panel-header"
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
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import {
  expandedProjectDirectoriesAtom,
  projectFileDiscardIntentAtom,
  workspacePanelEnteringAtom,
  workspacePanelOpenAtom,
  workspacePanelSurfaceAtom,
  workspacePanelWidthsAtom,
  projectFileDirtyAtom,
  projectFileDraftsAtom,
  selectedProjectFileAtom,
  sidebarCollapsedAtom,
  sidebarWidthAtom,
} from "@/state/web-atoms"
import {
  WORKSPACE_CHAT_MINIMUM_WIDTH,
  WORKSPACE_FILES_MINIMUM_WIDTH,
  WORKSPACE_PANEL_FULL_WIDTH_BREAKPOINT,
  workspaceDocumentWidthForViewport,
  workspacePanelMaximumWidthForViewport,
  workspacePanelWidthForViewport,
  type WorkspacePanelWidthMode,
} from "@/lib/workspace-panel-layout"
import {
  canMoveProjectEntryToDirectory,
  parentProjectPath,
  translateProjectPath,
} from "@/lib/project-file-paths"

const ProjectTextEditor = lazy(() => import("./editor").then((module) => ({ default: module.ProjectTextEditor })))

interface FileNode {
  readonly id: string
  readonly name: string
  readonly path: RelativePath
  readonly kind: "directory" | "file"
  readonly loadState?: "idle" | "loading" | "loaded" | "failed"
  readonly children?: readonly FileNode[]
  readonly openFile: (path: RelativePath) => void
  readonly schedulePrefetch: (path: RelativePath) => void
  readonly cancelPrefetch: (path: RelativePath) => void
  readonly retryDirectory: (path: RelativePath) => void
}

const DIRECTORY_PREFETCH_DELAY_MS = 200
const DIRECTORY_PREFETCH_MOUNT_MS = 1_000
const MAX_ACTIVE_PREFETCHES = 2

const windowHeight = () => typeof window === "undefined" ? 800 : window.innerHeight
const windowWidth = () => typeof window === "undefined" ? 1440 : window.innerWidth
const subscribeWindow = (listener: () => void) => {
  window.addEventListener("resize", listener)
  return () => window.removeEventListener("resize", listener)
}

const buildNodes = (
  entries: readonly ProjectDirectoryEntry[],
  listings: Readonly<Record<string, { readonly entries: readonly ProjectDirectoryEntry[] }>>,
  loadStates: Readonly<Record<string, NonNullable<FileNode["loadState"]>>>,
  openFile: (path: RelativePath) => void,
  schedulePrefetch: (path: RelativePath) => void,
  cancelPrefetch: (path: RelativePath) => void,
  retryDirectory: (path: RelativePath) => void,
): readonly FileNode[] => entries.map((entry) => ({
  id: entry.path,
  name: entry.name,
  path: entry.path,
  kind: entry.kind,
  openFile,
  schedulePrefetch,
  cancelPrefetch,
  retryDirectory,
  ...(entry.kind === "directory" ? {
    loadState: loadStates[entry.path] ?? "idle",
    children: buildNodes(
      listings[entry.path]?.entries ?? [],
      listings,
      loadStates,
      openFile,
      schedulePrefetch,
      cancelPrefetch,
      retryDirectory,
    ),
  } : {}),
}))

function TreeNode({ node, style, dragHandle }: NodeRendererProps<FileNode>): ReactNode {
  const Icon = node.data.kind === "directory" ? (node.isOpen ? FolderOpen : Folder) : File
  return (
    <div
      ref={dragHandle}
      style={style}
      className={`group flex cursor-default items-center gap-1.5 rounded px-1.5 text-[13px] ${node.willReceiveDrop ? "bg-blue-200 text-slate-900 ring-1 ring-inset ring-blue-400 dark:bg-blue-900 dark:text-slate-100 dark:ring-blue-600" : node.isSelected ? "bg-blue-100 text-slate-900 dark:bg-blue-900 dark:text-slate-100" : "text-slate-700 hover:bg-slate-150 dark:text-slate-300 dark:hover:bg-slate-800"}`}
      onClick={() => {
        if (node.isInternal) {
          if (node.data.loadState === "failed") {
            node.data.retryDirectory(node.data.path)
            if (!node.isOpen) node.open()
          } else {
            node.toggle()
          }
        } else {
          node.select()
          node.data.openFile(node.data.path)
        }
      }}
      onDoubleClick={() => node.isInternal && node.toggle()}
      onMouseEnter={() => {
        if (node.data.kind === "directory" && node.data.loadState === "idle") {
          node.data.schedulePrefetch(node.data.path)
        }
      }}
      onMouseLeave={() => {
        if (node.data.kind === "directory") node.data.cancelPrefetch(node.data.path)
      }}
    >
      <span className="flex size-4 shrink-0 items-center justify-center text-slate-500 dark:text-slate-400">
        {node.isInternal ? (
          node.isOpen && node.data.loadState === "loading"
            ? <Spinner className="size-3.5 text-blue-600 motion-reduce:animate-none dark:text-blue-400" />
            : node.data.loadState === "failed"
              ? <CircleAlert className="size-3.5 text-red-600 dark:text-red-400" aria-label="Could not load folder" />
              : node.isOpen ? <ChevronDown size={13} /> : <ChevronRight size={13} />
        ) : null}
      </span>
      <Icon size={15} className={node.data.kind === "directory" ? "text-blue-600 dark:text-blue-400" : "text-slate-500 dark:text-slate-400"} />
      <span className="min-w-0 truncate">{node.data.name}</span>
    </div>
  )
}

function TreeDragPreview({
  offset,
  id,
  isDragging,
  kind,
}: DragPreviewProps & { readonly kind?: FileNode["kind"] }): ReactNode {
  if (!isDragging || offset === null || id === null) return null
  const name = id.slice(id.lastIndexOf("/") + 1)
  const Icon = kind === "directory" ? Folder : File
  return (
    <div className="pointer-events-none fixed inset-0 z-50">
      <div
        style={{ transform: `translate(${offset.x + 8}px, ${offset.y + 8}px)` }}
        className="absolute left-0 top-0 flex max-w-64 items-center gap-2 rounded-md border border-slate-300 bg-white px-2.5 py-1.5 text-[13px] text-slate-800 shadow-sm dark:border-slate-700 dark:bg-slate-800 dark:text-slate-100"
      >
        <Icon size={14} className="shrink-0 text-blue-600 dark:text-blue-400" />
        <span className="truncate">{name}</span>
      </div>
    </div>
  )
}

function TreeDropCursor({ top, left, indent }: CursorProps): ReactNode {
  return (
    <div
      aria-hidden="true"
      style={{ top: top - 1, left, right: indent }}
      className="pointer-events-none absolute z-10 h-0.5 rounded-full bg-blue-500 dark:bg-blue-400"
    />
  )
}

export function ProjectFilesPanel({ projectId }: { readonly projectId: ProjectId }): ReactNode {
  return <ProjectFilesPanelContent key={projectId} projectId={projectId} />
}

function ProjectFilesPanelContent({ projectId }: { readonly projectId: ProjectId }): ReactNode {
  const platform = usePlatform()
  useProjectFilesWatch(projectId)
  const panelRef = useRef<HTMLElement>(null)
  const closingRef = useRef(false)
  const [deleteOpen, setDeleteOpen] = useState(false)
  const [resizing, setResizing] = useState(false)
  const [prefetchedDirectories, setPrefetchedDirectories] = useState<ReadonlySet<RelativePath>>(() => new Set())
  const refreshDirectory = useProjectDirectoryRefresh()
  const scheduledPrefetchRef = useRef<{ readonly path: RelativePath; readonly timer: number } | null>(null)
  const prefetchExpiryTimersRef = useRef(new Map<RelativePath, number>())
  const close = useAtomSet(workspacePanelOpenAtom)
  const entering = useAtomValue(workspacePanelEnteringAtom)
  const setEntering = useAtomSet(workspacePanelEnteringAtom)
  const setSurface = useAtomSet(workspacePanelSurfaceAtom)
  const panelWidths = useAtomValue(workspacePanelWidthsAtom)
  const setPanelWidths = useAtomSet(workspacePanelWidthsAtom)
  const discardIntent = useAtomValue(projectFileDiscardIntentAtom)
  const setDiscardIntent = useAtomSet(projectFileDiscardIntentAtom)
  const dirty = useAtomValue(projectFileDirtyAtom)
  const setDirty = useAtomSet(projectFileDirtyAtom)
  const drafts = useAtomValue(projectFileDraftsAtom)
  const setDrafts = useAtomSet(projectFileDraftsAtom)
  const selectedFile = useAtomValue(selectedProjectFileAtom)
  const setSelectedFile = useAtomSet(selectedProjectFileAtom)
  const selectedPath = selectedFile?.projectId === projectId ? selectedFile.path : null
  const expandedByProject = useAtomValue(expandedProjectDirectoriesAtom)
  const setExpandedByProject = useAtomSet(expandedProjectDirectoriesAtom)
  const expandedDirectories = expandedByProject[projectId] ?? new Set<RelativePath>()
  const cancelScheduledPrefetch = useCallback((path?: RelativePath) => {
    const scheduled = scheduledPrefetchRef.current
    if (scheduled === null || (path !== undefined && scheduled.path !== path)) return
    window.clearTimeout(scheduled.timer)
    scheduledPrefetchRef.current = null
  }, [])
  const beginPrefetch = useCallback((path: RelativePath) => {
    setPrefetchedDirectories((current) => {
      if (current.has(path)) return current
      const next = new Set(current)
      while (next.size >= MAX_ACTIVE_PREFETCHES) next.delete(next.values().next().value!)
      next.add(path)
      return next
    })
    const previousExpiry = prefetchExpiryTimersRef.current.get(path)
    if (previousExpiry !== undefined) window.clearTimeout(previousExpiry)
    const expiry = window.setTimeout(() => {
      prefetchExpiryTimersRef.current.delete(path)
      setPrefetchedDirectories((current) => {
        if (!current.has(path)) return current
        const next = new Set(current)
        next.delete(path)
        return next
      })
    }, DIRECTORY_PREFETCH_MOUNT_MS)
    prefetchExpiryTimersRef.current.set(path, expiry)
  }, [])
  const schedulePrefetch = useCallback((path: RelativePath) => {
    cancelScheduledPrefetch()
    scheduledPrefetchRef.current = {
      path,
      timer: window.setTimeout(() => {
        scheduledPrefetchRef.current = null
        beginPrefetch(path)
      }, DIRECTORY_PREFETCH_DELAY_MS),
    }
  }, [beginPrefetch, cancelScheduledPrefetch])
  const prefetchLifecycleAtom = useMemo(
    () => Atom.make(Effect.addFinalizer(() => Effect.sync(() => {
        cancelScheduledPrefetch()
        for (const timer of prefetchExpiryTimersRef.current.values()) window.clearTimeout(timer)
        prefetchExpiryTimersRef.current.clear()
      }))),
    [cancelScheduledPrefetch],
  )
  useAtomMount(prefetchLifecycleAtom)
  const collapsePanel = useCallback(() => {
    if (closingRef.current) return
    setEntering(false)
    const panel = panelRef.current
    if (panel === null || window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
      close(false)
      return
    }
    closingRef.current = true
    const width = panel.getBoundingClientRect().width
    const animation = panel.animate(
      [{ width: `${width}px` }, { width: "0px" }],
      { duration: 150, easing: "ease-out", fill: "forwards" },
    )
    void animation.finished.then(
      () => close(false),
      () => close(false),
    )
  }, [close, setEntering])
  const handleCollapse = useCallback(() => {
    if (dirty) {
      setDiscardIntent("close")
      return
    }
    setDirty(false)
    collapsePanel()
  }, [collapsePanel, dirty, setDirty, setDiscardIntent])
  const setSelectedPath = useCallback((path: RelativePath | null) => {
    setSelectedFile(path === null ? null : { projectId, path })
  }, [projectId, setSelectedFile])
  const retryDirectory = useCallback(
    (directory: RelativePath) => refreshDirectory({ projectId, directory }),
    [projectId, refreshDirectory],
  )
  const rootPath = RelativePathSchema.make("")
  const demandedDirectories = useMemo(
    () => new Set([...expandedDirectories, ...prefetchedDirectories]),
    [expandedDirectories, prefetchedDirectories],
  )
  const directoryTree = useProjectDirectoryTree(projectId, rootPath, demandedDirectories, expandedDirectories)
  const file = useProjectFile(selectedPath === null ? null : { projectId, path: selectedPath })
  const draftKey = selectedPath === null ? null : `${projectId}\0${selectedPath}`
  const handleSaveSuccess = useCallback((saved: { readonly path: RelativePath }) => {
    const savedDraftKey = `${projectId}\0${saved.path}`
    setDrafts((current) => {
      const { [savedDraftKey]: _removed, ...rest } = current
      return rest
    })
    setDirty(false)
  }, [projectId, setDirty, setDrafts])
  const { result: saveResult, save } = useProjectFileSave({ onSuccess: handleSaveSuccess })
  const handleDeleteSuccess = useCallback(() => {
    if (draftKey !== null) {
      setDrafts((current) => {
        const { [draftKey]: _removed, ...rest } = current
        return rest
      })
    }
    setDeleteOpen(false)
    setDirty(false)
    setSelectedPath(null)
  }, [draftKey, setDirty, setDrafts, setSelectedPath])
  const { result: deleteResult, remove } = useProjectFileDelete({ onSuccess: handleDeleteSuccess })
  const handleMoveSuccess = useCallback((moved: {
    readonly sourcePath: RelativePath
    readonly destinationPath: RelativePath
    readonly kind: "directory" | "file"
  }) => {
    setSelectedFile((current) => current?.projectId === projectId
      ? { ...current, path: translateProjectPath(current.path, moved.sourcePath, moved.destinationPath) }
      : current)
    setExpandedByProject((current) => {
      const translated = new Set(
        [...(current[projectId] ?? [])].map((path) =>
          translateProjectPath(path, moved.sourcePath, moved.destinationPath)),
      )
      const destinationParent = parentProjectPath(moved.destinationPath)
      if (destinationParent !== "") translated.add(destinationParent)
      return { ...current, [projectId]: translated }
    })
    setDrafts((current) => {
      const projectPrefix = `${projectId}\0`
      const next = { ...current }
      for (const [key, value] of Object.entries(current)) {
        if (!key.startsWith(projectPrefix)) continue
        const path = RelativePathSchema.make(key.slice(projectPrefix.length))
        const translated = translateProjectPath(path, moved.sourcePath, moved.destinationPath)
        if (translated === path) continue
        delete next[key]
        next[`${projectPrefix}${translated}`] = value
      }
      return next
    })
  }, [projectId, setDrafts, setExpandedByProject, setSelectedFile])
  const { result: moveResult, move } = useProjectEntryMove({ onSuccess: handleMoveSuccess })
  const height = useSyncExternalStore(subscribeWindow, windowHeight, windowHeight) - 60
  const viewportWidth = useSyncExternalStore(subscribeWindow, windowWidth, windowWidth)
  const sidebarCollapsed = useAtomValue(sidebarCollapsedAtom)
  const sidebarWidth = useAtomValue(sidebarWidthAtom)
  const occupiedWidth = WORKSPACE_CHAT_MINIMUM_WIDTH + (sidebarCollapsed ? 0 : sidebarWidth)
  const panelMode: WorkspacePanelWidthMode = selectedPath === null ? "filesTree" : "document"
  const maximumWidth = workspacePanelMaximumWidthForViewport(viewportWidth, occupiedWidth)
  const filesWidth = workspacePanelWidthForViewport(
    panelWidths.filesTree,
    WORKSPACE_FILES_MINIMUM_WIDTH,
    viewportWidth,
    occupiedWidth,
  )
  const panelWidth = panelMode === "filesTree"
    ? filesWidth
    : workspaceDocumentWidthForViewport(
        panelWidths.filesTree,
        panelWidths.document,
        viewportWidth,
        occupiedWidth,
      )
  const minimumWidth = panelMode === "filesTree" ? WORKSPACE_FILES_MINIMUM_WIDTH : filesWidth
  const fullWidth = viewportWidth <= WORKSPACE_PANEL_FULL_WIDTH_BREAKPOINT

  const root = directoryTree.root
  const rootListing = Result.value(root)
  const listings = Object.fromEntries(directoryTree.directories.flatMap(({ state }) => Option.match(
    Result.value(state),
    { onNone: () => [], onSome: (listing) => [[listing.directory, listing] as const] },
  )))
  const directoryLoadStates = Object.fromEntries(directoryTree.directories.map(({ directory, state }) => {
    const value = Result.value(state)
    const loadState: NonNullable<FileNode["loadState"]> = Result.isFailure(state)
      ? "failed"
      : Option.isSome(value) ? "loaded" : "loading"
    return [directory, loadState]
  }))
  const rootEntries = Option.match(rootListing, { onNone: () => [], onSome: (listing) => listing.entries })
  const nodes = useMemo(
    () => buildNodes(
      rootEntries,
      listings,
      directoryLoadStates,
      setSelectedPath,
      schedulePrefetch,
      cancelScheduledPrefetch,
      retryDirectory,
    ),
    [rootEntries, listings, directoryLoadStates, setSelectedPath, schedulePrefetch, cancelScheduledPrefetch, retryDirectory],
  )
  const initialOpenState = useMemo(
    () => Object.fromEntries([...expandedDirectories].map((path) => [path, true])),
    [expandedDirectories],
  )
  const nodesById = useMemo(() => {
    const index = new Map<string, FileNode>()
    const visit = (node: FileNode) => {
      index.set(node.id, node)
      node.children?.forEach(visit)
    }
    nodes.forEach(visit)
    return index
  }, [nodes])
  const renderDragPreview = useCallback((props: DragPreviewProps) => (
    <TreeDragPreview {...props} kind={props.id === null ? undefined : nodesById.get(props.id)?.kind} />
  ), [nodesById])
  const snapshot = Result.isSuccess(file) ? file.value : null
  const failure = Result.isFailure(saveResult) ? Cause.failureOption(saveResult.cause) : Option.none()
  const conflict = Option.isSome(failure) && failure.value._tag === "ProjectFileChanged" ? failure.value.current : null
  const draft = draftKey === null ? undefined : drafts[draftKey]
  const activeDraft = snapshot?._tag === "text" && draft?.content !== snapshot.content ? draft : undefined
  const displayedConflict = conflict ?? (
    snapshot?._tag === "text" && activeDraft !== undefined && activeDraft.baseContentHash !== snapshot.contentHash
      ? snapshot
      : null
  )
  const moveFailure = Result.isFailure(moveResult) ? Cause.failureOption(moveResult.cause) : Option.none()

  return (
    <aside
      ref={panelRef}
      aria-label="Project files"
      style={{ width: fullWidth ? "100%" : panelWidth }}
      onAnimationEnd={(event) => {
        if (event.currentTarget === event.target) setEntering(false)
      }}
      className={`relative flex shrink-0 flex-col overflow-hidden border-l border-slate-200 bg-white dark:border-slate-800 dark:bg-slate-850 ${entering && !resizing ? "animate-[workspace-panel-open_150ms_ease-out]" : ""} ${resizing || fullWidth ? "" : "transition-[width] duration-150 ease-out"}`}
    >
      {!fullWidth ? (
        <ResizableEdge
          side="left"
          value={panelWidth}
          minimum={minimumWidth}
          maximum={maximumWidth}
          onValueChange={(width) => setPanelWidths((current) => ({ ...current, [panelMode]: width }))}
          onDraggingChange={setResizing}
          label="Resize project files"
        />
      ) : null}
      <WorkspacePanelHeader
        surface="files"
        filesEnabled
        browserEnabled={platform.embeddedBrowser !== undefined}
        collapseTooltipDisabled={entering}
        onSurfaceChange={(next) => {
          if (next !== "browser") return
          setEntering(false)
          if (dirty) {
            setDiscardIntent("browser")
            return
          }
          setSurface("browser")
        }}
        onCollapse={handleCollapse}
      />
      {selectedPath !== null && (
        <div className="flex h-9 shrink-0 items-center gap-2 border-b border-slate-200 px-2 dark:border-slate-800">
          <ActionTooltip
            label="Back to project files"
            side="bottom"
            trigger={
              <Button variant="ghost" size="icon-sm" onClick={() => {
                if (dirty) {
                  setDiscardIntent("back")
                  return
                }
                setDirty(false)
                setSelectedPath(null)
              }} className="[-webkit-app-region:no-drag]" aria-label="Back to project files"><ArrowLeft size={16} /></Button>
            }
          />
          <div className="min-w-0 flex-1 truncate font-sans text-xs font-medium text-slate-700 dark:text-slate-300">{selectedPath}</div>
        {snapshot !== null && snapshot._tag !== "unsupported" && (
          <DropdownMenu>
            <DropdownMenuTrigger render={<Button variant="ghost" size="icon-sm" className="[-webkit-app-region:no-drag]" aria-label="File actions" />}><Ellipsis size={16} /></DropdownMenuTrigger>
            <DropdownMenuContent align="end" className="w-40">
              <DropdownMenuItem variant="destructive" onClick={() => setDeleteOpen(true)}><Trash2 />Remove file</DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        )}
        </div>
      )}
      {selectedPath === null ? (
        <div className="relative min-h-0 flex-1 px-1 py-2">
          {Option.isNone(rootListing)
            ? Result.isFailure(root)
              ? <div className="px-3 py-4 text-sm text-red-600 dark:text-red-400">Could not load project files.</div>
              : <div className="flex h-full min-h-48 flex-col items-center justify-center gap-3 text-[13px] font-medium text-slate-600 dark:text-slate-400">
                  <Spinner className="size-5 text-blue-600 motion-reduce:animate-none dark:text-blue-400" />
                  <span>Loading project files…</span>
                </div>
            : <Tree<FileNode>
                data={nodes as FileNode[]}
                aria-label="Project file tree"
                width="100%"
                height={Math.max(200, height)}
                rowHeight={28}
                openByDefault={false}
                initialOpenState={initialOpenState}
                indent={18}
                disableMultiSelection
                disableDrag={Result.isWaiting(moveResult)}
                disableDrop={({ parentNode, dragNodes }) => {
                  const source = dragNodes[0]?.data
                  if (source === undefined || dragNodes.length !== 1) return true
                  const destinationDirectory = parentNode.isRoot
                    ? rootPath
                    : parentNode.data.path
                  if (!parentNode.isRoot && parentNode.data.kind !== "directory") return true
                  return !canMoveProjectEntryToDirectory(source, destinationDirectory)
                }}
                disableEdit
                renderDragPreview={renderDragPreview}
                renderCursor={TreeDropCursor}
                onToggle={(id) => {
                  const node = nodesById.get(id)
                  if (node?.kind !== "directory") return
                  const next = new Set(expandedDirectories)
                  if (next.has(node.path)) next.delete(node.path)
                  else {
                    cancelScheduledPrefetch(node.path)
                    next.add(node.path)
                  }
                  setExpandedByProject((current) => ({ ...current, [projectId]: next }))
                }}
                onFocus={(node) => {
                  if (
                    node.data.kind === "directory"
                    && node.data.loadState === "idle"
                    && !expandedDirectories.has(node.data.path)
                  ) {
                    schedulePrefetch(node.data.path)
                  } else {
                    cancelScheduledPrefetch()
                  }
                }}
                onActivate={(node) => {
                  if (node.data.kind === "file") node.data.openFile(node.data.path)
                }}
                onMove={({ dragNodes, parentNode }) => {
                  const source = dragNodes[0]?.data
                  if (source === undefined) return
                  const destinationDirectory = parentNode === null || parentNode.isRoot
                    ? rootPath
                    : parentNode.data.path
                  return move({ projectId, sourcePath: source.path, destinationDirectory })
                }}
              >{TreeNode}</Tree>}
          {Option.isSome(moveFailure) && (
            <div role="alert" className="absolute bottom-3 left-3 right-3 rounded-md border border-red-300 bg-red-200/40 px-3 py-2 text-xs text-red-700 shadow-sm dark:border-red-700 dark:bg-red-800/30 dark:text-red-400">
              {moveFailure.value._tag === "ProjectFileAlreadyExists"
                ? "A file or folder with that name already exists there."
                : moveFailure.value._tag === "ProjectFileAccessDenied"
                  ? moveFailure.value.kind === "already_in_destination"
                    ? "The entry is already in that folder."
                    : moveFailure.value.kind === "self_move"
                      ? "A folder cannot be moved into itself."
                      : "This entry cannot be moved."
                  : "The file or folder could not be moved."}
            </div>
          )}
        </div>
      ) : snapshot === null ? (
        Result.isFailure(file) ? (
          <div className="px-4 py-6 text-sm text-red-600 dark:text-red-400">Could not open this file.</div>
        ) : (
          <div className="flex min-h-48 flex-1 flex-col items-center justify-center gap-3 text-[13px] font-medium text-slate-600 dark:text-slate-400">
            <Spinner className="size-5 text-blue-600 motion-reduce:animate-none dark:text-blue-400" />
            <span>Opening file…</span>
          </div>
        )
      ) : snapshot._tag === "image" ? (
        <div className="flex min-h-0 flex-1 items-center justify-center overflow-auto p-4"><img src={`data:${snapshot.mediaType};base64,${snapshot.data}`} alt={snapshot.path} className="max-h-full max-w-full object-contain" /></div>
      ) : snapshot._tag === "unsupported" ? (
        <div className="px-4 py-6 text-sm text-slate-500">This file cannot be displayed ({snapshot.reason.replaceAll("_", " ")}).</div>
      ) : (
        <Suspense fallback={(
          <div className="flex min-h-48 flex-1 flex-col items-center justify-center gap-3 text-[13px] font-medium text-slate-600 dark:text-slate-400">
            <Spinner className="size-5 text-blue-600 motion-reduce:animate-none dark:text-blue-400" />
            <span>Loading editor…</span>
          </div>
        )}>
          <ProjectTextEditor
            key={`${projectId}:${snapshot.path}`}
            projectId={projectId}
            snapshot={snapshot}
            saving={Result.isWaiting(saveResult)}
            conflict={displayedConflict}
            initialContent={activeDraft?.content ?? snapshot.content}
            onSave={(content) => save({
              projectId,
              path: snapshot.path,
              content,
              expectedContentHash:
                displayedConflict?.contentHash ?? activeDraft?.baseContentHash ?? snapshot.contentHash,
            })}
            onDraftChange={(content, isDirty) => {
              setDirty(isDirty)
              const key = `${projectId}\0${snapshot.path}`
              setDrafts((current) => {
                const existing = current[key]
                if (!isDirty) {
                  const { [key]: _removed, ...rest } = current
                  return rest
                }
                return {
                  ...current,
                  [key]: { content, baseContentHash: existing?.baseContentHash ?? snapshot.contentHash },
                }
              })
            }}
          />
        </Suspense>
      )}
      <AlertDialog open={deleteOpen} onOpenChange={setDeleteOpen}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Remove this file?</AlertDialogTitle>
            <AlertDialogDescription>
              {selectedPath === null ? "This file" : selectedPath} will be permanently deleted from the project.
            </AlertDialogDescription>
          </AlertDialogHeader>
          {Result.isFailure(deleteResult) && (
            <div className="text-xs text-red-600 dark:text-red-400">The file changed or could not be removed. Close this dialog and try again.</div>
          )}
          <AlertDialogFooter>
            <AlertDialogCancel disabled={Result.isWaiting(deleteResult)}>Cancel</AlertDialogCancel>
            <AlertDialogAction
              variant="destructive"
              disabled={snapshot === null || snapshot._tag === "unsupported" || Result.isWaiting(deleteResult)}
              onClick={(event) => {
                event.preventDefault()
                if (snapshot !== null && snapshot._tag !== "unsupported") {
                  remove({ projectId, path: snapshot.path, expectedContentHash: snapshot.contentHash })
                }
              }}
            >{Result.isWaiting(deleteResult) ? "Removing…" : "Remove"}</AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
      <AlertDialog open={discardIntent !== null} onOpenChange={(open) => {
        if (!open) setDiscardIntent(null)
      }}>
        <AlertDialogContent className="max-w-[400px]">
          <AlertDialogHeader>
            <AlertDialogTitle>Discard unsaved changes?</AlertDialogTitle>
            <AlertDialogDescription>
              Your edits to {selectedPath ?? "this file"} have not been saved.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Keep editing</AlertDialogCancel>
            <AlertDialogAction variant="destructive" onClick={() => {
              const intent = discardIntent
              if (draftKey !== null) {
                setDrafts((current) => {
                  const { [draftKey]: _removed, ...rest } = current
                  return rest
                })
              }
              setDiscardIntent(null)
              setDirty(false)
              if (intent === "back") setSelectedPath(null)
              if (intent === "close") collapsePanel()
              if (intent === "browser") setSurface("browser")
            }}>Discard changes</AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </aside>
  )
}
