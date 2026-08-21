import { lazy, Suspense, useCallback, useMemo, useRef, useState, useSyncExternalStore, type ReactNode } from "react"
import {
  Tree,
  type CursorProps,
  type DragPreviewProps,
  type NodeRendererProps,
} from "react-arborist"
import { CircleAlert, ChevronDown, ChevronRight, Ellipsis, File, FileText, Folder, FolderOpen, Save as SaveIcon, Trash2 } from "lucide-react"
import { Atom, Result, useAtomMount, useAtomSet, useAtomValue } from "@effect-atom/atom-react"
import { Cause, Effect, Option } from "effect"
import {
  useProjectDirectoryRefresh,
  useProjectDirectoryTree,
  useProjectEntryMove,
  useProjectFile,
  useProjectFileDelete,
  useProjectFileSave,
} from "@magnitudedev/client-common"
import { RelativePathSchema, type ProjectDirectoryEntry, type ProjectId, type RelativePath } from "@magnitudedev/sdk"
import { Button } from "@/components/ui/button"
import { Spinner } from "@/components/ui/spinner"
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
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import {
  expandedProjectDirectoriesAtom,
  projectFileDraftsAtom,
  workspacePresentationAtom,
} from "@/state/web-atoms"
import {
  canMoveProjectEntryToDirectory,
  parentProjectPath,
  translateProjectPath,
} from "@/lib/project-file-paths"
import { replaceWorkspaceFilePath, type WorkspaceFileTab } from "@/lib/workspace-tabs"

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
      className={`group flex cursor-default items-center gap-1 rounded px-1 text-[12px] ${node.willReceiveDrop ? "bg-blue-200 text-slate-900 ring-1 ring-inset ring-blue-400 dark:bg-blue-900 dark:text-slate-100 dark:ring-blue-600" : node.isSelected ? "bg-blue-100 text-slate-900 dark:bg-blue-900 dark:text-slate-100" : "text-slate-700 hover:bg-slate-150 dark:text-slate-300 dark:hover:bg-slate-800"}`}
      onClick={() => {
        if (node.isInternal) {
          if (node.data.loadState === "failed") {
            node.data.retryDirectory(node.data.path)
            if (!node.isOpen) node.open()
          } else node.toggle()
        } else {
          node.select()
          node.data.openFile(node.data.path)
        }
      }}
      onMouseEnter={() => {
        if (node.data.kind === "directory" && node.data.loadState === "idle") node.data.schedulePrefetch(node.data.path)
      }}
      onMouseLeave={() => {
        if (node.data.kind === "directory") node.data.cancelPrefetch(node.data.path)
      }}
    >
      <span className="flex size-3.5 shrink-0 items-center justify-center text-slate-500 dark:text-slate-400">
        {node.isInternal ? (
          node.isOpen && node.data.loadState === "loading"
            ? <Spinner className="size-3 text-blue-600 motion-reduce:animate-none dark:text-blue-400" />
            : node.data.loadState === "failed"
              ? <CircleAlert className="size-3 text-red-600 dark:text-red-400" aria-label="Could not load folder" />
              : node.isOpen ? <ChevronDown size={12} /> : <ChevronRight size={12} />
        ) : null}
      </span>
      <Icon size={13} className={node.data.kind === "directory" ? "shrink-0 text-blue-600 dark:text-blue-400" : "shrink-0 text-slate-500 dark:text-slate-400"} />
      <span className="min-w-0 truncate">{node.data.name}</span>
    </div>
  )
}

function TreeDragPreview({ offset, id, isDragging, kind }: DragPreviewProps & { readonly kind?: FileNode["kind"] }): ReactNode {
  if (!isDragging || offset === null || id === null) return null
  const name = id.slice(id.lastIndexOf("/") + 1)
  const Icon = kind === "directory" ? Folder : File
  return (
    <div className="pointer-events-none fixed inset-0 z-50">
      <div style={{ transform: `translate(${offset.x + 8}px, ${offset.y + 8}px)` }} className="absolute left-0 top-0 flex max-w-64 items-center gap-2 rounded-md border border-slate-300 bg-white px-2.5 py-1.5 text-[13px] text-slate-800 shadow-sm dark:border-slate-700 dark:bg-slate-800 dark:text-slate-100">
        <Icon size={14} className="shrink-0 text-blue-600 dark:text-blue-400" />
        <span className="truncate">{name}</span>
      </div>
    </div>
  )
}

function TreeDropCursor({ top, left, indent }: CursorProps): ReactNode {
  return <div aria-hidden="true" style={{ top: top - 1, left, right: indent }} className="pointer-events-none absolute z-10 h-0.5 rounded-full bg-blue-500 dark:bg-blue-400" />
}

export function ProjectTreeDock({
  projectId,
  selectedPath,
  onOpenFile,
}: {
  readonly projectId: ProjectId
  readonly selectedPath: RelativePath | null
  readonly onOpenFile: (path: RelativePath) => void
}): ReactNode {
  const [prefetchedDirectories, setPrefetchedDirectories] = useState<ReadonlySet<RelativePath>>(() => new Set())
  const refreshDirectory = useProjectDirectoryRefresh()
  const scheduledPrefetchRef = useRef<{ readonly path: RelativePath; readonly timer: number } | null>(null)
  const prefetchExpiryTimersRef = useRef(new Map<RelativePath, number>())
  const expandedByProject = useAtomValue(expandedProjectDirectoriesAtom)
  const setExpandedByProject = useAtomSet(expandedProjectDirectoriesAtom)
  const expandedDirectories = expandedByProject[projectId] ?? new Set<RelativePath>()
  const setWorkspace = useAtomSet(workspacePresentationAtom)

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
  const lifecycleAtom = useMemo(
    () => Atom.make(Effect.addFinalizer(() => Effect.sync(() => {
      cancelScheduledPrefetch()
      for (const timer of prefetchExpiryTimersRef.current.values()) window.clearTimeout(timer)
      prefetchExpiryTimersRef.current.clear()
    }))),
    [cancelScheduledPrefetch],
  )
  useAtomMount(lifecycleAtom)

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
  const { result: moveResult, move } = useProjectEntryMove({
    onSuccess: (moved) => {
      setWorkspace((current) => replaceWorkspaceFilePath(
        current,
        projectId,
        (path) => translateProjectPath(path, moved.sourcePath, moved.destinationPath),
      ))
      setExpandedByProject((current) => {
        const translated = new Set(
          [...(current[projectId] ?? [])].map((path) => translateProjectPath(path, moved.sourcePath, moved.destinationPath)),
        )
        const destinationParent = parentProjectPath(moved.destinationPath)
        if (destinationParent !== "") translated.add(destinationParent)
        return { ...current, [projectId]: translated }
      })
    },
  })
  const root = directoryTree.root
  const rootListing = Result.value(root)
  const listings = Object.fromEntries(directoryTree.directories.flatMap(({ state }) => Option.match(
    Result.value(state),
    { onNone: () => [], onSome: (listing) => [[listing.directory, listing] as const] },
  )))
  const loadStates = Object.fromEntries(directoryTree.directories.map(({ directory, state }) => {
    const value = Result.value(state)
    return [directory, Result.isFailure(state) ? "failed" : Option.isSome(value) ? "loaded" : "loading"] as const
  }))
  const rootEntries = Option.match(rootListing, { onNone: () => [], onSome: (listing) => listing.entries })
  const nodes = useMemo(
    () => buildNodes(rootEntries, listings, loadStates, onOpenFile, schedulePrefetch, cancelScheduledPrefetch, retryDirectory),
    [rootEntries, listings, loadStates, onOpenFile, schedulePrefetch, cancelScheduledPrefetch, retryDirectory],
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
  const moveFailure = Result.isFailure(moveResult) ? Cause.failureOption(moveResult.cause) : Option.none()
  const height = useSyncExternalStore(subscribeWindow, windowHeight, windowHeight) - 88

  return (
    <section aria-label="Project files" className="relative flex min-h-0 min-w-0 flex-1 flex-col bg-slate-50 dark:bg-slate-900">
      <div className="flex h-9 shrink-0 items-center border-b border-slate-200 px-2.5 font-sans text-xs font-medium text-slate-700 dark:border-slate-800 dark:text-slate-300">Project Files</div>
      <div className="relative min-h-0 flex-1 px-1 py-2">
        {Option.isNone(rootListing)
          ? Result.isFailure(root)
            ? <div className="px-2 py-4 text-xs text-red-600 dark:text-red-400">Could not load project files.</div>
            : <div className="flex h-full min-h-40 flex-col items-center justify-center gap-2 text-xs font-medium text-slate-600 dark:text-slate-400"><Spinner className="size-4 text-blue-600 motion-reduce:animate-none dark:text-blue-400" /><span>Loading project files…</span></div>
          : <Tree<FileNode>
              data={nodes as FileNode[]}
              aria-label="Project file tree"
              selection={selectedPath ?? undefined}
              width="100%"
              height={Math.max(160, height)}
              rowHeight={26}
              openByDefault={false}
              initialOpenState={Object.fromEntries([...expandedDirectories].map((path) => [path, true]))}
              indent={14}
              disableMultiSelection
              disableDrag={Result.isWaiting(moveResult)}
              disableDrop={({ parentNode, dragNodes }) => {
                const source = dragNodes[0]?.data
                if (source === undefined || dragNodes.length !== 1) return true
                const destination = parentNode.isRoot ? rootPath : parentNode.data.path
                if (!parentNode.isRoot && parentNode.data.kind !== "directory") return true
                return !canMoveProjectEntryToDirectory(source, destination)
              }}
              disableEdit
              renderDragPreview={(props) => <TreeDragPreview {...props} kind={props.id === null ? undefined : nodesById.get(props.id)?.kind} />}
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
                if (node.data.kind === "directory" && node.data.loadState === "idle" && !expandedDirectories.has(node.data.path)) schedulePrefetch(node.data.path)
                else cancelScheduledPrefetch()
              }}
              onActivate={(node) => {
                if (node.data.kind === "file") node.data.openFile(node.data.path)
              }}
              onMove={({ dragNodes, parentNode }) => {
                const source = dragNodes[0]?.data
                if (source === undefined) return
                const destinationDirectory = parentNode === null || parentNode.isRoot ? rootPath : parentNode.data.path
                return move({ projectId, sourcePath: source.path, destinationDirectory })
              }}
            >{TreeNode}</Tree>}
        {Option.isSome(moveFailure) ? (
          <div role="alert" className="absolute bottom-3 left-2 right-2 rounded-md border border-red-300 bg-red-200/40 px-2 py-2 text-xs text-red-700 shadow-sm dark:border-red-700 dark:bg-red-800/30 dark:text-red-400">
            {moveFailure.value._tag === "ProjectFileAlreadyExists" ? "An entry with that name already exists there." : "The file or folder could not be moved."}
          </div>
        ) : null}
      </div>
    </section>
  )
}

export function ProjectFileContent({
  tab,
  onDeleted,
}: {
  readonly tab: WorkspaceFileTab
  readonly onDeleted: () => void
}): ReactNode {
  const document = tab.document
  const [deleteOpen, setDeleteOpen] = useState(false)
  const drafts = useAtomValue(projectFileDraftsAtom)
  const setDrafts = useAtomSet(projectFileDraftsAtom)
  const selectedFile = useMemo(() => document === null ? null : ({ projectId: tab.projectId, path: document.path }), [document, tab.projectId])
  const file = useProjectFile(selectedFile)
  const draftKey = tab.id
  const handleSaveSuccess = useCallback(() => {
    setDrafts((current) => {
      const { [tab.id]: _removed, ...rest } = current
      return rest
    })
  }, [setDrafts, tab.id])
  const { result: saveResult, save } = useProjectFileSave({ onSuccess: handleSaveSuccess })
  const { result: deleteResult, remove } = useProjectFileDelete({ onSuccess: () => {
    setDrafts((current) => {
      const { [draftKey]: _removed, ...rest } = current
      return rest
    })
    setDeleteOpen(false)
    onDeleted()
  } })
  const snapshot = Result.isSuccess(file) ? file.value : null
  const saveFailure = Result.isFailure(saveResult) ? Cause.failureOption(saveResult.cause) : Option.none()
  const conflict = Option.isSome(saveFailure) && saveFailure.value._tag === "ProjectFileChanged" ? saveFailure.value.current : null
  const draft = drafts[draftKey]
  const activeDraft = snapshot?._tag === "text" && draft?.content !== snapshot.content ? draft : undefined
  const displayedConflict = conflict ?? (
    snapshot?._tag === "text" && activeDraft !== undefined && activeDraft.baseContentHash !== snapshot.contentHash
      ? snapshot : null
  )

  if (document === null) {
    return (
      <div className="flex min-h-0 flex-1 flex-col items-center justify-center gap-3 px-6 text-center">
        <FileText size={28} className="text-slate-400 dark:text-slate-500" />
        <div>
          <div className="font-mono text-base font-semibold text-slate-900 dark:text-slate-100">Select a file</div>
          <div className="mt-1 font-sans text-sm text-slate-500 dark:text-slate-400">Choose a file from Project Files.</div>
        </div>
      </div>
    )
  }

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div className="flex h-9 shrink-0 items-center gap-2 border-b border-slate-200 px-2 dark:border-slate-800">
        <div className="min-w-0 flex-1 truncate font-sans text-xs font-medium text-slate-700 dark:text-slate-300">{document.path}</div>
        {snapshot?._tag === "text" ? (
          <>
            {Result.isWaiting(saveResult) ? (
              <span className="font-sans text-xs text-slate-500 dark:text-slate-400">Saving…</span>
            ) : activeDraft !== undefined ? (
              <span className="font-sans text-xs text-slate-500 dark:text-slate-400">Unsaved</span>
            ) : null}
            <ActionTooltip label="Save" side="bottom" trigger={(
              <Button
                variant="ghost"
                size="icon-sm"
                className="[-webkit-app-region:no-drag]"
                aria-label="Save"
                disabled={activeDraft === undefined || Result.isWaiting(saveResult)}
                onClick={() => save({
                  projectId: tab.projectId,
                  path: snapshot.path,
                  content: activeDraft?.content ?? snapshot.content,
                  expectedContentHash: displayedConflict?.contentHash ?? activeDraft?.baseContentHash ?? snapshot.contentHash,
                })}
              ><SaveIcon size={16} /></Button>
            )} />
          </>
        ) : null}
        {snapshot !== null && snapshot._tag !== "unsupported" ? (
          <DropdownMenu>
            <DropdownMenuTrigger render={<Button variant="ghost" size="icon-sm" className="[-webkit-app-region:no-drag]" aria-label="File actions" />}><Ellipsis size={16} /></DropdownMenuTrigger>
            <DropdownMenuContent align="end" className="w-40">
              <DropdownMenuItem variant="destructive" onClick={() => setDeleteOpen(true)}><Trash2 />Remove file</DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        ) : null}
      </div>
      {snapshot === null ? (
        Result.isFailure(file)
          ? <div className="px-4 py-6 text-sm text-red-600 dark:text-red-400">Could not open this file.</div>
          : <div className="flex min-h-48 flex-1 flex-col items-center justify-center gap-3 text-[13px] font-medium text-slate-600 dark:text-slate-400"><Spinner className="size-5 text-blue-600 motion-reduce:animate-none dark:text-blue-400" /><span>Opening file…</span></div>
      ) : snapshot._tag === "image" ? (
        <div className="flex min-h-0 flex-1 items-center justify-center overflow-auto p-4"><img src={`data:${snapshot.mediaType};base64,${snapshot.data}`} alt={snapshot.path} className="max-h-full max-w-full object-contain" /></div>
      ) : snapshot._tag === "unsupported" ? (
        <div className="px-4 py-6 text-sm text-slate-500">This file cannot be displayed ({snapshot.reason.replaceAll("_", " ")}).</div>
      ) : (
        <Suspense fallback={<div className="flex min-h-48 flex-1 flex-col items-center justify-center gap-3 text-[13px] font-medium text-slate-600 dark:text-slate-400"><Spinner className="size-5 text-blue-600 motion-reduce:animate-none dark:text-blue-400" /><span>Loading editor…</span></div>}>
          <ProjectTextEditor
            key={`${tab.id}:${snapshot.path}`}
            projectId={tab.projectId}
            snapshot={snapshot}
            conflict={displayedConflict}
            initialContent={activeDraft?.content ?? snapshot.content}
            onSave={(content) => save({
              projectId: tab.projectId,
              path: snapshot.path,
              content,
              expectedContentHash: displayedConflict?.contentHash ?? activeDraft?.baseContentHash ?? snapshot.contentHash,
            })}
            onDraftChange={(content, dirty) => {
              setDrafts((current) => {
                const existing = current[tab.id]
                if (!dirty) {
                  const { [tab.id]: _removed, ...rest } = current
                  return rest
                }
                return { ...current, [tab.id]: { content, baseContentHash: existing?.baseContentHash ?? snapshot.contentHash } }
              })
            }}
          />
        </Suspense>
      )}
      <AlertDialog open={deleteOpen} onOpenChange={setDeleteOpen}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Remove this file?</AlertDialogTitle>
            <AlertDialogDescription>{document.path} will be permanently deleted from the project.</AlertDialogDescription>
          </AlertDialogHeader>
          {Result.isFailure(deleteResult) ? <div className="text-xs text-red-600 dark:text-red-400">The file changed or could not be removed. Close this dialog and try again.</div> : null}
          <AlertDialogFooter>
            <AlertDialogCancel disabled={Result.isWaiting(deleteResult)}>Cancel</AlertDialogCancel>
            <AlertDialogAction variant="destructive" disabled={snapshot === null || snapshot._tag === "unsupported" || Result.isWaiting(deleteResult)} onClick={(event) => {
              event.preventDefault()
              if (snapshot !== null && snapshot._tag !== "unsupported") remove({ projectId: tab.projectId, path: snapshot.path, expectedContentHash: snapshot.contentHash })
            }}>{Result.isWaiting(deleteResult) ? "Removing…" : "Remove"}</AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  )
}
