import { useState } from "react"
import { Result, useAtomValue } from "@effect-atom/atom-react"
import { Option } from "effect"
import { ArrowUpIcon, FileIcon, FolderIcon } from "@phosphor-icons/react"
import { formatStorageSize, useAgentClient } from "@magnitudedev/client-common"
import type { DirectoryListing } from "@magnitudedev/sdk"
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from "./ui/dialog"
import { Button } from "./ui/button"
import { Input } from "./ui/input"
import { ErrorNotice } from "./error-notice"
import { SkeletonLine } from "./page-skeletons"

/** What a directory browser lists: folders only, or folders and their GGUF files. */
export type PickerKind = "folder" | "gguf"

/**
 * Chooses a folder on the machine running Magnitude, which may not be the machine showing this
 * window.
 */
export function FolderPicker({ open, onOpenChange, initialPath, title, description, confirmLabel, onChoose }: {
  open: boolean
  onOpenChange: (open: boolean) => void
  initialPath: Option.Option<string>
  title: string
  description: string
  confirmLabel: string
  onChoose: (path: string) => void
}) {
  return <Dialog open={open} onOpenChange={onOpenChange}>
    <DialogContent className="sm:max-w-lg">
      <DialogHeader>
        <DialogTitle>{title}</DialogTitle>
        <DialogDescription>{description}</DialogDescription>
      </DialogHeader>
      {open && <FolderBrowser initialPath={initialPath} confirmLabel={confirmLabel} onCancel={() => onOpenChange(false)} onChoose={path => { onOpenChange(false); onChoose(path) }} />}
    </DialogContent>
  </Dialog>
}

function FolderBrowser({ initialPath, confirmLabel, onCancel, onChoose }: {
  initialPath: Option.Option<string>
  confirmLabel: string
  onCancel: () => void
  onChoose: (path: string) => void
}) {
  const browser = useDirectoryBrowser("folder", initialPath)
  const current = Result.isSuccess(browser.listing) ? Option.some(browser.listing.value.path) : Option.none<string>()
  return <>
    <DirectoryBrowser browser={browser} selectedFile={Option.none()} onSelectFile={() => {}} />
    <DialogFooter>
      <Button variant="outline" onClick={onCancel}>Cancel</Button>
      <Button disabled={Option.isNone(current)} onClick={() => { if (Option.isSome(current)) onChoose(current.value) }}>{confirmLabel}</Button>
    </DialogFooter>
  </>
}

/** The folder being browsed on the machine running Magnitude, and in GGUF mode its GGUF files. */
export function useDirectoryBrowser(kind: PickerKind, initialPath: Option.Option<string>) {
  const client = useAgentClient()
  const [path, setPath] = useState(initialPath)
  const [typed, setTyped] = useState(Option.getOrElse(initialPath, () => ""))
  const listing = useAtomValue(client.Configuration.BrowseDirectories({ path, ggufFiles: kind === "gguf" })).result
  return { kind, listing, typed, setTyped, go: (next: string) => { setPath(Option.some(next)); setTyped(next) } }
}

/** A path field and one folder's entries; in GGUF mode clicking a file selects it. */
export function DirectoryBrowser({ browser, selectedFile, onSelectFile }: {
  browser: ReturnType<typeof useDirectoryBrowser>
  selectedFile: Option.Option<string>
  onSelectFile: (path: string) => void
}) {
  const { kind, listing, typed, setTyped, go } = browser
  return <>
    <form className="flex gap-2" onSubmit={event => { event.preventDefault(); if (typed.trim()) go(typed.trim()) }}>
      <Input aria-label="Folder path" value={typed} onChange={event => setTyped(event.target.value)} className="min-w-0 flex-1 font-mono text-xs" />
      <Button type="submit" variant="outline" size="sm">Go</Button>
    </form>
    <div className="max-h-72 min-h-40 overflow-y-auto rounded-md border border-slate-200 dark:border-slate-700" aria-label={kind === "gguf" ? "Folders and GGUF files" : "Folders"}>
      {Result.isFailure(listing) ? <ErrorNotice className="m-3" title="Couldn’t open this folder" description="Check that it exists and that Magnitude can read it." />
        : !Result.isSuccess(listing) ? <div className="space-y-2 p-3"><SkeletonLine className="h-5" width="60%" /><SkeletonLine className="h-5" width="45%" /></div>
        : <DirectoryEntries kind={kind} listing={listing.value} selectedFile={selectedFile} onOpen={go} onSelectFile={onSelectFile} />}
    </div>
  </>
}

const entryButton = "flex w-full items-center gap-2 px-3 py-2 text-left text-sm hover:bg-slate-100 focus-visible:outline-2 focus-visible:outline-blue-500 dark:hover:bg-slate-800"

/** One folder's contents: its parent, its folders, and in GGUF mode its GGUF files. */
export function DirectoryEntries({ kind, listing, selectedFile, onOpen, onSelectFile }: {
  kind: PickerKind
  listing: DirectoryListing
  selectedFile: Option.Option<string>
  onOpen: (path: string) => void
  onSelectFile: (path: string) => void
}) {
  const files = kind === "gguf" ? listing.files : []
  return <ul className="divide-y divide-slate-100 dark:divide-slate-800">
    {Option.isSome(listing.parent) && <li><button type="button" className={entryButton} onClick={() => onOpen(Option.getOrThrow(listing.parent))}><ArrowUpIcon aria-hidden="true" className="size-4 shrink-0 text-slate-500" />Parent folder</button></li>}
    {listing.directories.map(entry => <li key={entry.path}><button type="button" className={entryButton} onClick={() => onOpen(entry.path)}><FolderIcon aria-hidden="true" className="size-4 shrink-0 text-blue-600 dark:text-blue-400" /><span className="min-w-0 truncate">{entry.name}</span></button></li>)}
    {files.map(entry => {
      const selected = Option.contains(selectedFile, entry.path)
      return <li key={entry.path}><button type="button" aria-pressed={selected} className={`${entryButton} ${selected ? "bg-blue-50 dark:bg-slate-800" : ""}`} onClick={() => onSelectFile(entry.path)}>
        <FileIcon aria-hidden="true" className="size-4 shrink-0 text-slate-500" /><span className="min-w-0 flex-1 truncate">{entry.name}</span><span className="shrink-0 text-xs tabular-nums text-slate-500">{formatStorageSize(entry.sizeBytes)}</span>
      </button></li>
    })}
    {listing.directories.length === 0 && files.length === 0 && <li className="px-3 py-6 text-center text-sm text-slate-500">{kind === "gguf" ? "No folders or GGUF files inside" : "No folders inside"}</li>}
  </ul>
}
