import { useMemo, useState } from "react"
import { Atom, Result, useAtomSet, useAtomValue } from "@effect-atom/atom-react"
import { Effect, Option } from "effect"
import { ArrowUpIcon, FolderIcon } from "@phosphor-icons/react"
import { useAgentClient, type HostOpenPathRequest } from "@magnitudedev/client-common"
import { useSession } from "../session"
import type { DirectoryListing } from "@magnitudedev/sdk"
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from "./ui/dialog"
import { Button } from "./ui/button"
import { Input } from "./ui/input"
import { ErrorNotice } from "./error-notice"
import { SkeletonLine } from "./page-skeletons"


/** Shows the native open dialog and calls back with the path chosen; nothing happens when the user cancels. */
export type NativeOpenPath = (request: HostOpenPathRequest, onChosen: (path: string) => void) => void

/**
 * The desktop's native open dialog, present whenever the window has a native host on any OS; a
 * browser has none and browses the service's folders in-app instead.
 */
export function useNativeOpenPath(): Option.Option<NativeOpenPath> {
  const session = useSession()
  const run = useAtomSet(useMemo(() => Atom.fn(({ request, onChosen }: { request: HostOpenPathRequest; onChosen: (path: string) => void }) =>
    Option.match(session.openPathDialog, {
      onNone: () => Effect.void,
      onSome: open => open(request).pipe(Effect.flatMap(Option.match({ onNone: () => Effect.void, onSome: path => Effect.sync(() => onChosen(path)) }))),
    })), [session]))
  return Option.map(session.openPathDialog, () => (request, onChosen) => run({ request, onChosen }))
}

/** The action that chooses a path: the native dialog when the window has one, else `onBrowse` opens the in-app browser. */
export const choosePath = (native: Option.Option<NativeOpenPath>, request: HostOpenPathRequest, onChosen: (path: string) => void, onBrowse: () => void) => () =>
  Option.match(native, { onNone: onBrowse, onSome: open => open(request, onChosen) })

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
  const browser = useDirectoryBrowser(initialPath)
  const current = Result.isSuccess(browser.listing) ? Option.some(browser.listing.value.path) : Option.none<string>()
  return <>
    <DirectoryBrowser browser={browser} />
    <DialogFooter>
      <Button variant="outline" onClick={onCancel}>Cancel</Button>
      <Button disabled={Option.isNone(current)} onClick={() => { if (Option.isSome(current)) onChoose(current.value) }}>{confirmLabel}</Button>
    </DialogFooter>
  </>
}

/** The folder being browsed on the machine running Magnitude. */
export function useDirectoryBrowser(initialPath: Option.Option<string>) {
  const client = useAgentClient()
  const [path, setPath] = useState(initialPath)
  const [typed, setTyped] = useState(Option.getOrElse(initialPath, () => ""))
  const listing = useAtomValue(client.Configuration.BrowseDirectories({ path })).result
  return { listing, typed, setTyped, go: (next: string) => { setPath(Option.some(next)); setTyped(next) } }
}

/** A path field and one folder's subfolders. */
export function DirectoryBrowser({ browser }: { browser: ReturnType<typeof useDirectoryBrowser> }) {
  const { listing, typed, setTyped, go } = browser
  return <>
    <form className="flex gap-2" onSubmit={event => { event.preventDefault(); if (typed.trim()) go(typed.trim()) }}>
      <Input aria-label="Folder path" value={typed} onChange={event => setTyped(event.target.value)} className="min-w-0 flex-1 font-mono text-xs" />
      <Button type="submit" variant="outline" size="sm">Go</Button>
    </form>
    <div className="max-h-72 min-h-40 overflow-y-auto rounded-md border border-slate-200 dark:border-slate-700" aria-label="Folders">
      {Result.isFailure(listing) ? <ErrorNotice className="m-3" title="Couldn’t open this folder" description="Check that it exists and that Magnitude can read it." />
        : !Result.isSuccess(listing) ? <div className="space-y-2 p-3"><SkeletonLine className="h-5" width="60%" /><SkeletonLine className="h-5" width="45%" /></div>
        : <DirectoryEntries listing={listing.value} onOpen={go} />}
    </div>
  </>
}

const entryButton = "flex w-full items-center gap-2 px-3 py-2 text-left text-sm hover:bg-slate-100 focus-visible:outline-2 focus-visible:outline-blue-500 dark:hover:bg-slate-800"

/** One folder's contents: its parent and its subfolders. */
export function DirectoryEntries({ listing, onOpen }: { listing: DirectoryListing; onOpen: (path: string) => void }) {
  return <ul className="divide-y divide-slate-100 dark:divide-slate-800">
    {Option.isSome(listing.parent) && <li><button type="button" className={entryButton} onClick={() => onOpen(Option.getOrThrow(listing.parent))}><ArrowUpIcon aria-hidden="true" className="size-4 shrink-0 text-slate-500" />Parent folder</button></li>}
    {listing.directories.map(entry => <li key={entry.path}><button type="button" className={entryButton} onClick={() => onOpen(entry.path)}><FolderIcon aria-hidden="true" className="size-4 shrink-0 text-blue-600 dark:text-blue-400" /><span className="min-w-0 truncate">{entry.name}</span></button></li>)}
    {listing.directories.length === 0 && <li className="px-3 py-6 text-center text-sm text-slate-500">No folders inside</li>}
  </ul>
}
