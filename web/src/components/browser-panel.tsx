import {
  useMemo,
  useRef,
  useState,
  type FormEvent,
  type ReactNode,
} from "react"
import { ArrowLeft, ArrowRight, Download, ExternalLink, FolderOpen, RotateCw, X } from "lucide-react"
import { Atom, useAtomMount } from "@effect-atom/atom-react"
import { Effect } from "effect"
import {
  formatStorageSize,
  type BrowserDownloadState,
  type BrowserTabState,
  type BrowserWorkspaceState,
  type EmbeddedBrowserCapability,
} from "@magnitudedev/client-common"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
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
import { notify } from "@/lib/notifications"

const runCommand = (command: Promise<unknown>, failure: string): void => {
  void command.catch((cause: unknown) => {
    console.error(`[browser] ${failure}`, cause)
    notify("error", failure)
  })
}

function BrowserDownload({
  download,
  browser,
}: {
  readonly download: BrowserDownloadState
  readonly browser: EmbeddedBrowserCapability
}): ReactNode {
  const total = download.totalBytes
  const progress = total > 0 ? Math.min(100, Math.round(download.receivedBytes / total * 100)) : null
  return (
    <div className="flex h-9 shrink-0 items-center gap-2 border-b border-slate-200 px-3 font-sans text-xs dark:border-slate-800">
      {download.status === "progressing"
        ? <Spinner className="size-3.5 text-blue-600 motion-reduce:animate-none dark:text-blue-400" />
        : <Download className="size-3.5 text-slate-500 dark:text-slate-400" />}
      <span className="min-w-0 flex-1 truncate text-slate-700 dark:text-slate-300">{download.fileName}</span>
      <span className="shrink-0 tabular-nums text-slate-500 dark:text-slate-400">
        {download.status === "progressing"
          ? progress === null ? formatStorageSize(download.receivedBytes) : `${progress}%`
          : download.status === "completed" ? "Downloaded"
            : download.status === "cancelled" ? "Cancelled" : "Failed"}
      </span>
      {download.status === "progressing" ? (
        <ActionTooltip label="Cancel download" side="bottom" trigger={(
          <Button variant="ghost" size="icon-xs" aria-label="Cancel download" onClick={() => runCommand(browser.cancelDownload(download.id), "Could not cancel download.")}>
            <X size={13} />
          </Button>
        )} />
      ) : download.status === "completed" ? (
        <ActionTooltip label="Reveal download" side="bottom" trigger={(
          <Button variant="ghost" size="icon-xs" aria-label="Reveal download" onClick={() => runCommand(browser.revealDownload(download.id), "Could not reveal download.")}>
            <FolderOpen size={13} />
          </Button>
        )} />
      ) : null}
    </div>
  )
}

export function BrowserContent({
  browser,
  state,
  activeTab,
}: {
  readonly browser: EmbeddedBrowserCapability
  readonly state: BrowserWorkspaceState
  readonly activeTab: BrowserTabState
}): ReactNode {
  const viewportRef = useRef<HTMLDivElement>(null)
  const addressRef = useRef<HTMLInputElement>(null)
  const [addressDrafts, setAddressDrafts] = useState<Readonly<Record<string, string>>>({})
  const addressValue = addressDrafts[activeTab.id] ?? activeTab.pendingUrl ?? activeTab.url
  const activeDownloads = state.downloads.filter((download) => download.status === "progressing")
  const latestCompletedDownload = [...state.downloads].reverse().find(
    (download) => download.status !== "progressing",
  )

  const focusLocationAtom = useMemo(
    () => Atom.make(Effect.sync(() => {
      if (state.focusLocationRevision === 0) return
      addressRef.current?.focus()
      addressRef.current?.select()
    })),
    [state.focusLocationRevision],
  )
  useAtomMount(focusLocationAtom)

  const viewportLifecycleAtom = useMemo(
    () => Atom.make(Effect.gen(function* () {
      if (state.permissionRequest !== null) {
        yield* Effect.promise(() => browser.setViewport(null))
        return
      }
      const element = viewportRef.current
      if (element === null) return
      let frame = 0
      const measure = () => {
        frame = 0
        const bounds = element.getBoundingClientRect()
        runCommand(browser.setViewport({
          x: Math.round(bounds.x),
          y: Math.round(bounds.y),
          width: Math.round(bounds.width),
          height: Math.round(bounds.height),
        }), "Could not position the browser.")
      }
      const schedule = () => {
        if (frame !== 0) cancelAnimationFrame(frame)
        frame = requestAnimationFrame(measure)
      }
      const observer = new ResizeObserver(schedule)
      observer.observe(element)
      window.addEventListener("resize", schedule)
      schedule()
      yield* Effect.addFinalizer(() => Effect.sync(() => {
        if (frame !== 0) cancelAnimationFrame(frame)
        observer.disconnect()
        window.removeEventListener("resize", schedule)
        void browser.setViewport(null)
      }))
    })),
    [activeTab.id, browser, state.permissionRequest],
  )
  useAtomMount(viewportLifecycleAtom)

  const navigate = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    const input = addressDrafts[activeTab.id] ?? addressValue
    setAddressDrafts((current) => {
      const { [activeTab.id]: _removed, ...rest } = current
      return rest
    })
    runCommand(browser.navigate(input), "Could not open this address.")
    addressRef.current?.blur()
  }

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <form onSubmit={navigate} className="relative flex h-11 shrink-0 items-center gap-1.5 border-b border-slate-200 px-2 dark:border-slate-800">
        <Button variant="ghost" size="icon-sm" type="button" disabled={!activeTab.canGoBack} aria-label="Back" onClick={() => runCommand(browser.goBack(), "Could not go back.")}><ArrowLeft size={16} /></Button>
        <Button variant="ghost" size="icon-sm" type="button" disabled={!activeTab.canGoForward} aria-label="Forward" onClick={() => runCommand(browser.goForward(), "Could not go forward.")}><ArrowRight size={16} /></Button>
        <Button variant="ghost" size="icon-sm" type="button" disabled={activeTab.phase === "blank"} aria-label={activeTab.phase === "loading" ? "Stop" : "Reload"} onClick={() => runCommand(activeTab.phase === "loading" ? browser.stop() : browser.reload(), activeTab.phase === "loading" ? "Could not stop loading." : "Could not reload this page.")}>
          {activeTab.phase === "loading" ? <X size={15} /> : <RotateCw size={15} />}
        </Button>
        <Input
          ref={addressRef}
          value={addressValue}
          aria-label="Address and search"
          autoCapitalize="off"
          autoCorrect="off"
          spellCheck={false}
          placeholder="Search or enter an address"
          className="h-8 flex-1 rounded-full bg-slate-100 px-3 dark:bg-slate-900"
          onFocus={(event) => event.currentTarget.select()}
          onChange={(event) => setAddressDrafts((current) => ({ ...current, [activeTab.id]: event.target.value }))}
          onKeyDown={(event) => {
            if (event.key !== "Escape") return
            setAddressDrafts((current) => {
              const { [activeTab.id]: _removed, ...rest } = current
              return rest
            })
            event.currentTarget.blur()
          }}
        />
        <ActionTooltip label="Open in default browser" side="top" trigger={(
          <Button variant="ghost" size="icon-sm" type="button" disabled={activeTab.url.length === 0} aria-label="Open in default browser" onClick={() => runCommand(browser.openExternal(), "Could not open the default browser.")}><ExternalLink size={15} /></Button>
        )} />
        {activeTab.phase === "loading" ? <div className="absolute inset-x-0 bottom-0 h-0.5 overflow-hidden bg-blue-200 dark:bg-blue-900"><div className="h-full w-1/3 animate-[browser-loading_1.2s_ease-in-out_infinite] bg-blue-600 motion-reduce:w-full motion-reduce:animate-none dark:bg-blue-400" /></div> : null}
      </form>
      {activeDownloads.map((download) => <BrowserDownload key={download.id} download={download} browser={browser} />)}
      {activeDownloads.length === 0 && latestCompletedDownload !== undefined
        ? <BrowserDownload download={latestCompletedDownload} browser={browser} /> : null}
      <div ref={viewportRef} data-browser-viewport className="relative min-h-0 flex-1 bg-white dark:bg-slate-900">
        {activeTab.phase === "blank" ? (
          <div className="absolute inset-0 flex flex-col items-center justify-center px-8 text-center">
            <div className="font-mono text-lg font-semibold text-slate-900 dark:text-slate-100">Browse the web</div>
            <div className="mt-1 font-sans text-sm text-slate-500 dark:text-slate-400">Search or enter an address above.</div>
          </div>
        ) : activeTab.phase === "failed" || activeTab.phase === "crashed" ? (
          <div className="absolute inset-0 flex flex-col items-center justify-center gap-4 px-8 text-center">
            <div>
              <div className="font-mono text-lg font-semibold text-slate-900 dark:text-slate-100">{activeTab.phase === "crashed" ? "This page crashed" : "Couldn’t open this page"}</div>
              <div className="mt-1 max-w-md font-sans text-sm text-slate-500 dark:text-slate-400">{activeTab.error}</div>
            </div>
            <div className="flex gap-2">
              <Button onClick={() => runCommand(browser.reload(), "Could not reload this page.")}>Retry</Button>
              {activeTab.url.length > 0 ? <Button variant="outline" onClick={() => runCommand(browser.openExternal(), "Could not open the default browser.")}>Open in default browser</Button> : null}
            </div>
          </div>
        ) : activeTab.phase === "insecure" ? (
          <div className="absolute inset-0 flex flex-col items-center justify-center gap-4 px-8 text-center">
            <div>
              <div className="font-mono text-lg font-semibold text-slate-900 dark:text-slate-100">Insecure connection</div>
              <div className="mt-1 max-w-md font-sans text-sm text-slate-500 dark:text-slate-400">This site uses unencrypted HTTP. Continue only if you trust it.</div>
              <div className="mt-2 truncate font-sans text-xs text-slate-500 dark:text-slate-400">{activeTab.insecureUrl}</div>
            </div>
            <div className="flex gap-2">
              <Button onClick={() => runCommand(browser.continueInsecureNavigation(), "Could not open this address.")}>Continue once</Button>
              <Button variant="outline" onClick={() => runCommand(browser.cancelInsecureNavigation(), "Could not cancel this navigation.")}>Cancel</Button>
            </div>
          </div>
        ) : null}
      </div>
      <AlertDialog open={state.permissionRequest !== null}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Allow site permission?</AlertDialogTitle>
            <AlertDialogDescription>{state.permissionRequest?.origin} wants permission to use {state.permissionRequest?.permission.replaceAll("-", " ")}.</AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel onClick={() => {
              const request = state.permissionRequest
              if (request !== null) runCommand(browser.respondToPermission(request.id, false), "Could not answer the permission request.")
            }}>Deny</AlertDialogCancel>
            <AlertDialogAction onClick={(event) => {
              event.preventDefault()
              const request = state.permissionRequest
              if (request !== null) runCommand(browser.respondToPermission(request.id, true), "Could not answer the permission request.")
            }}>Allow once</AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  )
}
