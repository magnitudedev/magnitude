import { Result, useAtomSet, useAtomValue } from "@effect-atom/atom-react"
import { ArrowCircleUpIcon, ArrowClockwiseIcon, CircleNotchIcon } from "@phosphor-icons/react"
import { useAgentClient, type ApplicationPage } from "@magnitudedev/client-common"
import type { ApplicationUpdateState, OwnerCapability } from "@magnitudedev/sdk"
import { Option } from "effect"
import { createContext, useContext, type ReactNode } from "react"
import { useDisconnectWarning } from "../remote-access"

export type UpdateTransfer = ApplicationUpdateState["transfer"]

/** The application that owns the service, and whether it supports a request. */
export function useApplicationOwner() {
  const client = useAgentClient()
  const owner = useAtomValue(client.Application.WatchApplicationOwner({})).result
  const supports = (capability: OwnerCapability) => Result.isSuccess(owner) && owner.value.capabilities.includes(capability)
  return { owner, supports }
}

/** The owner's update state; initial until the owner reports one. */
export function useApplicationUpdateObservation() {
  const { owner, supports } = useApplicationOwner()
  const observation = Result.flatMap(owner, value => Option.match(value.updates, { onNone: () => Result.initial(), onSome: updates => Result.success(updates) }))
  return { owner, supports, observation, snapshot: Result.isSuccess(observation) ? observation.value : null }
}

/** The update requests, their shared pending state, and Restart behind the remote-viewer disconnect warning. */
export function useApplicationUpdateActions() {
  const client = useAgentClient()
  const checkUpdate = useAtomSet(client.Application.CheckApplicationUpdate)
  const downloadUpdate = useAtomSet(client.Application.DownloadApplicationUpdate)
  const installUpdate = useAtomSet(client.Application.InstallApplicationUpdate)
  const discardUpdate = useAtomSet(client.Application.DiscardApplicationUpdate)
  const checking = useAtomValue(client.Application.CheckApplicationUpdate)
  const downloading = useAtomValue(client.Application.DownloadApplicationUpdate)
  const restarting = useAtomValue(client.Application.InstallApplicationUpdate)
  const discarding = useAtomValue(client.Application.DiscardApplicationUpdate)
  const disconnect = useDisconnectWarning()
  return {
    check: () => checkUpdate({}),
    download: () => downloadUpdate({}),
    restart: () => disconnect.warn("Restart", () => installUpdate({})),
    discard: () => discardUpdate({}),
    checking, downloading, restarting, discarding,
    pending: downloading.waiting || restarting.waiting || discarding.waiting,
    dialog: disconnect.dialog,
  }
}

/** What the sidebar offers for an update: only states a person can act on, or a download already under way. */
export type SidebarUpdate =
  | { readonly _tag: "None" }
  | { readonly _tag: "Available"; readonly version: string; readonly previouslyFailed: boolean }
  | { readonly _tag: "Downloading"; readonly version: string; readonly fraction: Option.Option<number> }
  | { readonly _tag: "Preparing"; readonly version: string }
  | { readonly _tag: "Ready"; readonly version: string; readonly retry: boolean }
  | { readonly _tag: "ImageAvailable"; readonly version: string }

export const sidebarUpdate = (transfer: UpdateTransfer | undefined): SidebarUpdate => {
  switch (transfer?._tag) {
    case "Available": return { _tag: "Available", version: transfer.version, previouslyFailed: Option.isSome(transfer.failed) }
    case "Downloading": return { _tag: "Downloading", version: transfer.version, fraction: transfer.total > 0 ? Option.some(Math.min(1, transfer.completed / transfer.total)) : Option.none() }
    case "Staging": return { _tag: "Preparing", version: transfer.version }
    case "Ready": return { _tag: "Ready", version: transfer.version, retry: false }
    case "InstallationFailed": return { _tag: "Ready", version: transfer.version, retry: true }
    case "ImageAvailable": return { _tag: "ImageAvailable", version: transfer.version }
    default: return { _tag: "None" }
  }
}

interface SidebarUpdateState {
  readonly update: SidebarUpdate
  readonly busy: boolean
  readonly download: () => void
  readonly restart: () => void
}

/** The sidebar also frames the boot skeleton, before any client exists, where no update is shown. */
const SidebarUpdateContext = createContext<SidebarUpdateState>({ update: { _tag: "None" }, busy: false, download: () => {}, restart: () => {} })
export const useSidebarUpdate = () => useContext(SidebarUpdateContext)

/** Observes the owner's updates once for the sidebar's row and its collapsed toggle. */
export function SidebarUpdateProvider({ children }: { children: ReactNode }) {
  const { supports, snapshot } = useApplicationUpdateObservation()
  const actions = useApplicationUpdateActions()
  const update = supports("Updates") ? sidebarUpdate(snapshot?.transfer) : { _tag: "None" as const }
  const busy = actions.pending || actions.checking.waiting || snapshot?.check._tag === "Checking"
  return <SidebarUpdateContext.Provider value={{ update, busy, download: actions.download, restart: actions.restart }}>
    {actions.dialog}{children}
  </SidebarUpdateContext.Provider>
}

const pillBase = "relative flex w-full items-center gap-2.5 overflow-hidden rounded-lg border px-3 py-2 text-left focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-blue-500"
const pillAction = `${pillBase} border-blue-300 text-blue-700 hover:bg-blue-50 disabled:opacity-60 dark:border-blue-400/30 dark:text-blue-400 dark:hover:bg-blue-400/10`
const iconClass = "size-[18px] shrink-0"

/** A title, then the version and what a click does, so the row says which release and how to get it. */
function PillText({ title, detail, attention = false }: { title: string; detail: string; attention?: boolean }) {
  return <>
    <span className="flex min-w-0 flex-1 flex-col">
      <span className="truncate text-sm font-medium leading-5">{title}</span>
      <span className="truncate text-xs leading-4 text-slate-500 dark:text-slate-400">{detail}</span>
    </span>
    {attention && <span aria-label="Needs attention" className="size-2 shrink-0 rounded-full bg-orange-400" />}
  </>
}

/** Presentation only: the sidebar row for an update, given what clicking it does. */
export function UpdatePill({ update, busy, onDownload, onRestart, onOpenSettings }: {
  update: SidebarUpdate; busy: boolean; onDownload: () => void; onRestart: () => void; onOpenSettings: () => void
}) {
  switch (update._tag) {
    case "None": return null
    case "Available": return <button type="button" data-update-pill="Available" disabled={busy} className={pillAction}
      title={update.previouslyFailed ? `Version ${update.version} didn’t install last time. Open Settings for details.` : `Download version ${update.version}`}
      onClick={update.previouslyFailed ? onOpenSettings : onDownload}>
      <ArrowCircleUpIcon weight="fill" aria-hidden="true" className={iconClass} />
      <PillText title="Update available" detail={update.previouslyFailed ? `${update.version} · See Settings` : `Click to download ${update.version}`} attention={update.previouslyFailed} />
    </button>
    case "ImageAvailable": return <button type="button" data-update-pill="ImageAvailable" className={pillAction} title={`Version ${update.version} is available as a new image`} onClick={onOpenSettings}>
      <ArrowCircleUpIcon weight="fill" aria-hidden="true" className={iconClass} />
      <PillText title="Update available" detail={`${update.version} · See Settings`} />
    </button>
    case "Downloading":
    case "Preparing": {
      const percent = update._tag === "Downloading" ? Option.map(update.fraction, fraction => Math.round(fraction * 100)) : Option.none<number>()
      const detail = `${update.version}${Option.match(percent, { onNone: () => "", onSome: value => ` · ${value}%` })}`
      return <div role="status" data-update-pill={update._tag} title={`${update._tag === "Preparing" ? "Preparing" : "Downloading"} version ${update.version}`}
        className={`${pillBase} border-blue-200 text-blue-700 dark:border-blue-400/20 dark:text-blue-400`}>
        <CircleNotchIcon aria-hidden="true" className={`${iconClass} motion-safe:animate-spin`} />
        <PillText title={update._tag === "Preparing" ? "Preparing update" : "Downloading"} detail={detail} />
        {Option.isSome(percent) && <span aria-hidden="true" className="absolute inset-x-0 bottom-0 h-0.5 bg-blue-100 dark:bg-slate-700">
          <span className="block h-full bg-blue-500 transition-[width] duration-300 motion-reduce:transition-none" style={{ width: `${percent.value}%` }} />
        </span>}
      </div>
    }
    case "Ready": return <button type="button" data-update-pill="Ready" disabled={busy} className={pillAction}
      title={update.retry ? `Version ${update.version} didn’t finish installing. Click to restart and try again.` : `Restart Magnitude to install version ${update.version}. Restarting stops the running model and service.`}
      onClick={onRestart}>
      <ArrowClockwiseIcon weight="bold" aria-hidden="true" className={iconClass} />
      <PillText title="Click to restart" detail={update.retry ? `${update.version} didn’t install` : `Update ${update.version} ready`} attention={update.retry} />
    </button>
  }
}

/** The sidebar's update row; absent while the application is up to date or can't update itself. */
export function SidebarUpdatePill({ navigate, onNavigate }: { navigate?: (page: ApplicationPage) => void; onNavigate?: () => void }) {
  const { update, busy, download, restart } = useSidebarUpdate()
  if (update._tag === "None") return null
  return <div className="mb-2">
    <UpdatePill update={update} busy={busy} onDownload={download} onRestart={restart}
      onOpenSettings={() => { navigate?.("settings"); onNavigate?.() }} />
  </div>
}
