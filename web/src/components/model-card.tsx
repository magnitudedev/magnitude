import { useState, type ReactElement, type ReactNode } from "react"
import { Option } from "effect"
import { Menu } from "@base-ui/react/menu"
import { ArrowClockwiseIcon, ArrowUpRightIcon, CaretDownIcon, DotsThreeVerticalIcon, DownloadSimpleIcon, EyeSlashIcon, LightningIcon, MagnifyingGlassIcon, PlayIcon, PlusIcon, SquareIcon, TrashIcon, XIcon } from "@phosphor-icons/react"
import {
  catalogComponentName, catalogComponentOffers, catalogModelReplacement, customModelDrafterOffer,
  formatLocalModelDisplayName, formatMemorySize, formatStorageSize, localModelFacts,
  localModelIsInstalled, localModelMemoryUse, MODEL_STOPPED_FOR_MEMORY_MESSAGE, modelStoppedForMemory, PENDING_DRAFTER_LABEL, removalKind,
  useLocalModelCommandStatus, useLocalModelMutations, useLocalModelStopStatus,
} from "@magnitudedev/client-common"
import { customModelSourceLabel, localModelDeprecation, type CatalogLocalModel, type LocalInferenceHardware, type LocalModel } from "@magnitudedev/sdk"
import { Button } from "./ui/button"
import { ActionTooltip, TooltipProvider } from "./ui/tooltip"
import { ConfirmDialog } from "./confirm-dialog"
import { ErrorNotice, NoticeAction } from "./error-notice"
import { HuggingFaceLogo, ModelLogo } from "./model-logo"
import { SpeedInfo } from "./discovery-visuals"
import { DrafterPickerDialog } from "./drafter-list"
import { customModelSource } from "./custom-model-details"
import { removalConfirmation } from "./model-remove-action"
import { fitNotice, loadRefusal } from "./model-fit"
import { downloadNotice, modelCommandNotice, modelRemovalNotice, ModelLoadNotice } from "./model-error"
import { acquiring } from "./models-page"
import { addedModelAwaitingFiles, catalogDrafterOnlyUpdate, modelCanDownload, modelCardDrafter, modelCardStatus, modelSubtitle, modelTransferEstimate, modelUpdateDescription, type StatusTone } from "./model-card-state"

interface PrimaryAction {
  readonly label: ReactNode
  readonly icon: ReactNode
  readonly disabled: boolean
  readonly outline?: boolean
  readonly title?: string
  readonly tooltip?: string
  readonly onClick: () => void
}

interface MenuAction {
  readonly label: string
  readonly icon: ReactNode
  readonly disabled: boolean
  readonly destructive?: boolean
  readonly onClick: () => void
}

interface Transfer {
  readonly label: string
  readonly estimate: string | null
  readonly percent: number | null
  readonly cancelLabel: string
  readonly pending: boolean
  readonly onCancel: () => void
}

const transferStages = { queued: "Queued", resolving: "Preparing", checking_space: "Checking space", downloading: "Downloading", verifying: "Verifying", publishing: "Finishing" }

export interface ModelCardProps {
  readonly model: LocalModel
  readonly models: readonly LocalModel[]
  readonly page: "catalog" | "models"
  readonly replacing?: string
  readonly hardware: Option.Option<LocalInferenceHardware>
  readonly onFindOtherVersions?: () => void
}

/**
 * A Catalog or My Models card, complete without expanding. The header holds the model, its quant and (for an added
 * or found model) a source badge, its main action (a download fills that button) and a "⋮" menu for the rest; the
 * strip below states status, memory, drafter, estimated speed and intelligence, each acting where it is shown.
 */
export function ModelCard({ model, models, page, replacing, hardware, onFindOtherVersions }: ModelCardProps) {
  const { install, load, stop, cancel, remove, hide, dismissFailure, downloadComponent, setDrafter } = useLocalModelMutations()
  const command = useLocalModelCommandStatus(model.modelId)
  const stopping = useLocalModelStopStatus()
  const [confirmingLoad, setConfirmingLoad] = useState(false)
  const [confirmingRemove, setConfirmingRemove] = useState(false)
  const [choosingDrafter, setChoosingDrafter] = useState(false)
  const deprecation = localModelDeprecation(model)
  const replacement = Option.flatMap(deprecation, value => catalogModelReplacement(models, value))
  const replacementCommand = useLocalModelCommandStatus(Option.match(replacement, { onNone: () => model.modelId, onSome: target => target.modelId }))

  const acquisition = model.acquisitionState
  const pending = command.pending || stopping.pending || acquisition._tag === "Removing"
  const supported = model._tag === "Custom" || model.catalogData.support._tag === "Supported"
  const installed = "residencyState" in acquisition
  const residency = installed ? acquisition.residencyState : undefined
  const canStop = residency !== undefined && ["Ready", "Loading", "Requested", "Stopping"].includes(residency._tag)
  const transferring = acquiring(acquisition)
  const refusal = loadRefusal(model)
  const loadFailure = residency?._tag === "Failed" && !command.pendingOperations.includes("load") ? residency.failure : null
  const loadNotice = refusal === null ? loadFailure : null
  const downloadFailure = (acquisition._tag === "InstallFailed" || acquisition._tag === "UpdateFailed") && !command.pendingOperations.includes("install") ? acquisition.failure : null
  const awaitingFiles = addedModelAwaitingFiles(model)
  const canDownload = modelCanDownload(model)
  const fit = supported && !awaitingFiles ? fitNotice(model) : null
  const modelUpdate = Option.isNone(deprecation) && !transferring && installed && supported && acquisition._tag === "UpdateAvailable" && !catalogDrafterOnlyUpdate(model)
  const requestLoad = () => { if (replacing) setConfirmingLoad(true); else load(model.modelId) }
  const name = formatLocalModelDisplayName(model)

  const transfer: Transfer | null = (() => {
    if (acquisition._tag === "Optimizing") {
      const { stage, completed, total } = acquisition.progress
      const percent = stage === "tuning" && total > 0 ? Math.min(100, completed * 100 / total) : null
      return { label: percent === null ? "Optimizing" : `Optimizing ${Math.floor(percent)}%`, estimate: null, percent, cancelLabel: "Skip optimization", pending: command.pending, onCancel: () => cancel(model.modelId) }
    }
    if (acquisition._tag !== "Installing" && acquisition._tag !== "Updating") return null
    const { progress } = acquisition
    const percent = progress.totalBytes > 0 ? progress.completedBytes / progress.totalBytes * 100 : null
    const label = progress.stage === "downloading" && percent !== null ? `${Math.floor(percent)}%` : transferStages[progress.stage]
    return { label, estimate: modelTransferEstimate(progress), percent, cancelLabel: "Cancel download", pending: command.pending, onCancel: () => cancel(model.modelId) }
  })()

  const primary: PrimaryAction | null = Option.isSome(deprecation)
    ? Option.match(replacement, {
      onNone: () => null,
      onSome: (target: CatalogLocalModel) => {
        const fits = target.servingState._tag === "Assessed" && target.servingState.assessment._tag === "Fits"
        return { label: "Switch", title: `Switch to ${formatLocalModelDisplayName(target)}`, icon: null, disabled: replacementCommand.pending || acquiring(target.acquisitionState) || !fits, onClick: () => localModelIsInstalled(target) ? load(target.modelId) : install(target.modelId) }
      },
    })
    : transferring || acquisition._tag === "Unavailable" || downloadFailure ? null
    : !installed ? { label: <>Download{model.storageBytes > 0 && <> <span className="tabular-nums opacity-75">{formatStorageSize(model.storageBytes)}</span></>}</>, icon: <DownloadSimpleIcon />, disabled: pending || !canDownload, onClick: () => install(model.modelId) }
    : !supported ? null
    : canStop ? { label: "Stop", icon: <SquareIcon />, disabled: stopping.pending, outline: true, onClick: () => stop() }
    : modelUpdate ? { label: "Update", ...Option.match(modelUpdateDescription(model), { onNone: () => ({}), onSome: tooltip => ({ tooltip }) }), icon: <ArrowClockwiseIcon />, disabled: pending, onClick: () => install(model.modelId) }
    : loadNotice ? null
    : { label: "Load", icon: <PlayIcon />, disabled: pending || refusal !== null, ...(refusal ? { title: refusal } : {}), onClick: requestLoad }

  const kind = removalKind(model)
  const removable = !transferring && (installed || acquisition._tag === "Unavailable" || awaitingFiles) && !(Option.isSome(deprecation) && !localModelIsInstalled(model))
  const menu: MenuAction[] = [
    ...(modelUpdate && !canStop && !loadNotice ? [{ label: "Load this version", icon: <PlayIcon />, disabled: pending || refusal !== null, onClick: requestLoad }] : []),
    ...(model._tag === "Catalog" ? catalogComponentOffers(model).filter(({ component }) => component !== "Drafter").map(offer => ({ label: `Add ${catalogComponentName(offer.component).toLowerCase()}${offer.downloadBytes > 0 ? ` · ${formatStorageSize(offer.downloadBytes)}` : ""}`, icon: <DownloadSimpleIcon />, disabled: command.pending, onClick: () => downloadComponent(model.modelId, offer.component) })) : []),
    ...(onFindOtherVersions ? [{ label: "Find other versions", icon: <MagnifyingGlassIcon />, disabled: false, onClick: onFindOtherVersions }] : []),
    ...(removable && (kind === "hide" || kind === "parts") ? [{ label: "Hide from Magnitude", icon: <EyeSlashIcon />, disabled: pending, onClick: () => hide(model.modelId) }] : []),
    ...(removable && kind !== "hide" ? [{ label: kind === "parts" ? "Remove downloaded files" : "Remove download", icon: <TrashIcon />, disabled: pending, destructive: true, onClick: () => setConfirmingRemove(true) }] : []),
  ]

  const status = modelCardStatus(model, hardware)

  const drafterState = modelCardDrafter(model)
  const drafter: ReactNode = (() => {
    switch (drafterState._tag) {
      case "Add": return <AddDrafterButton disabled={pending} onAdd={() => {
        if (catalogDrafterOnlyUpdate(model)) install(model.modelId)
        else if (model._tag === "Catalog") downloadComponent(model.modelId, "Drafter")
        else Option.map(customModelDrafterOffer(model), offer => setDrafter(model.modelId, { _tag: "Reference", reference: offer.candidate.reference }))
      }} />
      case "Adding": return <span className="tabular-nums text-blue-700 dark:text-blue-300">Adding · {drafterState.percent === null ? "Preparing" : `${drafterState.percent}%`}</span>
      case "Withheld": return <span className="text-slate-500" title={drafterState.note}>Off · doesn’t fit</span>
      case "Pending": return <span className="text-slate-500">{PENDING_DRAFTER_LABEL}</span>
      case "Unknown": return <span className="text-slate-500">—</span>
      case "Attached":
      case "None": {
        const attached = drafterState._tag === "Attached"
        const label = attached ? <><LightningIcon weight="fill" aria-hidden="true" className="size-3.5 shrink-0" /><span className={`truncate ${drafterState.changeable ? "underline decoration-dotted decoration-current/40 underline-offset-4 group-hover:decoration-current" : ""}`}>{drafterState.label}</span></>
          : <span className={drafterState.changeable ? "underline decoration-dotted decoration-current/40 underline-offset-4 group-hover:decoration-current" : ""}>No drafter</span>
        const tone = attached ? "text-orange-600 dark:text-orange-300" : "text-slate-500"
        return <span className="flex min-w-0 items-center gap-2">
          {drafterState.changeable
            ? <button type="button" disabled={pending} onClick={() => setChoosingDrafter(true)} title={attached ? `${drafterState.title} · Change drafter` : "Choose a drafter"}
              className={`group inline-flex min-w-0 items-center gap-1.5 rounded-sm focus-visible:outline-2 focus-visible:outline-blue-500 disabled:opacity-50 ${tone}`}>
              {label}<CaretDownIcon aria-hidden="true" className="size-3 shrink-0 opacity-60 group-hover:opacity-100" />
            </button>
            : <span className={`inline-flex min-w-0 items-center gap-1.5 ${tone}`} {...(attached ? { title: drafterState.title } : {})}>{label}</span>}
          {attached && drafterState.update && <Button variant="unstyled" size="unstyled" disabled={pending} onClick={() => install(model.modelId)} title="A newer version of this drafter is available."
            className="inline-flex h-6 shrink-0 items-center gap-1 rounded-full bg-blue-50 px-2 text-xs font-medium text-blue-700 hover:bg-blue-100 dark:bg-blue-500/15 dark:text-blue-300 dark:hover:bg-blue-500/25"><ArrowClockwiseIcon aria-hidden="true" className="size-3" />Update</Button>}
        </span>
      }
    }
  })()

  const sourceBadge = model._tag === "Catalog" ? null : (() => {
    const { label, href } = customModelSource(model)
    const badge = "inline-flex shrink-0 items-center gap-1 rounded border border-slate-200 px-1 text-[11px] leading-4 font-normal text-slate-500 dark:border-slate-700 dark:text-slate-400"
    return Option.match(href, {
      onNone: () => <span className={badge} title={label}>{customModelSourceLabel(model.origin)}</span>,
      onSome: url => <a href={url} target="_blank" rel="noreferrer" title={label} className={`${badge} hover:border-slate-300 hover:text-slate-700 focus-visible:outline-2 focus-visible:outline-blue-500 dark:hover:border-slate-600 dark:hover:text-slate-200`}>
        <HuggingFaceLogo className="size-3" />Hugging Face<ArrowUpRightIcon aria-hidden="true" className="size-3" />
      </a>,
    })
  })()

  const memoryUse = localModelMemoryUse(model)
  const memory = model.servingState._tag === "Assessed" && model.servingState.assessment._tag === "Fits"
    ? <span className="flex min-w-0 items-center gap-2"><span className="whitespace-nowrap tabular-nums">{formatMemorySize(model.servingState.assessment.memory.totalRequiredBytes)}</span>
      {Option.isSome(memoryUse) && <span aria-hidden="true" className="h-1 w-10 shrink-0 xl:w-14 overflow-hidden rounded-full bg-slate-200 dark:bg-slate-700"><span className={`block h-full rounded-full ${memoryUse.value > 0.8 ? "bg-orange-500 dark:bg-orange-400" : "bg-slate-400"}`} style={{ width: `${Math.max(4, memoryUse.value * 100)}%` }} /></span>}</span>
    : model.servingState._tag === "Assessed" ? <span className="text-slate-500">Doesn’t fit</span>
    : model.servingState._tag === "Assessing" && !awaitingFiles ? <span className="text-slate-500">Checking…</span>
    : <span className="text-slate-500">—</span>

  const attachedAvailability = model._tag === "Custom" ? Option.getOrUndefined(Option.map(model.drafter, value => value.availability)) : undefined
  const notices = <>
    {Option.isSome(deprecation) && <ErrorNotice severity="info" title={Option.isSome(replacement) ? "A replacement model is available" : "This model is no longer supported"} description={Option.isSome(replacement) ? `Switch to ${formatLocalModelDisplayName(replacement.value)} to continue receiving support.` : "Choose another model from Catalog."} className="mt-3" />}
    {Option.isSome(replacement) && replacementCommand.failures.map(failure => <ErrorNotice key={`replacement-${failure.operation}`} {...modelCommandNotice(failure)} className="mt-3" />)}
    {Option.isNone(deprecation) && fit !== null && (refusal !== null || !loadFailure) && !downloadFailure && <ErrorNotice severity="info" title={fit} className="mt-3" />}
    {residency !== undefined && modelStoppedForMemory(residency) && !command.pendingOperations.includes("load") && <ErrorNotice severity="warning" title={MODEL_STOPPED_FOR_MEMORY_MESSAGE} description="It was stopped to keep your other apps running. Quit apps you aren’t using before loading it again." className="mt-3" />}
    {model._tag === "Catalog" && model.catalogData.support._tag === "Disabled" && <ErrorNotice severity="warning" title="This model is unavailable" description="Choose another model from Catalog." className="mt-3" />}
    {acquisition._tag === "Unavailable" && <ErrorNotice severity="warning" title="This model’s files can’t be read" description="Check that the file is still there and its drive is connected." className="mt-3" />}
    {downloadFailure && <ErrorNotice {...downloadNotice(downloadFailure)} className="mt-3" actions={<>
      {canDownload && <NoticeAction disabled={pending} onClick={() => install(model.modelId)}>Retry download</NoticeAction>}
      <NoticeAction disabled={pending} onClick={() => dismissFailure(model.modelId)}>Dismiss</NoticeAction>
    </>} />}
    {acquisition._tag === "RemoveFailed" && <ErrorNotice {...modelRemovalNotice(acquisition.failure)} className="mt-3" />}
    {loadNotice && Option.isNone(deprecation) && <div className="mt-3"><ModelLoadNotice failure={loadNotice} actions={supported && loadNotice.retryable ? <NoticeAction disabled={pending} onClick={requestLoad}>Load again</NoticeAction> : undefined} /></div>}
    {command.failures.map(failure => <ErrorNotice key={failure.operation} {...modelCommandNotice(failure)} className="mt-3" />)}
    {attachedAvailability?._tag === "DownloadFailed" && <ErrorNotice {...downloadNotice(attachedAvailability.failure)} className="mt-3"
      actions={<><NoticeAction disabled={pending} onClick={() => install(model.modelId)}>Retry drafter download</NoticeAction><NoticeAction disabled={pending} onClick={() => dismissFailure(model.modelId)}>Dismiss</NoticeAction></>} />}
  </>

  const facts = localModelFacts(model)
  return <TooltipProvider><article className="min-w-0 overflow-hidden rounded-2xl border border-slate-200 bg-white dark:border-slate-750 dark:bg-slate-850">
    <div className="flex items-center gap-4 px-5 py-4">
      <ModelLogo model={model} className="size-10" />
      <div className="min-w-0 flex-1">
        <h2 title={name} className="flex min-w-0 items-center gap-2 font-sans text-[15px] font-medium leading-6 tracking-normal">
          <span className="min-w-0 truncate">{model.presentation.displayName}</span>
          <span className="shrink-0 rounded bg-slate-100 px-1 font-mono text-[11px] leading-4 font-normal text-slate-500 dark:bg-slate-800 dark:text-slate-400">{model.presentation.variantLabel}</span>
          {sourceBadge}
        </h2>
        <p className="truncate text-xs text-slate-500">{modelSubtitle(model, models, page)}</p>
      </div>
      <CardActions name={name} primary={primary} transfer={transfer} menu={menu} />
    </div>
    <dl className="grid grid-cols-3 gap-px border-t border-slate-200 bg-slate-200 lg:grid-cols-5 dark:border-slate-750 dark:bg-slate-750">
      <StripCell label="Status"><StatusLabel {...status} /></StripCell>
      <StripCell label="Memory">{memory}</StripCell>
      <StripCell label="Drafter">{drafter}</StripCell>
      <StripCell label="Estimated speed">
        {Option.match(facts.speed, { onNone: () => <span className="text-slate-500">—</span>, onSome: speed => <span className="flex min-w-0 items-center gap-1"><span className="truncate tabular-nums" title={speed}>{speed}</span><SpeedInfo /></span> })}
      </StripCell>
      <StripCell label="Intelligence">
        {Option.match(facts.intelligence, { onNone: () => <span className="truncate text-slate-500">Intelligence not rated</span>, onSome: score => <span className="truncate"><span className="tabular-nums">{score}%</span> <span className="text-slate-500">intelligence</span></span> })}
      </StripCell>
    </dl>
    <div className="px-5 pb-4 empty:hidden">{notices}</div>
    {replacing && <ConfirmDialog open={confirmingLoad} onOpenChange={setConfirmingLoad} title={`Load ${name}?`}
      description={`Loading ${name} will stop ${replacing}.`} confirmLabel="Load model" onConfirm={() => load(model.modelId)} />}
    {removable && kind !== "hide" && <ConfirmDialog open={confirmingRemove} onOpenChange={setConfirmingRemove} destructive {...removalConfirmation(model, canStop)} onConfirm={() => remove(model.modelId)} />}
    {model._tag === "Custom" && <DrafterPickerDialog model={model} open={choosingDrafter} onOpenChange={setChoosingDrafter} onChoose={choice => setDrafter(model.modelId, choice)} />}
  </article></TooltipProvider>
}

/** One fact in the strip; its value speaks for itself, so the label is only for screen readers. */
function StripCell({ label, children }: { label: string; children: ReactNode }) {
  return <div className="min-w-0 bg-white px-4 py-2.5 dark:bg-slate-850">
    <dt className="sr-only">{label}</dt>
    <dd className="flex h-6 min-w-0 items-center whitespace-nowrap text-[13px] text-slate-700 dark:text-slate-300">{children}</dd>
  </div>
}

const toneDot: Record<StatusTone, string> = {
  ready: "bg-green-500 shadow-[0_0_0_3px] shadow-green-500/20 dark:bg-green-400",
  busy: "bg-blue-500 motion-safe:animate-pulse dark:bg-blue-400",
  absent: "border border-slate-400 dark:border-slate-500",
  stored: "bg-slate-400",
  problem: "bg-red-500 dark:bg-red-400",
}
const toneText: Record<StatusTone, string> = {
  ready: "text-green-700 dark:text-green-300",
  busy: "text-blue-700 dark:text-blue-300",
  absent: "text-slate-500",
  stored: "",
  problem: "text-red-700 dark:text-red-300",
}

function StatusLabel({ label, tone }: { label: string; tone: StatusTone }) {
  return <span className={`flex min-w-0 items-center gap-2 ${toneText[tone]}`}><span aria-hidden="true" className={`size-2 shrink-0 rounded-full ${toneDot[tone]}`} /><span className="truncate" title={label}>{label}</span></span>
}

function AddDrafterButton({ disabled, onAdd }: { disabled: boolean; onAdd: () => void }) {
  return <Button variant="unstyled" size="unstyled" disabled={disabled} onClick={onAdd}
    className="inline-flex h-6 min-w-0 items-center gap-1 rounded-full border border-dashed border-blue-400/70 px-2.5 text-xs font-medium whitespace-nowrap text-blue-700 hover:bg-blue-50 focus-visible:outline-2 focus-visible:outline-blue-500 dark:text-blue-300 dark:hover:bg-blue-500/10">
    <PlusIcon aria-hidden="true" className="size-3 shrink-0" />Add drafter
  </Button>
}

const menuItem = "flex w-full cursor-default items-center gap-2.5 rounded-md px-2.5 py-2 text-[13px] outline-hidden select-none data-highlighted:bg-slate-100 data-disabled:pointer-events-none data-disabled:opacity-50 dark:data-highlighted:bg-slate-700 [&_svg]:size-4 [&_svg]:shrink-0"
const withTooltip = (label: string | undefined, trigger: ReactElement) => label === undefined ? trigger : <ActionTooltip label={label} trigger={trigger} />

/** The main action as one plain button (a transfer fills it), then a narrow "⋮" holding everything else. */
function CardActions({ name, primary, transfer, menu }: { name: string; primary: PrimaryAction | null; transfer: Transfer | null; menu: readonly MenuAction[] }) {
  // Without a menu, the bar occupies the button (32), gap (1) and menu (6) together.
  const transferWidth = menu.length > 0 ? "w-32" : "w-39"
  const main = transfer
    ? <Button variant="outline" disabled={transfer.pending} onClick={transfer.onCancel} aria-label={`${transfer.cancelLabel} for ${name}`} title={[transfer.label, transfer.estimate, transfer.cancelLabel].filter(Boolean).join(" · ")}
      className={`relative ${transferWidth} justify-between overflow-hidden`}>
      <span aria-hidden="true" className={`absolute inset-y-0 left-0 bg-blue-500/20 transition-[width] duration-500 ease-out ${transfer.percent === null ? "motion-safe:animate-pulse" : ""}`} style={{ width: `${transfer.percent ?? 100}%` }} />
      <span className="relative flex min-w-0 items-center gap-1.5 whitespace-nowrap tabular-nums">
        <span className="shrink-0 text-blue-800 dark:text-blue-200">{transfer.label}</span>
        {transfer.estimate !== null && <><span aria-hidden="true" className="text-slate-400">·</span><span className="min-w-0 truncate text-xs font-normal text-slate-600 dark:text-slate-300" title={transfer.estimate}>{transfer.estimate}</span></>}
      </span>
      <XIcon aria-hidden="true" className="relative text-slate-500" />
    </Button>
    : primary
      ? withTooltip(primary.tooltip, <Button variant={primary.outline ? "outline" : "default"} disabled={primary.disabled} onClick={primary.onClick} {...(primary.title ? { title: primary.title } : {})} className="min-w-32">{primary.icon}{primary.label}</Button>)
      : null
  if (main === null && menu.length === 0) return null
  return <div className="flex shrink-0 items-center gap-1">
    {main}
    {menu.length > 0 && <Menu.Root>
      <Menu.Trigger render={<Button variant="ghost" size="unstyled" aria-label={`More actions for ${name}`} title="More actions" className="h-8 w-6 rounded-md" />}>
        <DotsThreeVerticalIcon weight="bold" aria-hidden="true" className="size-4" />
      </Menu.Trigger>
      <Menu.Portal>
        <Menu.Positioner align="end" sideOffset={4} className="isolate z-50">
          <Menu.Popup className="min-w-52 rounded-lg border border-slate-300 bg-white p-1 font-sans text-slate-900 shadow-md outline-hidden dark:border-slate-600 dark:bg-slate-750 dark:text-slate-100">
            {menu.map((item, index) => <div key={item.label}>
              {item.destructive && index > 0 && <div className="my-1 h-px bg-slate-200 dark:bg-slate-600" />}
              <Menu.Item className={`${menuItem} ${item.destructive ? "text-red-600 dark:text-red-400" : ""}`} disabled={item.disabled} onClick={item.onClick}>{item.icon}{item.label}</Menu.Item>
            </div>)}
          </Menu.Popup>
        </Menu.Positioner>
      </Menu.Portal>
    </Menu.Root>}
  </div>
}
