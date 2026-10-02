import { ErrorNotice, NoticeAction } from "./components/error-notice"
import { ModelLoadFailureIndicator, ModelLoadNotice, modelRemovalNotice, downloadNotice, modelCommandNotice } from "./components/model-error"
import { LoadingRegion, SkeletonLine, ModelsSkeleton, RecommendationsSkeleton, ConnectionsSkeleton } from "./components/page-skeletons"
import { pageLayout } from "./components/page-layout"
import { RecommendationPreference } from "./components/model-preference-slider"
import { ServingUsage } from "./components/serving-usage"
import { setAppearancePreference, useAppearancePreference } from "./stores/appearance-store"
import { ActionTooltip, TooltipProvider } from "./components/ui/tooltip"
import { Button } from "./components/ui/button"
import { Switch } from "./components/ui/switch"
import { Input } from "./components/ui/input"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "./components/ui/select"
import { Progress } from "./components/ui/progress"
import { MagnitudeMark } from "./components/magnitude-mark"
import {
  SidebarSimpleIcon,
  CaretDownIcon,
  EyeIcon,
  ArrowUpRightIcon,
  StackIcon,
  SquaresFourIcon,
  CubeIcon,
  PlugIcon,
  PulseIcon,
  ChartBarIcon,
  CheckCircleIcon,
  SlidersIcon,
  DownloadSimpleIcon,
  CircleNotchIcon,
  PlayIcon,
  SquareIcon,
  TrashIcon,
  XIcon,
  MonitorIcon,
  SunIcon,
  MoonIcon,
  FolderOpenIcon,
  BuildingsIcon,
} from "@phosphor-icons/react"
import { CopyCommand } from "./components/copy-command"
import { useId, useMemo, useRef, useState, type ReactNode } from "react"
import { Atom, Result, useAtomValue, useAtomSet, useAtomMount } from "@effect-atom/atom-react"
import { Effect, Option } from "effect"
import { localModelDeprecation, type ProviderModelId, type CatalogLocalModel, type LocalInferenceHardware, type ModelOptimizationProgress, type ModelResidency } from "@magnitudedev/sdk"
import type { ApplicationSnapshot, AppearancePreference } from "@magnitudedev/sdk/desktop-host"
import type { ApplicationUpdateState, NetworkAccessChange, NetworkBind, OwnerCapability } from "@magnitudedev/sdk"
import { FolderPicker } from "./components/folder-picker"
import {
  activeLocalModel, useAgentClient, type ApplicationPage, type HostNotice,
  useCatalogModels, useLocalModelCommandStatus, useLocalModelMutations, useLocalModelStopStatus, useLocalModels, modelTrayPresentation, useLocalInferenceHardware, formatLocalModelDisplayName,
  describeModelLoadStage, describeModelOptimization, formatModelLoadPercentage, formatModelMemory,
  formatStorageSize, formatTransferRate, formatMemorySize, localModelIsInstalled, localModelProviderModelId, rankedLocalModelOptions, featuredCatalogModels, targetPhysicalMemoryBytes,
  catalogModelReplacement, performanceRangeSpeedLabel, localModelSpeedNote,
  LOCAL_MODEL_RANKING_SCALE_VALUES,
} from "@magnitudedev/client-common"
import { appearanceReadError } from "./appearance"
import { useNavigate, useServerPlatform, useSession } from "./session"
import { ConfirmDialog } from "./components/confirm-dialog"
import { AlertDialog, AlertDialogAction, AlertDialogCancel, AlertDialogContent, AlertDialogDescription, AlertDialogFooter, AlertDialogHeader, AlertDialogTitle } from "./components/ui/alert-dialog"
import { HardwareOverview, ModelRadar, SpeedInfo } from "./components/discovery-visuals"
import { MemoryBreakdown } from "./components/memory-breakdown"
import { HarnessConnections } from "./components/harness-connections"
import { OtherApps } from "./components/other-apps"
import { LabLogo, ModelLogo, modelLab, modelLabs, type ModelLab } from "./components/model-logo"

const ALL_INTERFACES = "all"
const inferenceUrl = (address: string, port: number) => `http://${address}:${port}/inference/v1`
/** The exact command that moves an existing store; the root holds only hub/, locks/ and one JSON file. */
const moveModelsCommand = (platform: string, from: string, to: string) => platform === "win32"
  ? `robocopy "${from}" "${to}" /E /MOVE`
  : `mv "${from}/"* "${to}/"`
const pageNames: Record<ApplicationPage, string> = { discover: "Discover", catalog: "Catalog", models: "My Models", connections: "Connections", usage: "Usage", status: "Status", settings: "Settings" }
const pageIcons = { discover: StackIcon, catalog: SquaresFourIcon, models: CubeIcon, connections: PlugIcon, usage: ChartBarIcon, status: PulseIcon, settings: SlidersIcon }

/** Memory fit is advice, separate from an attempted operation's failure. */
const fitNotice = (model: CatalogLocalModel): string | null => {
  const serving = model.servingState
  if (serving._tag === "Assessing") return "Assessing memory and speed…"
  if (serving._tag === "Failed") return "Assessment failed."
  const assessment = serving.assessment
  if (assessment._tag === "DoesNotFit") return `This model needs ${formatMemorySize(assessment.deficitBytes, { rounding: "up" })} more memory than this computer can provide. Choose a smaller model.`
  return null
}
function ModelDetails({ model, radar = false, open, contentId, compact = false }: { model: CatalogLocalModel; radar?: boolean; open?: boolean; contentId?: string; compact?: boolean }) {
  const serving = model.servingState
  const content = (
    <div className={compact ? "grid gap-5 text-sm" : "mt-3 grid items-start gap-8 border-t border-slate-200 pt-5 dark:border-slate-750 lg:grid-cols-2"}>
      <div className={radar ? "space-y-5" : "contents"}>
      <div className="min-w-0 space-y-5">
        <dl className="flex flex-wrap gap-x-8 gap-y-3">
          <div><dt className="text-xs text-slate-500">License</dt><dd className="mt-1">{Option.getOrElse(model.presentation.license, () => "Not specified")}</dd></div>
          {serving._tag === "Assessed" && <div><dt className="text-xs text-slate-500">Context window</dt><dd className="mt-1">{serving.assessment.profile.contextLength.toLocaleString()} tokens</dd></div>}
          {serving._tag === "Assessed" && serving.capabilities.vision && <div className="self-end"><TooltipProvider><ActionTooltip label="Supports vision" trigger={<button type="button" aria-label="Supports vision" className="rounded p-1 text-slate-500 hover:text-slate-800 focus-visible:outline-2 focus-visible:outline-blue-500 dark:hover:text-slate-200"><EyeIcon aria-hidden="true" className="size-4" /></button>} /></TooltipProvider></div>}
        </dl>
        {model.presentation.sourceUrls.length > 0 && <div><p className="mb-2 text-xs text-slate-500">Sources</p><div className="flex flex-wrap gap-x-4 gap-y-2">{model.presentation.sourceUrls.map(url => {
          const source = new URL(url)
          const label = source.hostname === "huggingface.co" ? `Hugging Face · ${source.pathname.split("/")[1]}` : source.hostname.replace(/^www\./, "")
          return <a className="inline-flex items-center gap-1 text-sm text-slate-600 hover:underline dark:text-slate-300" key={url} href={url} title={url} target="_blank" rel="noreferrer">{label}<ArrowUpRightIcon aria-hidden="true" className="size-3.5" /></a>
        })}</div></div>}
      </div>
      {serving._tag === "Failed" && <div className="min-w-0"><p className="mb-1 text-xs text-slate-500">Assessment</p><p className="break-words text-slate-600 dark:text-slate-300">{serving.failure.message}</p></div>}
      {serving._tag === "Assessed" && serving.assessment._tag === "Fits" && <div className="min-w-0">
        <p className="mb-1 flex items-center gap-1.5 font-medium">Estimated speed on your machine<SpeedInfo /></p>
        <p className="text-sm tabular-nums">{performanceRangeSpeedLabel(serving.assessment.performance, serving.assessment.profile.contextLength)}</p>
        <p className="mt-1 text-xs text-slate-500">{localModelSpeedNote}</p>
      </div>}
      </div>
      {radar && <ModelRadar model={model} />}
    </div>
  )
  if (open !== undefined) return open ? <div id={contentId}>{content}</div> : null
  return <details className="group mt-4 text-sm">
    <summary className="flex cursor-pointer list-none items-center justify-end gap-1 rounded py-1 text-slate-600 focus-visible:outline-2 focus-visible:outline-blue-500 dark:text-slate-300 [&::-webkit-details-marker]:hidden">Model details<CaretDownIcon aria-hidden="true" className="size-4 group-open:rotate-180" /></summary>
    {content}
  </details>
}
/** A catalog model's download and the one-time optimization that follows it. */
const acquiring = (acquisition: CatalogLocalModel["acquisitionState"]) =>
  acquisition._tag === "Installing" || acquisition._tag === "Updating" || acquisition._tag === "Optimizing"
const minutesRemaining = (seconds: number) => `About ${Math.max(1, Math.ceil(seconds / 60))} min`
/** Tuning's time remaining from its observed rate; units are measured configurations, so they track time. */
function useTuningEstimate(progress: ModelOptimizationProgress | null): string {
  const baseline = useRef<{ at: number; completed: number } | null>(null)
  if (progress === null || progress.stage !== "tuning") {
    baseline.current = null
    return "—"
  }
  const now = Date.now()
  if (baseline.current === null || progress.completed < baseline.current.completed) baseline.current = { at: now, completed: progress.completed }
  const elapsed = (now - baseline.current.at) / 1000
  const done = progress.completed - baseline.current.completed
  return elapsed < 5 || done <= 0 ? "Estimating…" : minutesRemaining((progress.total - progress.completed) * elapsed / done)
}
/**
 * One card from the first byte to the model being ready: once the download is verified it turns into
 * the optimization in place. Every line keeps its place, and the finished download's bar fades into
 * tuning progress rather than jumping back.
 */
function DownloadProgress({ acquisition, modelName, onCancel, pending = false, layout = "panel" }: { acquisition: CatalogLocalModel["acquisitionState"]; modelName: string; onCancel?: () => void; pending?: boolean; layout?: "panel" | "row" }) {
  const hardware = useLocalInferenceHardware()
  const optimization = acquisition._tag === "Optimizing" ? acquisition.progress : null
  const tuningEstimate = useTuningEstimate(optimization)
  if (acquisition._tag !== "Installing" && acquisition._tag !== "Updating" && acquisition._tag !== "Optimizing") return null
  const stages = { queued: "Queued", resolving: "Preparing download", checking_space: "Checking space", downloading: "Downloading", verifying: "Verifying download", publishing: "Finishing download" }
  const bar = "absolute inset-y-0 left-0 rounded-md bg-blue-700 transition-[width,opacity] duration-500 ease-out dark:bg-blue-500"
  let heading: string, title: ReactNode, detail: string, percent: number | null, downloadWidth: number, pulse: boolean, stat: { label: string; value: string }, remaining: string, cancelLabel: string
  if (acquisition._tag === "Optimizing") {
    const { stage, completed, total } = acquisition.progress
    heading = describeModelOptimization(acquisition.progress, Result.isSuccess(hardware) ? Option.some(hardware.value) : Option.none())
    title = <span className="min-w-0 flex-1 truncate">{heading}</span>
    detail = "One-time setup for this device"
    percent = stage === "tuning" && total > 0 ? Math.min(100, completed * 100 / total) : null
    downloadWidth = 100
    pulse = percent === null
    stat = { label: "Model", value: modelName }
    remaining = tuningEstimate
    cancelLabel = "Skip optimization"
  } else {
    const { progress } = acquisition
    const downloading = progress.stage === "downloading"
    const rate = downloading ? Option.getOrNull(progress.bytesPerSecond) : null
    heading = stages[progress.stage]
    title = downloading ? <><span className="shrink-0">Downloading</span><span className="min-w-0 flex-1 truncate" title={modelName}>{modelName}</span></> : <span className="min-w-0 flex-1 truncate">{heading}</span>
    detail = `${formatStorageSize(progress.completedBytes)} / ${progress.totalBytes > 0 ? formatStorageSize(progress.totalBytes) : "Unknown total"}`
    percent = progress.totalBytes > 0 ? progress.completedBytes / progress.totalBytes * 100 : null
    downloadWidth = percent ?? 100
    pulse = percent === null
    stat = { label: "Download speed", value: rate !== null ? formatTransferRate(rate) : "—" }
    remaining = downloading && rate !== null && rate > 0 && progress.totalBytes > 0
      ? minutesRemaining((progress.totalBytes - progress.completedBytes) / rate)
      : downloading ? "Estimating…" : "—"
    cancelLabel = "Cancel download"
  }
  const tuning = optimization !== null && percent !== null
  const phase = acquisition._tag === "Optimizing" ? "optimizing" : "download"
  const fade = "motion-safe:animate-[fade-in_400ms_ease-out]"
  const spinner = <CircleNotchIcon aria-hidden="true" className="size-3.5 shrink-0 text-blue-700 motion-safe:animate-spin dark:text-blue-400" />
  const percentText = percent !== null ? `${Math.floor(percent)}%` : "—"
  const cancelButton = (className: string, size?: "sm") => onCancel && <Button variant="ghost" size={size} className={`hover:bg-transparent hover:text-red-600 dark:hover:bg-transparent dark:hover:text-red-400 ${className}`} disabled={pending} onClick={onCancel}><XIcon />{cancelLabel}</Button>
  const progressBar = <div role="progressbar" aria-label={optimization ? "Optimization progress" : "Download progress"} aria-valuemin={0} aria-valuemax={100} aria-valuenow={percent ?? undefined} aria-valuetext={detail} className="relative h-2 w-full overflow-hidden rounded-md bg-slate-100 dark:bg-slate-800">
    <div className={`${bar} ${pulse ? "motion-safe:animate-pulse" : ""}`} style={{ width: `${downloadWidth}%`, opacity: tuning ? 0 : 1 }} />
    <div className={bar} style={{ width: `${tuning ? percent : 0}%`, opacity: tuning ? 1 : 0 }} />
  </div>
  // A model row already names the model: one status line over the bar, one line of detail under it.
  if (layout === "row") {
    const facts = [optimization === null ? stat.value : null, remaining].filter(fact => fact !== null && fact !== "—").join(" · ")
    return <div className="w-full min-w-0" aria-label="Model download">
      <div className="flex h-7 items-center gap-2 text-sm">
        {spinner}
        <span key={phase} className={`min-w-0 flex-1 truncate font-medium ${fade}`}>{heading}</span>
        <span className="shrink-0 tabular-nums text-slate-600 dark:text-slate-300">{percent !== null ? percentText : null}</span>
        {cancelButton("-mr-2.5 ml-2", "sm")}
      </div>
      <div className="mt-2">{progressBar}</div>
      <div className="mt-2 flex items-baseline justify-between gap-3 text-xs tabular-nums text-slate-500 dark:text-slate-400"><span className="min-w-0 truncate">{detail}</span><span className="shrink-0">{facts}</span></div>
    </div>
  }
  return <div className="w-full min-w-0" aria-label="Model download">
    <h3 className="mb-6 flex items-center gap-2 text-sm font-medium">
      <span key={phase} className={`flex min-w-0 flex-1 items-center gap-2 ${fade}`}>{title}</span>
      {spinner}
    </h3>
    {progressBar}
    <div className="mt-3 flex items-baseline justify-between gap-3 text-sm tabular-nums text-slate-600 dark:text-slate-300"><span className="min-w-0 truncate">{detail}</span><span>{percentText}</span></div>
    <dl className="mt-6 grid grid-cols-2 gap-4 text-sm"><div className="min-w-0"><dt className="text-xs text-slate-500 dark:text-slate-400">{stat.label}</dt><dd className="mt-1 truncate font-medium tabular-nums text-slate-800 dark:text-slate-200" title={stat.value}>{stat.value}</dd></div><div className="text-right"><dt className="text-xs text-slate-500 dark:text-slate-400">Time remaining</dt><dd className="mt-1 font-medium tabular-nums text-slate-800 dark:text-slate-200">{remaining}</dd></div></dl>
    {onCancel && <div className="mt-7 flex justify-center">{cancelButton("")}</div>}
  </div>
}
/** One step from a deprecated model to its replacement: download it, or load it once downloaded. */
function SwitchToReplacement({ target }: { target: CatalogLocalModel }) {
  const { install, load } = useLocalModelMutations()
  const command = useLocalModelCommandStatus(target.modelId)
  const fits = target.servingState._tag === "Assessed" && target.servingState.assessment._tag === "Fits"
  const transferring = target.acquisitionState._tag === "Installing" || target.acquisitionState._tag === "Updating"
  return <>
    <Button disabled={command.pending || transferring || !fits} onClick={() => localModelIsInstalled(target) ? load(target.modelId) : install(target.modelId)}>Switch to {formatLocalModelDisplayName(target)}</Button>
  </>
}
/** An installed deprecated model is removable and offers one step to its replacement; it never loads. */
function DeprecatedModelControls({ model, replacement, children }: { model: CatalogLocalModel; replacement: Option.Option<CatalogLocalModel>; children?: ReactNode }) {
  const { remove } = useLocalModelMutations()
  const command = useLocalModelCommandStatus(model.modelId)
  const replacementCommand = useLocalModelCommandStatus(Option.match(replacement, { onNone: () => model.modelId, onSome: target => target.modelId }))
  const acquisition = model.acquisitionState
  const pending = command.pending || acquisition._tag === "Removing"
  return <TooltipProvider><div className="contents">
    <div className="flex flex-wrap items-center justify-end gap-2">{children}
      {Option.match(replacement, { onNone: () => null, onSome: target => <SwitchToReplacement target={target} /> })}
      {localModelIsInstalled(model) && <RemoveDownloadButton model={model} running={false} disabled={pending} onRemove={() => remove(model.modelId)} />}
    </div>
    {Option.isSome(replacement) && replacementCommand.failures.map(failure => <ErrorNotice key={`replacement-${failure.operation}`} {...modelCommandNotice(failure)} className="col-span-full mt-3" />)}
    <ErrorNotice severity="info" title={Option.isSome(replacement) ? "A replacement model is available" : "This model is no longer supported"} description={Option.isSome(replacement) ? `Switch to ${formatLocalModelDisplayName(replacement.value)} to continue receiving support.` : "Choose another model from Catalog."} className="col-span-full mt-3" />
    {acquisition._tag === "RemoveFailed" && <ErrorNotice {...modelRemovalNotice(acquisition.failure)} className="col-span-full mt-3" />}
    {command.failures.map(failure => <ErrorNotice key={failure.operation} {...modelCommandNotice(failure)} className="col-span-full mt-3" />)}
  </div></TooltipProvider>
}
/** `inlineLoadFailure` false leaves a failed load to the caller's own presentation and keeps the Load action available. */
function ModelControls({ model, replacing, children, onConnectAgent, inlineLoadFailure = true }: { model: CatalogLocalModel; replacing?: string; children?: ReactNode; onConnectAgent?: () => void; inlineLoadFailure?: boolean }) {
  const { install, load, stop, cancel, remove, dismissFailure: dismiss } = useLocalModelMutations()
  const command = useLocalModelCommandStatus(model.modelId)
  const stopping = useLocalModelStopStatus()
  const pending = command.pending || stopping.pending || model.acquisitionState._tag === "Removing"
  const acquisition = model.acquisitionState
  const installed = "residencyState" in acquisition
  const residency = installed ? acquisition.residencyState : undefined
  const canStop = residency !== undefined && ["Ready", "Loading", "Requested", "Stopping"].includes(residency._tag)
  const transferring = acquiring(acquisition)
  const fit = model.catalogData.support._tag === "Supported" ? fitNotice(model) : null
  const loadFailure = residency?._tag === "Failed" && !command.pendingOperations.includes("load") ? residency.failure : null
  const loadNotice = inlineLoadFailure ? loadFailure : null
  const downloadFailure = (acquisition._tag === "InstallFailed" || acquisition._tag === "UpdateFailed") && !command.pendingOperations.includes("install") ? acquisition.failure : null
  const canDownload = model.catalogData.support._tag === "Supported" && model.servingState._tag === "Assessed" && model.servingState.assessment._tag === "Fits"
  const [confirmingLoad, setConfirmingLoad] = useState(false)
  const requestLoad = () => { if (replacing) setConfirmingLoad(true); else load(model.modelId) }

  return <TooltipProvider><div className="contents">
    <div className="flex flex-wrap items-center justify-end gap-2">{children}
      {transferring ? null : !installed ? downloadFailure ? null : <Button disabled={pending || model.catalogData.support._tag !== "Supported" || model.servingState._tag !== "Assessed" || model.servingState.assessment._tag !== "Fits"} onClick={() => { install(model.modelId) }}><DownloadSimpleIcon />Download ({formatStorageSize(model.storageBytes).replace(/\s/g, "")})</Button> : <>
        {model.catalogData.support._tag === "Supported" && (onConnectAgent ? <Button className="min-w-28" disabled={pending} onClick={onConnectAgent}><PlugIcon />Connect Agent</Button> : canStop ? <Button className="min-w-28" variant="outline" disabled={stopping.pending} onClick={() => stop()}><SquareIcon />Stop model</Button> : loadNotice ? null : <Button className="min-w-28" disabled={pending} onClick={requestLoad}><PlayIcon />Load model</Button>)}
        {!onConnectAgent && <RemoveDownloadButton model={model} running={canStop} disabled={pending} onRemove={() => remove(model.modelId)} />}
        {model.catalogData.support._tag === "Supported" && acquisition._tag === "UpdateAvailable" && <Button variant="outline" disabled={pending} onClick={() => install(model.modelId)}>Update</Button>}
      </>}

    </div>
    {transferring && <div className="col-span-full mt-1"><DownloadProgress layout="row" modelName={formatLocalModelDisplayName(model)} acquisition={acquisition} pending={command.pending} onCancel={() => cancel(model.modelId)} /></div>}
    {fit !== null && !loadFailure && !downloadFailure && <ErrorNotice severity="info" title={fit} className="col-span-full mt-3" />}
    {model.catalogData.support._tag === "Disabled" && <ErrorNotice severity="warning" title="This model is unavailable" description="Choose another model from Catalog." className="col-span-full mt-3" />}
    {downloadFailure && <ErrorNotice {...downloadNotice(downloadFailure)} className="col-span-full mt-3" actions={<>
      {canDownload && <NoticeAction disabled={pending} onClick={() => install(model.modelId)}>Retry download</NoticeAction>}
      <NoticeAction disabled={pending} onClick={() => dismiss(model.modelId)}>Dismiss</NoticeAction>
    </>} />}
    {acquisition._tag === "RemoveFailed" && <ErrorNotice {...modelRemovalNotice(acquisition.failure)} className="col-span-full mt-3" />}
    {loadNotice && <div className="col-span-full mt-3"><ModelLoadNotice failure={loadNotice} actions={model.catalogData.support._tag === "Supported" && loadNotice.retryable && !onConnectAgent ? <NoticeAction disabled={pending} onClick={requestLoad}>Load again</NoticeAction> : undefined} /></div>}
    {command.failures.map(failure => <ErrorNotice key={failure.operation} {...modelCommandNotice(failure)} className="col-span-full mt-3" />)}
    {replacing && <ConfirmDialog open={confirmingLoad} onOpenChange={setConfirmingLoad} title={`Load ${formatLocalModelDisplayName(model)}?`}
      description={`Loading ${formatLocalModelDisplayName(model)} will stop ${replacing}.`} confirmLabel="Load model" onConfirm={() => load(model.modelId)} />}
  </div></TooltipProvider>
}
/** Removing a download always asks first; a running model is stopped before its files are removed. */
function RemoveDownloadButton({ model, running, disabled, onRemove }: { model: CatalogLocalModel; running: boolean; disabled: boolean; onRemove: () => void }) {
  const [confirming, setConfirming] = useState(false)
  const name = formatLocalModelDisplayName(model)
  return <>
    <Button variant="ghost" size="icon" aria-label={`Remove ${name}`} title="Remove download" disabled={disabled} onClick={() => setConfirming(true)}><TrashIcon /></Button>
    <ConfirmDialog open={confirming} onOpenChange={setConfirming} destructive
      title={running ? `Stop ${name} and remove its downloaded files?` : `Remove the downloaded files for ${name}?`}
      description="You can download it again later." confirmLabel={running ? "Stop and remove" : "Remove"} onConfirm={onRemove} />
  </>
}
function ModelCard({ model, models, showMemory = false, replacing }: { model: CatalogLocalModel; models: readonly CatalogLocalModel[]; showMemory?: boolean; replacing?: string }) {
  const [detailsOpen, setDetailsOpen] = useState(false)
  const detailsId = useId()
  const deprecation = localModelDeprecation(model)
  const detailsToggle = <Button variant="ghost" aria-expanded={detailsOpen} aria-controls={detailsId} onClick={() => setDetailsOpen(value => !value)}>Details<CaretDownIcon aria-hidden="true" className={`size-4 ${detailsOpen ? "rotate-180" : ""}`} /></Button>
  const acquisition = model.acquisitionState
  const residency = "residencyState" in acquisition ? acquisition.residencyState : undefined
  const statusLabel = acquisition._tag === "Removing" ? "Removing…" : acquisition._tag === "RemoveFailed" ? "Removal failed" : residency?._tag === "Ready" ? "Loaded" : residency?._tag === "Unloaded" ? "Downloaded" : residency?._tag === "Failed" ? "Not loaded" : residency?._tag === "Requested" ? "Preparing…" : residency?._tag ?? (acquisition._tag === "NotInstalled" ? "" : acquisition._tag === "InstallFailed" ? "Not downloaded" : acquisition._tag === "UpdateFailed" ? "Update incomplete" : acquisition._tag === "UpdateAvailable" ? "Update available" : acquisition._tag)
  const status = (statusLabel || showMemory) && <div className="mt-1 flex flex-wrap items-center gap-x-3 text-sm text-slate-500">{statusLabel && <span className={residency?._tag === "Ready" ? "text-green-600 dark:text-green-400" : ""}>{statusLabel}</span>}{showMemory && model.servingState._tag === "Assessed" && model.servingState.assessment._tag === "Fits" && <><span aria-hidden="true">·</span><span>{formatMemorySize(model.servingState.assessment.memory.totalRequiredBytes)} memory</span></>}</div>
  return <article className={pageLayout.modelCard}>
    <div className={pageLayout.modelRow}>
      <div className="flex min-w-0 items-center gap-4"><ModelLogo model={model} /><div className="min-w-0"><h2 className="flex flex-wrap items-center gap-2 text-lg font-semibold">{formatLocalModelDisplayName(model)}</h2>{status}</div></div>
      {Option.match(deprecation, {
        onNone: () => <ModelControls model={model} {...(replacing ? { replacing } : {})}>{detailsToggle}</ModelControls>,
        onSome: value => <DeprecatedModelControls model={model} replacement={catalogModelReplacement(models, value)}>{detailsToggle}</DeprecatedModelControls>,
      })}
    </div>
    <ModelDetails model={model} radar open={detailsOpen} contentId={detailsId} />
  </article>
}
function SelectedRecommendation({ model, active }: { model: CatalogLocalModel; active: ReturnType<typeof activeLocalModel> }) {
  const client = useAgentClient()
  const navigate = useNavigate()
  const connectAgent = () => navigate("connections")
  const [view, setView] = useState<"profile" | "details">("profile")
  const { cancel } = useLocalModelMutations()
  const command = useLocalModelCommandStatus(model.modelId)
  const transferring = acquiring(model.acquisitionState)
  return <div className={`relative ${pageLayout.recommendationPane}`} aria-label="Selected model profile">
    <div className={transferring ? "invisible" : undefined} inert={transferring} aria-hidden={transferring}>
    <div className={pageLayout.recommendationToolbar}>
    <div className="flex items-center gap-1" aria-label="Model information">
      <Button variant={view === "profile" ? "secondary" : "ghost"} aria-pressed={view === "profile"} onClick={() => setView("profile")}>Profile</Button>
      <Button variant={view === "details" ? "secondary" : "ghost"} aria-pressed={view === "details"} onClick={() => setView("details")}>Details</Button>
    </div>
      {transferring ? <Button disabled><DownloadSimpleIcon />Download ({formatStorageSize(model.storageBytes).replace(/\s/g, "")})</Button> : <ModelControls model={model} inlineLoadFailure={false} onConnectAgent={() => connectAgent()} {...(Option.isSome(active) && active.value.model.modelId !== model.modelId ? { replacing: formatLocalModelDisplayName(active.value.model) } : {})} />}
    </div>
    <div className="grid min-h-72">
      <div className={`col-start-1 row-start-1 min-w-0 ${view === "profile" ? "" : "invisible"}`} aria-hidden={view !== "profile"}><ModelRadar model={model} /></div>
      <div className={`col-start-1 row-start-1 min-w-0 ${view === "details" ? "" : "invisible"}`} aria-hidden={view !== "details"}><ModelDetails model={model} compact open /></div>
    </div>
    </div>
    {transferring && <div className="absolute inset-5 flex items-center justify-center overflow-y-auto" aria-label="Download panel">
      <div className="w-full max-w-sm px-3 py-4">
        <DownloadProgress modelName={formatLocalModelDisplayName(model)} acquisition={model.acquisitionState} pending={command.pending} onCancel={() => cancel(model.modelId)} />
        {command.failures.map(failure => <ErrorNotice key={failure.operation} {...modelCommandNotice(failure)} className="mt-3" />)}
      </div>
    </div>}
  </div>
}
function Recommendations({ models, active, preference }: { models: readonly CatalogLocalModel[]; preference: number; active: ReturnType<typeof activeLocalModel> }) {
  const [selection, setSelection] = useState<{ preference: number; modelId: CatalogLocalModel["modelId"] | null }>({ preference, modelId: null })
  if (selection.preference !== preference) setSelection({ preference, modelId: null })
  const selectedId = selection.preference === preference ? selection.modelId : null
  const downloads = models.filter(model => acquiring(model.acquisitionState))
  const selectable = downloads.length > 0 ? downloads : models
  const selected = selectable.find(model => model.modelId === selectedId) ?? selectable[0]
  if (!selected) return null
  return <section aria-label="Top recommendations" className="mb-8">
    <TooltipProvider><div className={pageLayout.recommendations}>
      <div className={pageLayout.recommendationList} aria-label="Recommended models">{models.map((model, rank) => <button key={model.modelId} type="button" aria-pressed={model.modelId === selected.modelId} onClick={() => setSelection({ preference, modelId: model.modelId })} className={`${pageLayout.recommendationRow} focus-visible:outline-2 focus-visible:outline-blue-500 ${model.modelId === selected.modelId ? "border-blue-300 bg-blue-50 dark:border-blue-700 dark:bg-slate-800" : "border-transparent hover:bg-slate-100 dark:hover:bg-slate-800"}`}>
        <span className="w-4 shrink-0 text-sm tabular-nums text-slate-500">{rank + 1}</span>
        <ModelLogo model={model} className="size-7" />
        <span className="flex min-w-0 flex-1 items-center text-sm font-medium">
          <span className="min-w-0 truncate"
            onMouseEnter={({ currentTarget }) => {
              if (currentTarget.scrollWidth > currentTarget.clientWidth) currentTarget.title = currentTarget.textContent?.trimEnd() ?? ""
            }}
            onMouseLeave={({ currentTarget }) => currentTarget.removeAttribute("title")}
          >{model.presentation.displayName}{"\u00a0"}</span>
          <span className="shrink-0 whitespace-nowrap">({model.presentation.variantLabel})</span>
        </span>
        {"residencyState" in model.acquisitionState && model.acquisitionState.residencyState._tag === "Failed" && <ModelLoadFailureIndicator failure={model.acquisitionState.residencyState.failure} />}
      </button>)}</div>
      <SelectedRecommendation model={selected} active={active} />
    </div></TooltipProvider>
  </section>
}
function LabOption({ lab }: { lab: ModelLab | null }) {
  return <span className="flex items-center gap-2">{lab ? <LabLogo lab={lab} className="size-4" /> : <BuildingsIcon aria-hidden="true" className="size-4 text-slate-500" />}{lab ? lab.name : "Any lab"}</span>
}
function Models({ page }: { page: "discover" | "catalog" | "models" }) {
  const installedOnly = page === "models"
  const discover = page === "discover"
  const catalog = useCatalogModels()
  const localModels = useLocalModels()
  const active = Result.isSuccess(localModels) ? Option.getOrUndefined(activeLocalModel(localModels.value)) : undefined
  const stopResult = useLocalModelStopStatus()
  const hardware = useLocalInferenceHardware()
  const [search, setSearch] = useState("")
  const filterOptions = installedOnly
    ? [{ value: "all", label: "All models" }, { value: "downloaded", label: "Downloaded" }, { value: "downloading", label: "Downloading" }]
    : [{ value: "all", label: "All models" }, { value: "fits", label: "Fits my machine" }]
  const sortOptions = [
    ...(!installedOnly ? [{ value: "recommended", label: "Recommended" }] : []),
    { value: "name", label: "Name A–Z" }, { value: "smallest", label: "Smallest download" }, { value: "largest", label: "Largest download" },
  ]
  const [filter, setFilter] = useState("all")
  const [lab, setLab] = useState<ModelLab | null>(null)
  const [sort, setSort] = useState(installedOnly ? "name" : "recommended")
  const client = useAgentClient()
  const session = useSession()
  const preference = useAtomValue(session.rankingPreference)
  const setPreference = useAtomSet(useMemo(() => Atom.fn((index: number) => session.setRankingPreference(index)), [session]))
  if (Result.isFailure(catalog)) return <>{!discover && <h1 className={pageLayout.pageTitle}>{pageNames[page]}</h1>}<ErrorNotice title="Couldn’t load the model catalog" description="Model information is unavailable. Check the service on Status." className="mt-5" /></>
  if (!Result.isSuccess(catalog) && !discover) return <ModelsSkeleton page={page} />
  const models = (Result.isSuccess(catalog) ? catalog.value.models : []).filter((model): model is CatalogLocalModel => model._tag === "Catalog")
  const ranked = !installedOnly && Result.isSuccess(hardware) ? rankedLocalModelOptions(models.map(model => ({ id: model.modelId, kind: localModelIsInstalled(model) ? "stored" as const : "downloadable" as const, model })), { fastToSmart: LOCAL_MODEL_RANKING_SCALE_VALUES[preference]!, memoryBudgetBytes: targetPhysicalMemoryBytes(hardware.value) }, models.length).flatMap(option => option.model._tag === "Catalog" ? [option.model] : []) : []
  const assessment = Result.isSuccess(catalog) ? catalog.value.preparation.assessment : undefined
  const recommendationsPending = !Result.isFailure(hardware) && (Result.isInitial(hardware) || !assessment?.complete)
  const rankedIds = new Set(ranked.map(model => model.modelId))
  const ordered = installedOnly ? models : [...ranked, ...models.filter(model => !rankedIds.has(model.modelId))]
  // Deprecated and disabled models are listed only where installed.
  const library = ordered.filter(model => model.acquisitionState._tag !== "NotInstalled"
    || !installedOnly && model.catalogData.support._tag === "Supported")
  const labOptions = modelLabs.filter(entry => library.some(model => modelLab(model) === entry))
  const visible = library.filter(model => {
    const acquisition = model.acquisitionState
    const matchesFilter = filter === "all"
      || filter === "fits" && model.servingState._tag === "Assessed" && model.servingState.assessment._tag === "Fits"
      || filter === "downloaded" && localModelIsInstalled(model)
      || filter === "downloading" && acquiring(acquisition)
    return matchesFilter && (lab === null || modelLab(model) === lab) && `${formatLocalModelDisplayName(model)} ${model.presentation.description}`.toLowerCase().includes(search.trim().toLowerCase())
  })
  if (sort !== "recommended") visible.sort((a, b) => {
    const byName = formatLocalModelDisplayName(a).localeCompare(formatLocalModelDisplayName(b), undefined, { numeric: true }) || a.modelId.localeCompare(b.modelId)
    return sort === "smallest" ? a.storageBytes - b.storageBytes || byName : sort === "largest" ? b.storageBytes - a.storageBytes || byName : byName
  })
  return <>
    {!discover && <>
      <div className={pageLayout.modelHeader}>
        <h1 className={pageLayout.pageTitle}>{pageNames[page]}</h1>
        <span className="text-sm tabular-nums text-slate-500" role="status">{visible.length} {visible.length === 1 ? "model" : "models"}</span>
      </div>
      <div className={pageLayout.catalogToolbar}>
        <div className="flex flex-wrap items-center gap-2">
          <Select items={filterOptions} value={filter} onValueChange={value => { if (value !== null) setFilter(value) }}>
            <SelectTrigger aria-label="Filter models"><SelectValue /></SelectTrigger>
            <SelectContent>{filterOptions.map(option => <SelectItem key={option.value} value={option.value}>{option.label}</SelectItem>)}</SelectContent>
          </Select>
          <Select value={lab} onValueChange={setLab}>
            <SelectTrigger aria-label="Filter by lab"><SelectValue>{(value: ModelLab | null) => <LabOption lab={value} />}</SelectValue></SelectTrigger>
            <SelectContent>{[<SelectItem key="any" value={null}><LabOption lab={null} /></SelectItem>, ...labOptions.map(option => <SelectItem key={option.name} value={option}><LabOption lab={option} /></SelectItem>)]}</SelectContent>
          </Select>
          <Select items={sortOptions} value={sort} onValueChange={value => { if (value !== null) setSort(value) }}>
            <SelectTrigger aria-label="Sort models"><span className="text-slate-500">Sort:</span><SelectValue /></SelectTrigger>
            <SelectContent>{sortOptions.map(option => <SelectItem key={option.value} value={option.value}>{option.label}</SelectItem>)}</SelectContent>
          </Select>
        </div>
        <Input aria-label="Search models" placeholder="Search models…" className={pageLayout.modelSearch} value={search} onChange={event => setSearch(event.target.value)} />
      </div>
    </>}
    {discover && <HardwareOverview /> }
    {Option.isSome(stopResult.failure) && <ErrorNotice title="Couldn’t stop the model" description={stopResult.failure.value} className="mt-5" />}
    {discover && <RecommendationPreference value={preference} onChange={setPreference} />}
    {!discover && assessment && !assessment.complete && <p className="mb-4 text-sm text-slate-500">Assessing models · {assessment.settledModels} of {assessment.totalModels}</p>}
    {discover && (recommendationsPending
      ? <RecommendationsSkeleton assessment={assessment} waitingForHardware={Result.isInitial(hardware)} />
      : <Recommendations preference={preference} models={featuredCatalogModels(ranked, 5)} active={Option.fromNullable(active)} />)}
    {!discover && <>
    <div className="grid items-start gap-5">{visible.map(model => <ModelCard key={model.modelId} model={model} models={models} showMemory={installedOnly} {...(active && active.model.modelId !== model.modelId ? { replacing: formatLocalModelDisplayName(active.model) } : {})} />)}</div>
    {visible.length === 0 && <p className="py-8 text-slate-500">{search.trim() || filter !== "all" || lab !== null ? "No models match your search or filter." : installedOnly ? "No models downloaded yet. Find one in Discover." : "No models match this filter."}</p>}
    </>}
    {discover && ranked.length === 0 && !recommendationsPending && Result.isSuccess(hardware) && <p className="py-8 text-slate-500">No fitting recommendations right now. Explore Catalog for memory and speed details.</p>}
  </>
}
function Connections({ serviceReady, selectedModel }: { serviceReady: boolean; selectedModel: Option.Option<ProviderModelId> }) {
  const service = useSession()
  const models = useLocalModels()
  const canConnect = serviceReady && Result.isSuccess(models) && models.value.models.some(model => Option.isSome(localModelProviderModelId(model)))
  const navigate = useNavigate()
  const hardware = useLocalInferenceHardware()
  const state = useAtomValue(service.application)
  const platform = useServerPlatform()
  const available = Result.isSuccess(models) ? models.value.models.filter(model => Option.isSome(localModelProviderModelId(model))) : []
  const active = Result.isSuccess(models) ? Option.getOrUndefined(activeLocalModel(models.value))?.model.modelId : undefined
  const ranked = Result.isSuccess(hardware) ? rankedLocalModelOptions(available.map(model => ({ id: model.modelId, kind: "stored" as const, model })), { fastToSmart: 0.5, memoryBudgetBytes: targetPhysicalMemoryBytes(hardware.value) }, available.length).map(option => option.model) : available
  const commandModels = ranked.map(model => ({ id: model.modelId, label: formatLocalModelDisplayName(model) }))
  const defaultModel = commandModels.find(model => model.id === active)?.id ?? commandModels[0]?.id
  const client = useAgentClient()
  const rows = useAtomValue(client.Connections.WatchHarnessConnections({})).result
  const connect = useAtomSet(client.Connections.ConnectHarness)
  const disconnect = useAtomSet(client.Connections.DisconnectHarness)
  const connecting = useAtomValue(client.Connections.ConnectHarness)
  const disconnecting = useAtomValue(client.Connections.DisconnectHarness)
  const busy = connecting.waiting || disconnecting.waiting
  const error = !busy && firstFailure([connecting, disconnecting])
  return <>
    {!serviceReady && <p className="mt-5 text-sm text-slate-500">Configuration checks are available. Start the service from Status before connecting a harness.</p>}
    {serviceReady && !canConnect && !Result.isInitial(models) && <ErrorNotice severity={Result.isFailure(models) ? "error" : "info"} title={Result.isFailure(models) ? "Couldn’t check available models" : "Download a model to connect an agent"} description={Result.isFailure(models) ? "Check the service on Status." : "Choose a compatible model. It doesn’t need to be loaded."} className="mt-5" actions={!Result.isFailure(models) && <NoticeAction onClick={() => navigate("discover")}>Discover models</NoticeAction>} />}
    {error && Result.isFailure(error) && <ErrorNotice title={Result.isFailure(disconnecting) ? "Couldn’t disconnect this agent" : "Couldn’t connect this agent"} description="Check the agent’s configuration before trying again. Some changes may not have completed." className="mt-5" />}
    {Result.isFailure(rows) ? <ErrorNotice title="Couldn’t check your connections" description="Connection status is unavailable. Magnitude will check again automatically." className="mt-5" />
      : !Result.isSuccess(rows) ? <ConnectionsSkeleton />
      : rows.value._tag === "Unavailable" ? <ErrorNotice title="Couldn’t check your connections" description="Magnitude can’t read the saved connection information. Check that its configuration is accessible." className="mt-5" />
      : <HarnessConnections connections={rows.value.connections} busy={busy} canConnect={canConnect} models={commandModels} defaultModel={defaultModel} platform={platform}
          onConnect={harness => connect({ harness, model: selectedModel, installSkill: true })} onDisconnect={harness => disconnect({ harness })} />}
    {Result.isSuccess(state) && <OtherApps origin={`http://127.0.0.1:${new URL(state.value.endpoint).port}`} model={defaultModel} platform={platform} onOpenSettings={() => navigate("settings")} />}
  </>
}
/** The model row's status: the load stage in full while loading, the memory in use once loaded. */
const modelStatusText = (residency: ModelResidency, hardware: Option.Option<LocalInferenceHardware>): string => {
  switch (residency._tag) {
    case "Requested": return describeModelLoadStage("preparing", Option.none(), hardware)
    case "Loading": return describeModelLoadStage(residency.stage, residency.plannedAllocation, hardware)
    case "Ready": return `Loaded · ${formatModelMemory(residency.allocation)}`
    case "Stopping": return "Stopping…"
    case "Unloaded":
    case "Failed": return "Not loaded"
  }
}
/** A load's completed fraction; a requested load has not started. */
const loadFraction = (residency: ModelResidency): Option.Option<number> => {
  switch (residency._tag) {
    case "Requested": return Option.some(0)
    case "Loading": return Option.some(residency.fraction)
    default: return Option.none()
  }
}
function ModelStatus() {
  const models = useLocalModels()
  const hardware = useLocalInferenceHardware()
  const { stop } = useLocalModelMutations()
  const stopping = useLocalModelStopStatus()
  const presentation = Result.isSuccess(models) ? modelTrayPresentation(models.value) : null
  const active = Result.isSuccess(models) ? Option.getOrUndefined(activeLocalModel(models.value)) : undefined
  if (Result.isInitial(models)) return <LoadingRegion label="Loading model status" className="mt-5"><div className="flex h-12 items-center gap-3"><SkeletonLine className="h-6 w-64" /></div></LoadingRegion>
  if (Result.isSuccess(models) && !active) return <div className="mt-5 flex min-h-12 items-center"><p className="m-0 text-base text-slate-500 dark:text-slate-400">Your model loads automatically when you start chatting.</p></div>
  return <div className="mt-5">
    <div className="flex min-h-12 items-center justify-between gap-5">
      <div className="flex min-w-0 items-center gap-3">
        {active ? <ModelLogo model={active.model} className="size-6 shrink-0" /> : <CubeIcon aria-hidden="true" className="size-5 shrink-0 text-slate-400 dark:text-slate-500" />}
        <div className="flex min-w-0 items-baseline gap-2">
          <p title={active ? formatLocalModelDisplayName(active.model) : undefined} className={`m-0 min-w-0 truncate ${active ? "text-base font-medium text-slate-800 dark:text-slate-200" : "text-sm text-slate-500 dark:text-slate-400"}`}>{active ? formatLocalModelDisplayName(active.model) : presentation?.label ?? (Result.isFailure(models) ? "Model status unavailable" : "Reading model status…")}</p>
          {active && <><span aria-hidden="true" className="text-slate-400 dark:text-slate-500">·</span><span className={`shrink-0 text-xs ${active.residency._tag === "Ready" ? "text-green-700 dark:text-green-400" : "text-slate-500 dark:text-slate-400"}`}>{modelStatusText(active.residency, Result.isSuccess(hardware) ? Option.some(hardware.value) : Option.none())}</span></>}
        </div>
      </div>
      {presentation?.canStop && <Button variant="outline" className="hover:border-red-300 hover:text-red-600 dark:hover:border-red-800 dark:hover:text-red-400" disabled={stopping.pending} onClick={() => stop()}><SquareIcon />Stop model</Button>}
    </div>
    {active && Option.match(loadFraction(active.residency), {
      onNone: () => null,
      onSome: fraction => <div className="mt-4 flex items-center gap-3">
        <Progress aria-label="Model loading progress" className="flex-1" indicatorClassName="bg-blue-700 dark:bg-blue-500" value={fraction * 100} />
        <span className="w-9 shrink-0 text-right text-xs tabular-nums text-slate-500 dark:text-slate-400">{formatModelLoadPercentage(fraction)}</span>
      </div>,
    })}
    {Result.isFailure(models) && <ErrorNotice title="Couldn’t read model status" description="Magnitude can’t confirm whether a model is running." className="mt-2" />}
    {Option.isSome(stopping.failure) && <ErrorNotice title="Couldn’t stop the model" description={stopping.failure.value} className="mt-2" />}
  </div>
}
function DownloadActivity() {
  const models = useLocalModels()
  const active = Result.isSuccess(models) ? models.value.models.filter((model): model is CatalogLocalModel => model._tag === "Catalog" && (acquiring(model.acquisitionState) || model.acquisitionState._tag === "Removing")) : []
  if (Result.isInitial(models)) return null
  if (Result.isSuccess(models) && active.length === 0) return null
  return <div className="mt-5 border-t border-slate-200 pt-5 dark:border-slate-700">
    {!Result.isSuccess(models) ? <p className="mt-3 text-sm text-slate-500">{Result.isFailure(models) ? "Download activity unavailable" : "Reading download activity…"}</p> : <ul className="space-y-5">{active.map(model => <li key={model.modelId}><div className="flex items-center gap-3"><ModelLogo model={model} className="size-6" /><p className="text-sm">{formatLocalModelDisplayName(model)} · {model.acquisitionState._tag === "Removing" ? "Removing files…" : model.acquisitionState._tag === "Optimizing" ? "Optimizing" : model.acquisitionState._tag === "Updating" ? "Updating" : "Downloading"}</p></div><DownloadProgress modelName={formatLocalModelDisplayName(model)} acquisition={model.acquisitionState} /></li>)}</ul>}
  </div>
}
function Status({ snapshot }: { snapshot: typeof ApplicationSnapshot.Type | null }) {
  const session = useSession()
  const retry = useAtomSet(session.retryService)
  const retrying = useAtomValue(session.retryService)
  const service = snapshot?.service
  const tray = snapshot?.owner._tag === "Desktop" ? snapshot.owner.tray : undefined
  const ready=service?._tag === "Ready"
  return <div className={pageLayout.statusStack}>
    <section aria-busy={!snapshot} aria-label={!snapshot ? "Loading service status" : undefined} className={pageLayout.statusHero}>
      <div className="flex items-center justify-between gap-4 border-b border-slate-200 pb-4 dark:border-slate-700">
        <h2 className="m-0 text-base font-medium text-slate-600 dark:text-slate-300">Magnitude service</h2>
        <div className={`flex h-8 items-center gap-2 rounded-full px-3 text-sm font-medium ${ready ? "bg-green-200/20 text-green-700 dark:bg-green-800/20 dark:text-green-400" : "text-slate-500 dark:text-slate-400"}`}>
          {ready ? <CheckCircleIcon aria-label="Service ready" weight="fill" className="size-5 shrink-0" /> : <PulseIcon className="size-4 shrink-0" />}
          <span>{!snapshot ? <SkeletonLine className="h-4 w-16 text-xs" /> : ready ? "Ready" : service?._tag === "CleanupFailed" ? "Cleanup needs attention" : service?._tag === "Failed" ? "Unavailable" : "Starting"}</span>
        </div>
      </div>
      {ready ? <ModelStatus /> : !snapshot ? <SkeletonLine className="mt-5 h-12 w-64" /> : service?._tag === "Failed" || service?._tag === "CleanupFailed" ? null : <p className="mt-5 text-sm text-slate-500">Starting the service…</p>}
      {ready && <DownloadActivity />}
      {service && "message" in service && <ErrorNotice className="mt-5" title={service._tag === "CleanupFailed" ? "Couldn’t confirm the service has stopped" : "The service couldn’t start"}
        description={service._tag === "CleanupFailed" ? "Some background work may still be running. Quit Magnitude to retry cleanup." : "Check that another copy of Magnitude isn’t running, then retry the service."}
        actions={service._tag === "Failed" ? <NoticeAction disabled={retrying.waiting} onClick={() => retry()}>Retry service</NoticeAction> : undefined} />}
      {Result.isFailure(retrying) && !retrying.waiting && service?._tag !== "Failed" && service?._tag !== "Ready" && <ErrorNotice title="Couldn’t retry the service" className="mt-3" />}
    </section>
    <MemoryBreakdown />
    {ready && <StatusOverview />}
    {tray?._tag === "Unavailable" && <ErrorNotice severity="warning" title="The tray icon isn’t available" description="Closing this window keeps Magnitude running. Open it again from your applications menu." />}
  </div>
}
const compact = new Intl.NumberFormat(undefined, { notation: "compact", maximumFractionDigits: 1 })
function StatusTile({ label, value, detail, onClick }: { label: string; value: ReactNode; detail: ReactNode; onClick: () => void }) {
  return <button type="button" onClick={onClick} className="min-w-0 cursor-pointer rounded-lg p-3 text-left transition-colors hover:bg-slate-50 focus-visible:outline-2 focus-visible:outline-blue-500 dark:hover:bg-slate-800">
    <span className="block text-xs text-slate-500">{label}</span>
    <span className="mt-1 block truncate font-heading text-xl tabular-nums">{value}</span>
    <span className="mt-0.5 block truncate text-xs text-slate-500">{detail}</span>
  </button>
}
function StatusOverview() {
  const client = useAgentClient()
  const session = useSession()
  const navigate = useNavigate()
  const usage = useAtomValue(client.Models.GetServingUsage({ period: "Today", timeZone: Intl.DateTimeFormat().resolvedOptions().timeZone, model: Option.none() })).result
  const rows = useAtomValue(client.Connections.WatchHarnessConnections({})).result
  const network = useAtomValue(client.Configuration.GetNetworkAccess({})).result
  const today = Result.isSuccess(usage) && usage.value._tag === "Available" ? usage.value : null
  const installed = Result.isSuccess(rows) && rows.value._tag === "Ready" ? rows.value.connections.filter(row => row.installed) : null
  const connected = installed?.filter(row => row.inspection._tag === "Connected").length
  const address = Result.isSuccess(network) && network.value.enabled ? Option.getOrElse(network.value.bind, () => "All interfaces") : null
  const loading = <SkeletonLine className="h-4 text-xs" width="80px" />
  return <section aria-label="Activity" className={`${pageLayout.card} grid grid-cols-3 gap-2 p-3`}>
    <StatusTile label="Today" onClick={() => navigate("usage")}
      value={today ? `${compact.format(today.totalTokens)} tokens` : Result.isInitial(usage) ? loading : "Unavailable"}
      detail={today ? `${today.requests.toLocaleString()} ${today.requests === 1 ? "request" : "requests"}` : null} />
    <StatusTile label="Agents" onClick={() => navigate("connections")}
      value={connected !== undefined ? `${connected} connected` : Result.isInitial(rows) ? loading : "Unavailable"}
      detail={installed ? `${installed.length} installed` : null} />
    <StatusTile label="Network access" onClick={() => navigate("settings")}
      value={Result.isSuccess(network) ? network.value.enabled ? "On" : "Off" : Result.isInitial(network) ? loading : "Unavailable"}
      detail={Result.isSuccess(network) ? address ?? "This computer only" : null} />
  </section>
}
type UpdateTransfer = (typeof ApplicationUpdateState.Type)["transfer"]
const firstFailure = (results: ReadonlyArray<Result.Result<unknown, unknown>>) => results.find((result): result is Result.Failure<unknown, unknown> => Result.isFailure(result) && !result.waiting)
function SettingsGroup({ label, children }: { label: string; children: ReactNode }) {
  return <section aria-label={label} className="mt-7">
    <h2 className="mb-2 px-1 text-xs font-medium uppercase tracking-wide text-slate-500">{label}</h2>
    <div className="divide-y divide-slate-200 rounded-lg border border-slate-300 bg-white dark:divide-slate-800 dark:border-slate-750 dark:bg-slate-850">{children}</div>
  </section>
}
function SettingsRow({ label, hint, alert, control, children, nested = false }: { label: ReactNode; hint?: ReactNode; alert?: ReactNode; control?: ReactNode; children?: ReactNode; nested?: boolean }) {
  return <div className={nested ? "bg-slate-50 py-2.5 pl-10 pr-4 dark:bg-slate-900/40" : "px-4 py-3"}>
    <div className="flex items-center justify-between gap-6">
      <div className="min-w-0"><p className={nested ? "text-[13px] font-medium" : "text-sm font-medium"}>{label}</p>
        {hint && <div className="mt-0.5 text-xs text-slate-500">{hint}</div>}

      </div>
      {control && <div className="flex shrink-0 items-center gap-2">{control}</div>}
    </div>
    {alert && <div className="mt-2">{alert}</div>}
    {children}
  </div>
}
function ThemeRow() {
  const appearance = useAppearancePreference()
  const session = useSession()
  const saveAppearance = useMemo(() => Atom.fn((preference: AppearancePreference) => session.saveAppearance(preference).pipe(
    Effect.tap(() => Effect.sync(() => setAppearancePreference(preference))))), [session])
  const save = useAtomSet(saveAppearance)
  const saving = useAtomValue(saveAppearance)
  const readError = useAtomValue(appearanceReadError)
  return <SettingsRow label="Theme" alert={!saving.waiting && (Result.isFailure(saving) ? <ErrorNotice title="Your theme wasn’t saved" description="Your previous appearance setting is still in use." /> : readError ? <ErrorNotice title="Couldn’t read your saved theme" description="System appearance is being used for this window." /> : undefined)} control={
    <div className="inline-flex rounded-md border border-slate-300 p-0.5 dark:border-slate-700" role="group" aria-label="Theme">
      {(["system", "light", "dark"] as const).map(value => { const Icon = value === "system" ? MonitorIcon : value === "light" ? SunIcon : MoonIcon
        return <Button key={value} size="sm" variant={appearance === value ? "secondary" : "ghost"} aria-pressed={appearance === value} disabled={saving.waiting} onClick={() => save(value)}><Icon />{value[0]!.toUpperCase() + value.slice(1)}</Button> })}
    </div>} />
}
/** The application that owns the service, and whether it supports a request. */
function useApplicationOwner() {
  const client = useAgentClient()
  const owner = useAtomValue(client.Application.WatchApplicationOwner({})).result
  const supports = (capability: OwnerCapability) => Result.isSuccess(owner) && owner.value.capabilities.includes(capability)
  return { owner, supports }
}
function LaunchAtLoginRow() {
  const client = useAgentClient()
  const { owner, supports } = useApplicationOwner()
  const state = Result.flatMap(owner, value => Option.match(value.loginStartup, { onNone: () => Result.initial(), onSome: login => Result.success(login) }))
  const setLaunchAtLogin = useAtomSet(client.Application.SetLaunchAtLogin)
  const set = (enabled: boolean) => setLaunchAtLogin({ enabled })
  const change = useAtomValue(client.Application.SetLaunchAtLogin)
  if (Result.isSuccess(owner) && !supports("LaunchAtLogin")) return null
  const current = Result.isSuccess(state) ? state.value : null
  const enabled = current?._tag === "Enabled" || current?._tag === "RequiresApproval"
  const hint = current?._tag === "Unavailable" ? current.message
    : current?._tag === "RequiresApproval" ? "Allow Magnitude in your system login settings to finish enabling startup."
    : "Starts in the background with its tray icon."
  const alert = !change.waiting && Result.isFailure(change) ? <ErrorNotice title="Couldn’t update launch at login" description="Check Magnitude’s status in your system startup settings." /> : Result.isFailure(state) ? <ErrorNotice title="Couldn’t check launch at login" description="Magnitude can’t confirm whether it will open when you sign in." /> : undefined
  return <SettingsRow label="Launch at login" hint={Result.isInitial(state) ? <SkeletonLine className="h-4 text-xs" width="160px" /> : hint} alert={alert}
    control={<Switch aria-label="Launch at login" checked={enabled} disabled={!current || current._tag === "Unavailable" || change.waiting} onCheckedChange={checked => set(checked)} />} />
}
function ModelStorageRow() {
  const client = useAgentClient()
  const settings = useAtomValue(client.Configuration.GetModelStorage({})).result
  const save = useAtomSet(client.Configuration.SetModelStorage)
  const saving = useAtomValue(client.Configuration.SetModelStorage)
  const [choosing, setChoosing] = useState(false)
  const busy = saving.waiting
  const current = Result.isSuccess(settings) ? settings.value : null
  const failure = firstFailure([saving])
  return <>
    <FolderPicker open={choosing} onOpenChange={setChoosing} initialPath={current ? Option.some(current.path) : Option.none()}
      title="Choose a folder for downloaded models" description="Folders on the computer running Magnitude." confirmLabel="Use this folder"
      onChoose={path => save({ path: Option.some(path) })} />
    <SettingsRow label="Model storage" alert={!busy && failure ? <ErrorNotice title="The model folder wasn’t saved" description="Check that the folder is available and writable, then try again." /> : Result.isFailure(settings) ? <ErrorNotice title="Couldn’t read the model folder setting" description="The folder used by the running service has not been changed." /> : current && Option.isSome(current.warning) ? <ErrorNotice severity="warning" title="The saved model folder is invalid" description="The default folder is selected. Choose a different folder to save a valid location." /> : undefined}
      hint={current ? <span className="block truncate" title={current.path}>Current path is <span className="text-slate-700 dark:text-slate-300" data-testid="model-storage-path">{current.path}</span>{current.source === "Default" && " (default)"}</span>  : Result.isInitial(settings) ? <SkeletonLine className="h-4 text-xs" width="220px" /> : undefined}
      control={<>
        {current?.source === "Configured" && <Button size="sm" variant="ghost" disabled={busy} onClick={() => save({ path: Option.none() })}>Use default</Button>}
        <Button size="sm" variant="outline" disabled={!current || busy} onClick={() => setChoosing(true)}><FolderOpenIcon />Change…</Button>
      </>} />
  </>
}
function RestartRequiredToast() {
  const session = useSession()
  const client = useAgentClient()
  const platform = useServerPlatform()
  const storage = useAtomValue(client.Configuration.GetModelStorage({})).result
  const network = useAtomValue(client.Configuration.GetNetworkAccess({})).result
  const { supports } = useApplicationOwner()
  const restartApplication = useAtomSet(client.Application.RestartApplication)
  const relaunch = () => restartApplication({})
  const relaunching = useAtomValue(client.Application.RestartApplication)
  const storageValue = Result.isSuccess(storage) ? storage.value : null
  const storagePending = storageValue !== null && storageValue.path !== storageValue.active
  const networkPending = Result.isSuccess(network) && network.value.pending
  if (!storagePending && !networkPending) return null
  const reason = storagePending && networkPending ? "Magnitude is still using the previous model folder and network settings."
    : storagePending ? "Magnitude is still using the previous model folder." : "Magnitude is still using the previous network settings."
  return <div className="fixed bottom-4 right-4 z-50 w-96 max-w-[calc(100vw-2rem)] rounded-lg bg-white shadow-md dark:bg-slate-850">
    <ErrorNotice severity={Result.isFailure(relaunching) && !relaunching.waiting ? "error" : "info"}
      title={Result.isFailure(relaunching) && !relaunching.waiting ? "Magnitude couldn’t restart" : "Restart to apply your changes"}
      description={supports("RestartService") ? reason : `${reason} Restart the Magnitude server to apply them.`}
      actions={supports("RestartService") ? <NoticeAction disabled={relaunching.waiting} onClick={() => relaunch()}>Restart Magnitude</NoticeAction> : undefined}>
      {storagePending && storageValue && <details className="mt-2 text-xs text-slate-600 dark:text-slate-400">
        <summary className="w-fit cursor-pointer rounded py-0.5 hover:underline focus-visible:outline-2 focus-visible:outline-blue-500">Moving existing models</summary>
        <p className="my-2">Existing downloads stay in the previous folder. To move them, quit Magnitude, run this command, then open Magnitude again.</p>
        <CopyCommand command={moveModelsCommand(platform, storageValue.active, storageValue.path)} label="Copy move command" />
      </details>}
    </ErrorNotice>
  </div>
}

function NetworkAccessRows() {
  const client = useAgentClient()
  const settings = useAtomValue(client.Configuration.GetNetworkAccess({})).result
  const save = useAtomSet(client.Configuration.SetNetworkAccess)
  const update = (change: Partial<NetworkAccessChange>) => save({ enabled: Option.none(), bind: Option.none(), requireApiKey: Option.none(), ...change })
  const updating = useAtomValue(client.Configuration.SetNetworkAccess)
  const regenerate = useAtomSet(client.Configuration.RegenerateNetworkApiKey)
  const regenerating = useAtomValue(client.Configuration.RegenerateNetworkApiKey)
  const busy = updating.waiting || regenerating.waiting
  const current = Result.isSuccess(settings) ? settings.value : null
  const failure = firstFailure([updating, regenerating])
  const reachable = current?.enabled ? Option.getOrElse(current.bind, () => current.interfaces[0]?.address) : undefined
  return <>
    <SettingsRow label="Network access" hint={current ? <>Let other devices on your network use Magnitude for inference. <a href="https://docs.magnitude.dev/remote-server" target="_blank" rel="noreferrer" className="inline-flex items-center gap-0.5 font-medium text-slate-700 hover:underline dark:text-slate-300">Remote server guide<ArrowUpRightIcon aria-hidden="true" className="size-3" /></a></> : Result.isInitial(settings) ? <SkeletonLine className="h-4 text-xs" width="240px" /> : undefined}
      alert={!busy && failure ? <ErrorNotice title="Network settings weren’t saved" description="Your previous saved settings are still in use." /> : Result.isFailure(settings) ? <ErrorNotice title="Couldn’t read network settings" description="The running service’s network settings have not been changed." /> : current && Option.isSome(current.warning) ? <ErrorNotice severity="warning" title="The saved network address is invalid" description="All interfaces are selected. Choose an address below to save a valid setting." /> : undefined}
      control={<Switch aria-label="Network access" checked={current?.enabled ?? false} disabled={!current || busy} onCheckedChange={checked => update({ enabled: Option.some(checked) })} />} />
    {current?.enabled && <>
      <SettingsRow nested label="Address" hint={current.interfaces.length === 0 ? "No network interfaces were found." : "Which of this computer's addresses accepts connections."}
        control={<Select items={[{ value: ALL_INTERFACES, label: "All interfaces" }, ...current.interfaces.map(entry => ({ value: entry.address, label: `${entry.address} (${entry.kind === "tailscale" ? "Tailscale" : entry.name})` }))]}
          value={Option.getOrElse(current.bind, () => ALL_INTERFACES)} onValueChange={value => update({ bind: Option.some<NetworkBind>(value === ALL_INTERFACES || value === null ? { _tag: "AllInterfaces" } : { _tag: "Address", address: String(value) }) })}>
          <SelectTrigger aria-label="Network address" className="min-w-56"><SelectValue /></SelectTrigger>
          <SelectContent>{[<SelectItem key={ALL_INTERFACES} value={ALL_INTERFACES}>All interfaces</SelectItem>, ...current.interfaces.map(entry => <SelectItem key={entry.address} value={entry.address}>{entry.address} ({entry.kind === "tailscale" ? "Tailscale" : entry.name})</SelectItem>)]}</SelectContent>
        </Select>} />
      <SettingsRow nested label="API key" hint={current.requireApiKey ? "Other devices must send this key as a Bearer token." : "Other devices can connect without a key. Only do this on a network you trust."}
        control={<><Button size="sm" variant="ghost" disabled={busy} onClick={() => regenerate({})}>Regenerate</Button><Switch aria-label="Require API key" checked={current.requireApiKey} disabled={busy} onCheckedChange={checked => update({ requireApiKey: Option.some(checked) })} /></>}>
        {Option.isSome(current.apiKey) && current.requireApiKey && <div className="mt-2"><CopyCommand command={current.apiKey.value} label="Copy API key" /></div>}
      </SettingsRow>
      {reachable && <SettingsRow nested label="Reachable at" hint={`Use this as the OpenAI-compatible base URL on other devices${current.requireApiKey ? ", with the API key above" : ""}. Anthropic-compatible apps use /inference/anthropic on the same address.`}>
        <div className="mt-2"><CopyCommand command={inferenceUrl(reachable, current.port)} label="Copy base URL" /></div>
      </SettingsRow>}
    </>}
  </>
}
function AutomaticUpdatesRow() {
  const client = useAgentClient()
  const { owner, supports } = useApplicationOwner()
  const observation = Result.flatMap(owner, value => Option.match(value.updates, { onNone: () => Result.initial(), onSome: updates => Result.success(updates) }))
  const saveAutoDownload = useAtomSet(client.Application.SetApplicationAutoDownload)
  const setAutoDownload = (enabled: boolean) => saveAutoDownload({ enabled })
  const saving = useAtomValue(client.Application.SetApplicationAutoDownload)
  if (Result.isSuccess(owner) && !supports("Updates")) return null
  const snapshot = Result.isSuccess(observation) ? observation.value : null
  const preference = snapshot?.preference
  const closed = snapshot?.transfer._tag === "Closed"
  return <SettingsRow label="Automatic updates"
    hint={Result.isInitial(observation) ? <SkeletonLine className="h-4 text-xs" width="200px" /> : preference?._tag === "Known" ? "Download updates in the background when they are available." : undefined}
    alert={Result.isFailure(saving) && !saving.waiting ? <ErrorNotice title="Update preferences weren’t saved" description="Your previous preference is still in use." /> : Result.isFailure(observation) || preference?._tag === "Unavailable" ? <ErrorNotice title="Couldn’t read update preferences" description="Automatic downloads are unavailable until your preference can be read." /> : undefined}
    control={<Switch aria-label="Automatic updates" checked={preference?._tag === "Known" && preference.autoDownload} disabled={preference?._tag !== "Known" || saving.waiting || closed} onCheckedChange={checked => setAutoDownload(checked)} />} />
}
function AboutRow() {
  const service = useSession()
  const client = useAgentClient()
  const health = useAtomValue(client.Connection.Health({})).result
  const version = Result.isSuccess(health) ? `Magnitude ${health.value.version}` : Result.isFailure(health) ? "Magnitude" : <SkeletonLine className="h-5 text-sm" width="120px" />
  const { owner, supports } = useApplicationOwner()
  const observation = Result.flatMap(owner, value => Option.match(value.updates, { onNone: () => Result.initial(), onSome: updates => Result.success(updates) }))
  const checkUpdate = useAtomSet(client.Application.CheckApplicationUpdate)
  const check = () => checkUpdate({})
  const discardUpdate = useAtomSet(client.Application.DiscardApplicationUpdate)
  const discard = () => discardUpdate({})
  const discarding = useAtomValue(client.Application.DiscardApplicationUpdate)
  const downloadUpdate = useAtomSet(client.Application.DownloadApplicationUpdate)
  const download = () => downloadUpdate({})
  const installUpdate = useAtomSet(client.Application.InstallApplicationUpdate)
  const restart = () => installUpdate({})
  const checking = useAtomValue(client.Application.CheckApplicationUpdate)
  const downloading = useAtomValue(client.Application.DownloadApplicationUpdate)
  const restarting = useAtomValue(client.Application.InstallApplicationUpdate)
  const snapshot = Result.isSuccess(observation) ? observation.value : null
  const current: UpdateTransfer | undefined = snapshot?.transfer
  const pending = downloading.waiting || restarting.waiting || discarding.waiting
  const message = !current ? Result.isFailure(observation) ? "Update status unavailable." : "Reading update status…"
    : current._tag === "Idle" ? snapshot?.check._tag === "Succeeded" ? "You’re up to date." : "Checks for updates automatically."
    : current._tag === "Available" ? `Version ${current.version} is available · ${formatStorageSize(current.bytes)}`
    : current._tag === "Downloading" ? `Downloading version ${current.version} · ${formatStorageSize(current.completed)} of ${formatStorageSize(current.total)}`
    : current._tag === "Cancelling" ? "Stopping automatic download…"
    : current._tag === "Staging" ? `Preparing version ${current.version}…`
    : current._tag === "Ready" ? `Version ${current.version} is ready. Restarting stops the running model and service.`
    : current._tag === "Closed" ? "Magnitude is quitting…" : current._tag === "Unavailable" ? "Updates aren’t available right now." : undefined
  const actionFailure = firstFailure([checking, downloading, restarting, discarding])
  const checkFailed = snapshot?.check._tag === "Failed"
  const installable = current?._tag === "Ready" || current?._tag === "InstallationFailed"
  const hasFailure = Boolean(actionFailure || checkFailed || current?._tag === "Failed" || current?._tag === "InstallationFailed")
  const busy = pending || checking.waiting || snapshot?.check._tag === "Checking"
  const actions = <>
    {installable ? <><NoticeAction disabled={busy} onClick={() => restart()}>Retry update</NoticeAction><NoticeAction disabled={busy} onClick={() => discard()}>Discard download</NoticeAction></>
      : current?._tag === "Available" ? <NoticeAction disabled={busy} onClick={() => download()}>Download update</NoticeAction>
      : current && !["Unavailable", "Closed"].includes(current._tag) ? <NoticeAction disabled={busy} onClick={() => check()}>Check for updates</NoticeAction> : null}
  </>
  return <SettingsRow label={version} hint={Result.isInitial(observation) ? <SkeletonLine className="h-4 text-xs" width="160px" /> : message}
    alert={hasFailure && !busy ? <ErrorNotice
      title={Result.isFailure(discarding) && !discarding.waiting ? "Couldn’t discard the update" : current?._tag === "InstallationFailed" ? "The update wasn’t completed" : checkFailed ? "Couldn’t check for updates" : "The update couldn’t finish"}
      description={installable ? "The prepared update is still available. Retry it, or discard its download." : "Check your connection before trying again."}
      actions={actions} /> : undefined}
    control={hasFailure ? busy ? <span className="text-xs text-slate-500">Working…</span> : null : <>
      {installable && <Button size="sm" variant="ghost" disabled={pending} onClick={() => discard()}>Discard download</Button>}
      {installable ? <Button size="sm" disabled={pending} onClick={() => restart()}>Restart to update</Button>
        : current?._tag === "Available" ? <Button size="sm" disabled={pending} onClick={() => download()}>Download update</Button>
        : current && !["Unavailable", "Closed"].includes(current._tag) ? <Button size="sm" variant="outline" disabled={busy} onClick={() => check()}>{busy ? "Checking…" : "Check for updates"}</Button>
        : null}
    </>} />
}
function SettingsPage() {
  // A fresh mount observes hand edits; writes refresh in their own Effect actions.
  const client = useAgentClient()
  useAtomMount(useMemo(() => Atom.make(Effect.all([Atom.refresh(client.Configuration.GetModelStorage({})), Atom.refresh(client.Configuration.GetNetworkAccess({}))], { discard: true })), [client]))
  return <>
    <SettingsGroup label="General"><ThemeRow /><LaunchAtLoginRow /><ModelStorageRow /><NetworkAccessRows /><AutomaticUpdatesRow /></SettingsGroup>
    <SettingsGroup label="About"><AboutRow /></SettingsGroup>
  </>
}
export function App() {
  const session = useSession()
  const state = useAtomValue(session.application)
  const page = useAtomValue(session.page)
  const navigate = useNavigate()
  const service = Result.isSuccess(state) ? state.value.service : null
  const platform = Option.getOrUndefined(Option.map(session.clientWindow, window => window.platform))
  return <><RestartRequiredToast /><HostNotices /><QuitFailureDialog /><AppShell page={page} navigate={navigate} platform={platform}>
      {page === "status" ? Result.isFailure(state) ? <ErrorNotice title="Couldn’t read service status" description="Magnitude can’t confirm the service’s current state." className="mt-7" /> : <Status snapshot={Result.isSuccess(state) ? state.value : null} />
      : page === "usage" ? <ServingUsage />
      : page === "settings" ? <SettingsPage />
      : page === "connections" ? <Connections serviceReady={service?._tag === "Ready"} selectedModel={Option.none()} />
      : service?._tag !== "Ready" ? (service?._tag === "Failed" || service?._tag === "CleanupFailed" || Result.isFailure(state) ? <>{page !== "discover" && <h1 className={pageLayout.pageTitle}>{pageNames[page]}</h1>}<ErrorNotice title="Magnitude needs your attention" description="The inference service is unavailable." className="mt-8" actions={<NoticeAction onClick={() => navigate("status")}>Open Status</NoticeAction>} /></> : <ModelsSkeleton page={page} />)
      : page === "discover" || page === "catalog" || page === "models" ? <Models page={page} />
      : null}
  </AppShell></>
}
/** Messages raised by Electron main while this window exists, such as a link that couldn't open. */
function HostNotices() {
  const session = useSession()
  const notices = useAtomValue(session.notices)
  const dismiss = useAtomSet(useMemo(() => Atom.fn((notice: HostNotice) => session.dismissNotice(notice)), [session]))
  if (notices.length === 0) return null
  return <div className="fixed bottom-4 left-4 z-50 flex w-96 max-w-[calc(100vw-2rem)] flex-col gap-2">
    {notices.map((notice, index) => <div key={index} className="rounded-lg bg-white shadow-md dark:bg-slate-850">
      <ErrorNotice title={notice.title} description={notice.description} actions={<NoticeAction onClick={() => dismiss(notice)}>Dismiss</NoticeAction>} />
    </div>)}
  </div>
}
/** Electron main couldn't confirm background work stopped while quitting; the user decides what happens next. */
function QuitFailureDialog() {
  const session = useSession()
  const open = useAtomValue(session.quitFailed)
  const resolve = useAtomSet(session.resolveQuitFailure)
  const resolving = useAtomValue(session.resolveQuitFailure)
  return <AlertDialog open={open} onOpenChange={next => { if (!next && open) resolve("KeepOpen") }}>
    <AlertDialogContent>
      <AlertDialogHeader>
        <AlertDialogTitle>Magnitude could not finish quitting</AlertDialogTitle>
        <AlertDialogDescription>Background processes could not be confirmed stopped. Retry Quit tries to stop background work again. Force Quit closes Magnitude even though some background processes may still be running.</AlertDialogDescription>
      </AlertDialogHeader>
      <AlertDialogFooter>
        <AlertDialogCancel disabled={resolving.waiting} onClick={() => resolve("KeepOpen")}>Keep Magnitude Open</AlertDialogCancel>
        <Button variant="outline" disabled={resolving.waiting} onClick={() => resolve("ForceQuit")}>Force Quit</Button>
        <AlertDialogAction disabled={resolving.waiting} onClick={() => resolve("RetryQuit")}>Retry Quit</AlertDialogAction>
      </AlertDialogFooter>
    </AlertDialogContent>
  </AlertDialog>
}
export function AppShell({ page, navigate, platform, children }: { page: ApplicationPage; navigate?: (page: ApplicationPage) => void; platform: string | undefined; children: ReactNode }) {
  const [collapsed, setCollapsed] = useState(false)
  const integratedControls = platform === "darwin" || platform === "win32"
  const sidebarWidth = collapsed ? 0 : 224
  return <div className="relative flex h-screen bg-slate-50 font-sans text-slate-900 dark:bg-slate-925 dark:text-slate-200">
    {integratedControls && <div aria-hidden="true" data-window-drag-region style={{ left: sidebarWidth }} className="absolute right-0 top-0 z-50 h-8 select-none transition-[left] duration-250 ease-in-out motion-reduce:transition-none [-webkit-app-region:drag]" />}
    <div data-window-drag-region={integratedControls ? "" : undefined} style={{ width: collapsed ? (platform === "darwin" ? 128 : 64) : sidebarWidth }} className={`absolute left-0 top-0 z-50 flex h-[42px] items-center justify-end px-4 transition-[width] duration-250 ease-in-out motion-reduce:transition-none ${integratedControls ? "select-none [-webkit-app-region:drag]" : ""}`}>
      <button type="button" className="inline-flex size-6 items-center justify-center rounded-sm text-slate-500 hover:text-slate-900 focus-visible:outline-2 focus-visible:outline-blue-500 dark:hover:text-slate-100 [-webkit-app-region:no-drag]" aria-label={collapsed ? "Expand sidebar" : "Collapse sidebar"} aria-expanded={!collapsed} aria-controls="desktop-navigation" onClick={() => setCollapsed(value => !value)}>
        <SidebarSimpleIcon className="size-5" />
      </button>
    </div>
    <aside aria-hidden={collapsed} inert={collapsed} style={{ width: sidebarWidth }} className="shrink-0 overflow-hidden transition-[width] duration-250 ease-in-out motion-reduce:transition-none">
      <div className={`flex h-full w-56 flex-col border-r border-slate-200 pt-10 pb-8 transition-transform duration-250 ease-in-out motion-reduce:transition-none dark:border-slate-750 ${collapsed ? "-translate-x-full" : "translate-x-0"}`}>
      <div className="mb-10 mt-4 flex h-8 shrink-0 items-center gap-3 px-7 font-heading text-base font-semibold">
        <MagnitudeMark className="h-8 w-8 shrink-0" />Magnitude
      </div>
      <nav id="desktop-navigation" className="flex min-h-0 flex-1 flex-col gap-2 px-4">
        {(Object.keys(pageNames) as ApplicationPage[]).map(key => {
          const Icon = pageIcons[key]
          return <Button variant="ghost" key={key} disabled={!navigate} onClick={() => navigate?.(key)} aria-label={pageNames[key]} aria-current={page === key ? "page" : undefined} className={`h-10 gap-3 rounded-lg px-3 text-left text-sm font-medium justify-start ${key === "status" ? "mt-auto" : ""} ${page === key ? "bg-blue-50 text-blue-700 dark:bg-slate-800 dark:text-blue-400" : "hover:bg-slate-100 dark:hover:bg-slate-800"}`}>
            <Icon className="size-4 shrink-0" />{pageNames[key]}
          </Button>
        })}
      </nav>
      </div>
    </aside>
    <main key={page} className="min-w-0 flex-1 overflow-y-auto">
      <div data-page-content className={`mx-auto w-[calc(100vw-224px)] max-w-6xl px-10 pb-9 ${platform === "win32" ? "pt-14" : "pt-9"}`}>
        {page !== "catalog" && page !== "models" && <h1 className={pageLayout.pageTitle}>{pageNames[page]}</h1>}
        {children}
      </div>
    </main>
  </div>
}

