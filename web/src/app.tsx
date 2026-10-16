import { ErrorNotice, NoticeAction } from "./components/error-notice"
import { LoadModelButton, fitNotice, loadRefusal } from "./components/model-fit"
import { ModelLoadFailureIndicator, ModelLoadNotice, modelRemovalNotice, downloadNotice, modelCommandNotice } from "./components/model-error"
import { LoadingRegion, SkeletonLine, ModelsSkeleton, RecommendationsSkeleton, ConnectionsSkeleton } from "./components/page-skeletons"
import { pageLayout } from "./components/page-layout"
import { RecommendationPreference } from "./components/model-preference-slider"
import { ServingUsage } from "./components/serving-usage"
import { setAppearancePreference, useAppearancePreference } from "./stores/appearance-store"
import { useI18n, isLanguage, type MessageKey, readLanguage, translate, translateKnownMessage } from "./i18n"
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
import { Cause, Effect, Option } from "effect"
import { localModelDeprecation, type ProviderModelId, type CatalogLocalModel, type LocalInferenceHardware, type ModelOptimizationProgress, type ModelResidency } from "@magnitudedev/sdk"
import type { AppearancePreference } from "@magnitudedev/sdk/desktop-host"
import { ServerSettingsFailed, type ApplicationUpdateState, type NetworkAccessChange, type NetworkBind, type OwnerCapability } from "@magnitudedev/sdk"
import { FolderPicker } from "./components/folder-picker"
import {
  activeLocalModel, useAgentClient, type ApplicationPage, type HostNotice,
  useCatalogModels, useLocalModelCommandStatus, useLocalModelMutations, useLocalModelStopStatus, useLocalModels, modelTrayPresentation, useLocalInferenceHardware, formatLocalModelDisplayName,
  describeModelLoadStage, describeModelOptimization, formatModelLoadPercentage, formatModelMemory,
  formatStorageSize, formatTransferRate, formatMemorySize, localModelIsInstalled, localModelProviderModelId, rankedLocalModelOptions, featuredCatalogModels, targetPhysicalMemoryBytes, modelStoppedForMemory,
  catalogModelReplacement, performanceRangeSpeedLabel,
  LOCAL_MODEL_RANKING_SCALE_VALUES,
} from "@magnitudedev/client-common"
import { appearanceReadError } from "./appearance"
import { useNavigate, useServerPlatform, useSession } from "./session"
import { useReconnectAttempt, useServiceObservation, type ServiceObservation } from "./service-view"
import { ConfirmDialog } from "./components/confirm-dialog"
import { signOut, useDisconnectWarning, useRemoteAccess, viewerPlatform } from "./remote-access"
import { useNarrowViewport } from "./lib/viewport"
import { AlertDialog, AlertDialogAction, AlertDialogCancel, AlertDialogContent, AlertDialogDescription, AlertDialogFooter, AlertDialogHeader, AlertDialogTitle } from "./components/ui/alert-dialog"
import { HardwareOverview, ModelRadar, SpeedInfo } from "./components/discovery-visuals"
import { MemoryBreakdown } from "./components/memory-breakdown"
import { HarnessConnections, type CommandModel } from "./components/harness-connections"
import { HarnessSetupList } from "./components/harness-setup"
import { OtherApps } from "./components/other-apps"
import { LabLogo, ModelLogo, modelLab, modelLabs, type ModelLab } from "./components/model-logo"

const ALL_INTERFACES = "all"
const inferenceUrl = (address: string, port: number) => `http://${address}:${port}/inference/v1`
/** The exact command that moves an existing store; the root holds only hub/, locks/ and one JSON file. */
const moveModelsCommand = (platform: string, from: string, to: string) => platform === "win32"
  ? `robocopy "${from}" "${to}" /E /MOVE`
  : `mv "${from}/"* "${to}/"`
const pageNames: Record<ApplicationPage, string> = { discover: "Discover", catalog: "Catalog", models: "My Models", connections: "Connections", usage: "Usage", status: "Status", settings: "Settings" }
function T({ k }: { k: MessageKey }) { const { t } = useI18n(); return t(k) }
const tr = (key: MessageKey) => translate(readLanguage(), key)
function localizeModelText(text: string): string {
  const known: Record<string, MessageKey> = {
    "Not assessed": "notAssessed",
    "Waiting for memory…": "waitingForMemory",
    "Preparing…": "preparingEllipsis",
    "Loading weights…": "loadingWeights",
    "Finalizing…": "finalizing",
    "Optimizing…": "optimizingEllipsis",
    "Preparing to optimize…": "preparingToOptimize",
  }
  if (known[text]) return tr(known[text]!)
  if (text.startsWith("Optimizing for ") && text.endsWith("…")) {
    return tr("optimizingForDevice").replace("{device}", text.slice("Optimizing for ".length, -1))
  }
  return text
}
function localizedPageName(page: ApplicationPage, t: (key: import("./i18n").MessageKey) => string) {
  const keys = { discover: "discover", catalog: "catalog", models: "myModels", connections: "connections", usage: "usage", status: "status", settings: "settings" } as const
  return t(keys[page])
}
const pageIcons = { discover: StackIcon, catalog: SquaresFourIcon, models: CubeIcon, connections: PlugIcon, usage: ChartBarIcon, status: PulseIcon, settings: SlidersIcon }

function ModelDetails({ model, radar = false, open, contentId, compact = false }: { model: CatalogLocalModel; radar?: boolean; open?: boolean; contentId?: string; compact?: boolean }) {
  const serving = model.servingState
  const content = (
    <div className={compact ? "grid gap-5 text-sm" : "mt-3 grid items-start gap-8 border-t border-slate-200 pt-5 dark:border-slate-750 lg:grid-cols-2"}>
      <div className={radar ? "space-y-5" : "contents"}>
      <div className="min-w-0 space-y-5">
        <dl className="flex flex-wrap gap-x-8 gap-y-3">
          <div><dt className="text-xs text-slate-500"><T k="license" /></dt><dd className="mt-1">{Option.getOrElse(model.presentation.license, () => tr("licenseNotSpecified"))}</dd></div>
          {serving._tag === "Assessed" && <div><dt className="text-xs text-slate-500"><T k="contextWindow" /></dt><dd className="mt-1">{serving.assessment.profile.contextLength.toLocaleString()} tokens</dd></div>}
          {serving._tag === "Assessed" && serving.capabilities.vision && <div className="self-end"><TooltipProvider><ActionTooltip label={tr("supportsVision")} trigger={<button type="button" aria-label={tr("supportsVision")} className="rounded p-1 text-slate-500 hover:text-slate-800 focus-visible:outline-2 focus-visible:outline-blue-500 dark:hover:text-slate-200"><EyeIcon aria-hidden="true" className="size-4" /></button>} /></TooltipProvider></div>}
        </dl>
        {model.presentation.sourceUrls.length > 0 && <div><p className="mb-2 text-xs text-slate-500"><T k="sources" /></p><div className="flex flex-wrap gap-x-4 gap-y-2">{model.presentation.sourceUrls.map(url => {
          const source = new URL(url)
          const label = source.hostname === "huggingface.co" ? `Hugging Face · ${source.pathname.split("/")[1]}` : source.hostname.replace(/^www\./, "")
          return <a className="inline-flex items-center gap-1 text-sm text-slate-600 hover:underline dark:text-slate-300" key={url} href={url} title={url} target="_blank" rel="noreferrer">{label}<ArrowUpRightIcon aria-hidden="true" className="size-3.5" /></a>
        })}</div></div>}
      </div>
      {serving._tag === "Failed" && <div className="min-w-0"><p className="mb-1 text-xs text-slate-500"><T k="assessment" /></p><p className="break-words text-slate-600 dark:text-slate-300">{translateKnownMessage(serving.failure.message)}</p></div>}
      {serving._tag === "Assessed" && serving.assessment._tag === "Fits" && <div className="min-w-0">
        <p className="mb-1 flex items-center gap-1.5 font-medium"><T k="estimatedSpeed" /><SpeedInfo /></p>
        <p className="text-sm tabular-nums">{localizeModelText(performanceRangeSpeedLabel(serving.assessment.performance, serving.assessment.profile.contextLength))}</p>
        <p className="mt-1 text-xs text-slate-500">{tr("speedEstimateNote")}</p>
      </div>}
      </div>
      {radar && <ModelRadar model={model} />}
    </div>
  )
  if (open !== undefined) return open ? <div id={contentId}>{content}</div> : null
  return <details className="group mt-4 text-sm">
    <summary className="flex cursor-pointer list-none items-center justify-end gap-1 rounded py-1 text-slate-600 focus-visible:outline-2 focus-visible:outline-blue-500 dark:text-slate-300 [&::-webkit-details-marker]:hidden"><T k="modelDetails" /><CaretDownIcon aria-hidden="true" className="size-4 group-open:rotate-180" /></summary>
    {content}
  </details>
}
/** A catalog model's download and the one-time optimization that follows it. */
const acquiring = (acquisition: CatalogLocalModel["acquisitionState"]) =>
  acquisition._tag === "Installing" || acquisition._tag === "Updating" || acquisition._tag === "Optimizing"
const minutesRemaining = (seconds: number) => seconds < 60 ? tr("lessThanMinute") : `${tr("aboutMinutes")} ${Math.ceil(seconds / 60)} ${tr("minutesShort")}`
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
  return elapsed < 5 || done <= 0 ? tr("timeEstimate") : minutesRemaining((progress.total - progress.completed) * elapsed / done)
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
  const stages = { queued: tr("queued"), resolving: tr("preparingDownload"), checking_space: tr("checkingSpace"), downloading: tr("downloading"), verifying: tr("verifyingDownload"), publishing: tr("finishingDownload") }
  const bar = "absolute inset-y-0 left-0 rounded-md bg-blue-700 transition-[width,opacity] duration-500 ease-out dark:bg-blue-500"
  let heading: string, title: ReactNode, detail: string, percent: number | null, downloadWidth: number, pulse: boolean, stat: { label: string; value: string }, remaining: string, cancelLabel: string
  if (acquisition._tag === "Optimizing") {
    const { stage, completed, total } = acquisition.progress
    heading = localizeModelText(describeModelOptimization(acquisition.progress, Result.isSuccess(hardware) ? Option.some(hardware.value) : Option.none()))
    title = <span className="min-w-0 flex-1 truncate">{heading}</span>
    detail = tr("oneTimeSetup")
    percent = stage === "tuning" && total > 0 ? Math.min(100, completed * 100 / total) : null
    downloadWidth = 100
    pulse = percent === null
    stat = { label: tr("model"), value: modelName }
    remaining = tuningEstimate
    cancelLabel = tr("skipOptimization")
  } else {
    const { progress } = acquisition
    const downloading = progress.stage === "downloading"
    const rate = downloading ? Option.getOrNull(progress.bytesPerSecond) : null
    heading = stages[progress.stage]
    title = downloading ? <><span className="shrink-0"><T k="downloading" /></span><span className="min-w-0 flex-1 truncate" title={modelName}>{modelName}</span></> : <span className="min-w-0 flex-1 truncate">{heading}</span>
    detail = `${formatStorageSize(progress.completedBytes)} / ${progress.totalBytes > 0 ? formatStorageSize(progress.totalBytes) : tr("unknownTotal")}`
    percent = progress.totalBytes > 0 ? progress.completedBytes / progress.totalBytes * 100 : null
    downloadWidth = percent ?? 100
    pulse = percent === null
    stat = { label: tr("downloadSpeed"), value: rate !== null ? formatTransferRate(rate) : "—" }
    remaining = downloading && rate !== null && rate > 0 && progress.totalBytes > 0
      ? minutesRemaining((progress.totalBytes - progress.completedBytes) / rate)
      : downloading ? tr("timeEstimate") : "—"
    cancelLabel = tr("cancelDownload")
  }
  const tuning = optimization !== null && percent !== null
  const phase = acquisition._tag === "Optimizing" ? "optimizing" : "download"
  const fade = "motion-safe:animate-[fade-in_400ms_ease-out]"
  const spinner = <CircleNotchIcon aria-hidden="true" className="size-3.5 shrink-0 text-blue-700 motion-safe:animate-spin dark:text-blue-400" />
  const percentText = percent !== null ? `${Math.floor(percent)}%` : "—"
  const cancelButton = (className: string, size?: "sm") => onCancel && <Button variant="ghost" size={size} className={`hover:bg-transparent hover:text-red-600 dark:hover:bg-transparent dark:hover:text-red-400 ${className}`} disabled={pending} onClick={onCancel}><XIcon />{cancelLabel}</Button>
  const progressBar = <div role="progressbar" aria-label={optimization ? tr("optimizationProgress") : tr("downloadProgress")} aria-valuemin={0} aria-valuemax={100} aria-valuenow={percent ?? undefined} aria-valuetext={detail} className="relative h-2 w-full overflow-hidden rounded-md bg-slate-100 dark:bg-slate-800">
    <div className={`${bar} ${pulse ? "motion-safe:animate-pulse" : ""}`} style={{ width: `${downloadWidth}%`, opacity: tuning ? 0 : 1 }} />
    <div className={bar} style={{ width: `${tuning ? percent : 0}%`, opacity: tuning ? 1 : 0 }} />
  </div>
  // A model row already names the model: one status line over the bar, one line of detail under it.
  if (layout === "row") {
    const facts = [optimization === null ? stat.value : null, remaining].filter(fact => fact !== null && fact !== "—").join(" · ")
    return <div className="w-full min-w-0" aria-label={tr("modelDownload")}>
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
  return <div className="w-full min-w-0" aria-label={tr("modelDownload")}>
    <h3 className="mb-6 flex items-center gap-2 text-sm font-medium">
      <span key={phase} className={`flex min-w-0 flex-1 items-center gap-2 ${fade}`}>{title}</span>
      {spinner}
    </h3>
    {progressBar}
    <div className="mt-3 flex items-baseline justify-between gap-3 text-sm tabular-nums text-slate-600 dark:text-slate-300"><span className="min-w-0 truncate">{detail}</span><span>{percentText}</span></div>
    <dl className="mt-6 grid grid-cols-2 gap-4 text-sm"><div className="min-w-0"><dt className="text-xs text-slate-500 dark:text-slate-400">{stat.label}</dt><dd className="mt-1 truncate font-medium tabular-nums text-slate-800 dark:text-slate-200" title={stat.value}>{stat.value}</dd></div><div className="text-right"><dt className="text-xs text-slate-500 dark:text-slate-400"><T k="timeRemaining" /></dt><dd className="mt-1 font-medium tabular-nums text-slate-800 dark:text-slate-200">{remaining}</dd></div></dl>
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
    <Button disabled={command.pending || transferring || !fits} onClick={() => localModelIsInstalled(target) ? load(target.modelId) : install(target.modelId)}>{tr("switchToModel").replace("{name}", formatLocalModelDisplayName(target))}</Button>
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
    <ErrorNotice severity="info" title={Option.isSome(replacement) ? tr("replacementModelAvailable") : tr("modelNoLongerSupported")} description={Option.isSome(replacement) ? tr("switchToReplacementDescription").replace("{name}", formatLocalModelDisplayName(replacement.value)) : tr("chooseAnotherFromCatalog")} className="col-span-full mt-3" />
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
  const stoppedForMemory = residency !== undefined && modelStoppedForMemory(residency)
  const canStop = residency !== undefined && ["Ready", "Loading", "Requested", "Stopping"].includes(residency._tag)
  const transferring = acquiring(acquisition)
  const fit = model.catalogData.support._tag === "Supported" ? fitNotice(model) : null
  const loadFailure = residency?._tag === "Failed" && !command.pendingOperations.includes("load") ? residency.failure : null
  const doesNotFit = loadRefusal(model) !== null
  const loadNotice = inlineLoadFailure && !doesNotFit ? loadFailure : null
  const downloadFailure = (acquisition._tag === "InstallFailed" || acquisition._tag === "UpdateFailed") && !command.pendingOperations.includes("install") ? acquisition.failure : null
  const canDownload = model.catalogData.support._tag === "Supported" && model.servingState._tag === "Assessed" && model.servingState.assessment._tag === "Fits"
  const [confirmingLoad, setConfirmingLoad] = useState(false)
  const requestLoad = () => { if (replacing) setConfirmingLoad(true); else load(model.modelId) }

  return <TooltipProvider><div className="contents">
    <div className="flex flex-wrap items-center justify-end gap-2">{children}
      {transferring ? null : !installed ? downloadFailure ? null : <Button disabled={pending || model.catalogData.support._tag !== "Supported" || model.servingState._tag !== "Assessed" || model.servingState.assessment._tag !== "Fits"} onClick={() => { install(model.modelId) }}><DownloadSimpleIcon />{tr("downloadWithSize").replace("{size}", formatStorageSize(model.storageBytes).replace(/\s/g, ""))}</Button> : <>
        {model.catalogData.support._tag === "Supported" && (onConnectAgent ? <Button className="min-w-28" disabled={pending} onClick={onConnectAgent}><PlugIcon /><T k="connectAgent" /></Button> : canStop ? <Button className="min-w-28" variant="outline" disabled={stopping.pending} onClick={() => stop()}><SquareIcon /><T k="stopModel" /></Button> : loadNotice ? null : <LoadModelButton model={model} pending={pending} onLoad={requestLoad} />)}
        {!onConnectAgent && <RemoveDownloadButton model={model} running={canStop} disabled={pending} onRemove={() => remove(model.modelId)} />}
        {model.catalogData.support._tag === "Supported" && acquisition._tag === "UpdateAvailable" && <ActionTooltip label={tr("newFilesAvailable")} trigger={<Button variant="outline" disabled={pending} onClick={() => install(model.modelId)}><T k="update" /></Button>} />}
      </>}

    </div>
    {transferring && <div className="col-span-full mt-1"><DownloadProgress layout="row" modelName={formatLocalModelDisplayName(model)} acquisition={acquisition} pending={command.pending} onCancel={() => cancel(model.modelId)} /></div>}
    {fit !== null && (doesNotFit || !loadFailure) && !downloadFailure && <ErrorNotice severity="info" title={fit} className="col-span-full mt-3" />}
    {stoppedForMemory && !command.pendingOperations.includes("load") && <ErrorNotice severity="warning" title={tr("modelStoppedForMemory")} description={tr("modelStoppedForApps")} className="col-span-full mt-3" />}
    {model.catalogData.support._tag === "Disabled" && <ErrorNotice severity="warning" title={tr("modelUnavailable")} description={tr("chooseAnotherFromCatalog")} className="col-span-full mt-3" />}
    {downloadFailure && <ErrorNotice {...downloadNotice(downloadFailure)} className="col-span-full mt-3" actions={<>
      {canDownload && <NoticeAction disabled={pending} onClick={() => install(model.modelId)}><T k="retryDownload" /></NoticeAction>}
      <NoticeAction disabled={pending} onClick={() => dismiss(model.modelId)}><T k="dismiss" /></NoticeAction>
    </>} />}
    {acquisition._tag === "RemoveFailed" && <ErrorNotice {...modelRemovalNotice(acquisition.failure)} className="col-span-full mt-3" />}
    {loadNotice && <div className="col-span-full mt-3"><ModelLoadNotice failure={loadNotice} actions={model.catalogData.support._tag === "Supported" && loadNotice.retryable && !onConnectAgent ? <NoticeAction disabled={pending} onClick={requestLoad}><T k="loadAgain" /></NoticeAction> : undefined} /></div>}
    {command.failures.map(failure => <ErrorNotice key={failure.operation} {...modelCommandNotice(failure)} className="col-span-full mt-3" />)}
    {replacing && <ConfirmDialog open={confirmingLoad} onOpenChange={setConfirmingLoad} title={tr("loadConfirmationTitle").replace("{name}", formatLocalModelDisplayName(model))}
      description={tr("loadConfirmationDescription").replace("{name}", formatLocalModelDisplayName(model)).replace("{replacing}", replacing)} confirmLabel={tr("loadModel")} onConfirm={() => load(model.modelId)} />}
  </div></TooltipProvider>
}
/** Removing a download always asks first; a running model is stopped before its files are removed. */
function RemoveDownloadButton({ model, running, disabled, onRemove }: { model: CatalogLocalModel; running: boolean; disabled: boolean; onRemove: () => void }) {
  const [confirming, setConfirming] = useState(false)
  const name = formatLocalModelDisplayName(model)
  return <>
    <Button variant="ghost" size="icon" aria-label={`Remove ${name}`} title={tr("deleteDownload")} disabled={disabled} onClick={() => setConfirming(true)}><TrashIcon /></Button>
    <ConfirmDialog open={confirming} onOpenChange={setConfirming} destructive
      title={running ? tr("stopAndRemoveFilesQuestion").replace("{name}", name) : tr("removeFilesQuestion").replace("{name}", name)}
      description={tr("downloadAgainLater")} confirmLabel={running ? tr("stopAndRemove") : tr("remove")} onConfirm={onRemove} />
  </>
}
function ModelCard({ model, models, showMemory = false, replacing, hardware }: { model: CatalogLocalModel; models: readonly CatalogLocalModel[]; showMemory?: boolean; replacing?: string; hardware: Option.Option<LocalInferenceHardware> }) {
  const [detailsOpen, setDetailsOpen] = useState(false)
  const detailsId = useId()
  const deprecation = localModelDeprecation(model)
  const detailsToggle = <Button variant="ghost" aria-expanded={detailsOpen} aria-controls={detailsId} onClick={() => setDetailsOpen(value => !value)}><T k="details" /><CaretDownIcon aria-hidden="true" className={`size-4 ${detailsOpen ? "rotate-180" : ""}`} /></Button>
  const acquisition = model.acquisitionState
  const residency = "residencyState" in acquisition ? acquisition.residencyState : undefined
  const statusLabel = acquisition._tag === "Removing" ? tr("removing") : acquisition._tag === "RemoveFailed" ? tr("removalFailed") : residency?._tag === "Ready" ? tr("loaded") : residency?._tag === "Unloaded" || residency?._tag === "Stopped" ? tr("downloaded") : residency?._tag === "Failed" ? tr("notLoaded") : residency?._tag === "Requested" ? tr("preparingEllipsis") : residency?._tag === "Loading" ? localizeModelText(describeModelLoadStage(residency.stage, residency.plannedAllocation, hardware)) : residency?._tag ?? (acquisition._tag === "NotInstalled" ? "" : acquisition._tag === "InstallFailed" ? tr("notDownloaded") : acquisition._tag === "UpdateFailed" ? tr("updateIncomplete") : acquisition._tag === "UpdateAvailable" ? tr("updateAvailable") : acquisition._tag)
  const status = (statusLabel || showMemory) && <div className="mt-1 flex flex-wrap items-center gap-x-3 text-sm text-slate-500">{statusLabel && <span className={residency?._tag === "Ready" ? "text-green-600 dark:text-green-400" : ""}>{statusLabel}</span>}{showMemory && model.servingState._tag === "Assessed" && model.servingState.assessment._tag === "Fits" && <><span aria-hidden="true">·</span><span>{formatMemorySize(model.servingState.assessment.memory.totalRequiredBytes)} {tr("memory")}</span></>}</div>
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
  return <div className={`relative ${pageLayout.recommendationPane}`} aria-label={tr("selectedModelProfile")}>
    <div className={transferring ? "invisible" : undefined} inert={transferring} aria-hidden={transferring}>
    <div className={pageLayout.recommendationToolbar}>
    <div className="flex items-center gap-1" aria-label={tr("modelInformation")}>
      <Button variant={view === "profile" ? "secondary" : "ghost"} aria-pressed={view === "profile"} onClick={() => setView("profile")}><T k="profile" /></Button>
      <Button variant={view === "details" ? "secondary" : "ghost"} aria-pressed={view === "details"} onClick={() => setView("details")}><T k="details" /></Button>
    </div>
      {transferring ? <Button disabled><DownloadSimpleIcon />Download ({formatStorageSize(model.storageBytes).replace(/\s/g, "")})</Button> : <ModelControls model={model} inlineLoadFailure={false} onConnectAgent={() => connectAgent()} {...(Option.isSome(active) && active.value.model.modelId !== model.modelId ? { replacing: formatLocalModelDisplayName(active.value.model) } : {})} />}
    </div>
    <div className="grid min-h-72">
      <div className={`col-start-1 row-start-1 min-w-0 ${view === "profile" ? "" : "invisible"}`} aria-hidden={view !== "profile"}><ModelRadar model={model} /></div>
      <div className={`col-start-1 row-start-1 min-w-0 ${view === "details" ? "" : "invisible"}`} aria-hidden={view !== "details"}><ModelDetails model={model} compact open /></div>
    </div>
    </div>
    {transferring && <div className="absolute inset-5 flex items-center justify-center overflow-y-auto" aria-label={tr("downloadPanel")}>
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
  return <section aria-label={tr("topRecommendations")} className="mb-8">
    <TooltipProvider><div className={pageLayout.recommendations}>
      <div className={pageLayout.recommendationList} aria-label={tr("recommendedModels")}>{models.map((model, rank) => <button key={model.modelId} type="button" aria-pressed={model.modelId === selected.modelId} onClick={() => setSelection({ preference, modelId: model.modelId })} className={`${pageLayout.recommendationRow} focus-visible:outline-2 focus-visible:outline-blue-500 ${model.modelId === selected.modelId ? "border-blue-300 bg-blue-50 dark:border-blue-700 dark:bg-slate-800" : "border-transparent hover:bg-slate-100 dark:hover:bg-slate-800"}`}>
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
  return <span className="flex items-center gap-2">{lab ? <LabLogo lab={lab} className="size-4" /> : <BuildingsIcon aria-hidden="true" className="size-4 text-slate-500" />}{lab ? lab.name : tr("anyLab")}</span>
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
    ? [{ value: "all", label: "All models" }, { value: "downloaded", label: tr("downloaded") }, { value: "downloading", label: "Downloading" }]
    : [{ value: "all", label: "All models" }, { value: "fits", label: tr("fitsMyMachine") }]
  const sortOptions = [
    ...(!installedOnly ? [{ value: "recommended", label: tr("recommended") }] : []),
    { value: "name", label: tr("nameAZ") }, { value: "smallest", label: tr("smallestDownload") }, { value: "largest", label: tr("largestDownload") },
  ]
  const [filter, setFilter] = useState("all")
  const [lab, setLab] = useState<ModelLab | null>(null)
  const [sort, setSort] = useState(installedOnly ? "name" : "recommended")
  const client = useAgentClient()
  const session = useSession()
  const preference = useAtomValue(session.rankingPreference)
  const setPreference = useAtomSet(useMemo(() => Atom.fn((index: number) => session.setRankingPreference(index)), [session]))
  if (Result.isFailure(catalog)) return <>{!discover && <h1 className={pageLayout.pageTitle}>{localizedPageName(page, useI18n().t)}</h1>}<ErrorNotice title={tr("couldNotLoadCatalog")} description={tr("modelInfoUnavailable")} className="mt-5" /></>
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
        <h1 className={pageLayout.pageTitle}>{localizedPageName(page, t)}</h1>
        <span className="text-sm tabular-nums text-slate-500" role="status">{visible.length} {visible.length === 1 ? "model" : "models"}</span>
      </div>
      <div className={pageLayout.catalogToolbar}>
        <div className="flex flex-wrap items-center gap-2">
          <Select items={filterOptions} value={filter} onValueChange={value => { if (value !== null) setFilter(value) }}>
            <SelectTrigger aria-label={tr("filterModels")}><SelectValue /></SelectTrigger>
            <SelectContent>{filterOptions.map(option => <SelectItem key={option.value} value={option.value}>{option.label}</SelectItem>)}</SelectContent>
          </Select>
          <Select value={lab} onValueChange={setLab}>
            <SelectTrigger aria-label={tr("filterByLab")}><SelectValue>{(value: ModelLab | null) => <LabOption lab={value} />}</SelectValue></SelectTrigger>
            <SelectContent>{[<SelectItem key="any" value={null}><LabOption lab={null} /></SelectItem>, ...labOptions.map(option => <SelectItem key={option.name} value={option}><LabOption lab={option} /></SelectItem>)]}</SelectContent>
          </Select>
          <Select items={sortOptions} value={sort} onValueChange={value => { if (value !== null) setSort(value) }}>
            <SelectTrigger aria-label={tr("sortModels")}><span className="text-slate-500"><T k="sort" /></span><SelectValue /></SelectTrigger>
            <SelectContent>{sortOptions.map(option => <SelectItem key={option.value} value={option.value}>{option.label}</SelectItem>)}</SelectContent>
          </Select>
        </div>
        <Input aria-label={tr("searchModels")} placeholder={tr("searchModelsPlaceholder")} className={pageLayout.modelSearch} value={search} onChange={event => setSearch(event.target.value)} />
      </div>
    </>}
    {discover && <HardwareOverview /> }
    {Option.isSome(stopResult.failure) && <ErrorNotice title={tr("couldNotStopModel")} description={stopResult.failure.value} className="mt-5" />}
    {discover && <RecommendationPreference value={preference} onChange={setPreference} />}
    {!discover && assessment && !assessment.complete && <p className="mb-4 text-sm text-slate-500">Assessing models · {assessment.settledModels} of {assessment.totalModels}</p>}
    {discover && (recommendationsPending
      ? <RecommendationsSkeleton assessment={assessment} waitingForHardware={Result.isInitial(hardware)} />
      : <Recommendations preference={preference} models={featuredCatalogModels(ranked, 5)} active={Option.fromNullable(active)} />)}
    {!discover && <>
    <div className="grid items-start gap-5">{visible.map(model => <ModelCard key={model.modelId} model={model} models={models} showMemory={installedOnly} hardware={Result.value(hardware)} {...(active && active.model.modelId !== model.modelId ? { replacing: formatLocalModelDisplayName(active.model) } : {})} />)}</div>
    {visible.length === 0 && <p className="py-8 text-slate-500">{search.trim() || filter !== "all" || lab !== null ? "No models match your search or filter." : installedOnly ? "No models downloaded yet. Find one in Discover." : "{tr("noModelsMatch")}"}</p>}
    </>}
    {discover && ranked.length === 0 && !recommendationsPending && Result.isSuccess(hardware) && <p className="py-8 text-slate-500"><T k="noRecommendations" /></p>}
  </>
}
function Connections({ serviceReady, selectedModel }: { serviceReady: boolean; selectedModel: Option.Option<ProviderModelId> }) {
  const service = useSession()
  const models = useLocalModels()
  const canConnect = serviceReady && Result.isSuccess(models) && models.value.models.some(model => Option.isSome(localModelProviderModelId(model)))
  const navigate = useNavigate()
  const hardware = useLocalInferenceHardware()
  const state = useAtomValue(service.application)
  // The desktop app, its service, and the agents always share one computer and user, so it
  // configures agents in place. A browser can't know where its viewer's agents run.
  const desktop = Option.isSome(service.clientWindow)
  // A desktop window serves its own loopback endpoint; a browser reaches the service at its own origin.
  const apiOrigin = desktop
    ? (Result.isSuccess(state) ? Option.some(`http://127.0.0.1:${new URL(state.value.endpoint).port}`) : Option.none<string>())
    : Option.some(window.location.origin)
  const platform = useServerPlatform()
  const { remote } = useRemoteAccess()
  const client = useAgentClient()
  const network = useAtomValue(client.Configuration.GetNetworkAccess({})).result
  const available = Result.isSuccess(models) ? models.value.models.filter(model => Option.isSome(localModelProviderModelId(model))) : []
  const active = Result.isSuccess(models) ? Option.getOrUndefined(activeLocalModel(models.value))?.model.modelId : undefined
  const ranked = Result.isSuccess(hardware) ? rankedLocalModelOptions(available.map(model => ({ id: model.modelId, kind: "stored" as const, model })), { fastToSmart: 0.5, memoryBudgetBytes: targetPhysicalMemoryBytes(hardware.value) }, available.length).map(option => option.model) : available
  const commandModels = ranked.map(model => ({ id: model.modelId, label: formatLocalModelDisplayName(model) }))
  const defaultModel = commandModels.find(model => model.id === active)?.id ?? commandModels[0]?.id
  return <>
    {!serviceReady && <p className="mt-5 text-sm text-slate-500">{desktop ? tr("configurationChecksAvailable") : tr("startServiceBeforeAgent")}</p>}
    {serviceReady && !canConnect && !Result.isInitial(models) && <ErrorNotice severity={Result.isFailure(models) ? "error" : "info"} title={Result.isFailure(models) ? "Couldn’t check available models" : tr("downloadModelToConnect")} description={Result.isFailure(models) ? tr("checkServiceStatus") : tr("chooseCompatibleModel")} className="mt-5" actions={!Result.isFailure(models) && <NoticeAction onClick={() => navigate("discover")}><T k="discoverModels" /></NoticeAction>} />}
    {desktop
      ? <DesktopHarnessConnections canConnect={canConnect} models={commandModels} defaultModel={defaultModel} platform={platform} selectedModel={selectedModel} />
      : Option.isSome(apiOrigin) && <HarnessSetupList defaultModel={defaultModel} origin={apiOrigin.value} remote={remote} platform={viewerPlatform()} />}
    {Option.isSome(apiOrigin) && <OtherApps origin={apiOrigin.value} model={defaultModel} platform={remote ? viewerPlatform() : platform} remote={remote} networkAccess={Result.isSuccess(network) && network.value.enabled} onOpenSettings={() => navigate("settings")} />}
  </>
}
/** One-click connections, for the desktop app only: it runs where the agents run. */
function DesktopHarnessConnections({ canConnect, models, defaultModel, platform, selectedModel }: { canConnect: boolean; models: readonly CommandModel[]; defaultModel: ProviderModelId | undefined; platform: string; selectedModel: Option.Option<ProviderModelId> }) {
  const client = useAgentClient()
  const rows = useAtomValue(client.Connections.WatchHarnessConnections({})).result
  const connect = useAtomSet(client.Connections.ConnectHarness)
  const disconnect = useAtomSet(client.Connections.DisconnectHarness)
  const connecting = useAtomValue(client.Connections.ConnectHarness)
  const disconnecting = useAtomValue(client.Connections.DisconnectHarness)
  const busy = connecting.waiting || disconnecting.waiting
  const error = !busy && firstFailure([connecting, disconnecting])
  return <>
    {error && Result.isFailure(error) && <ErrorNotice title={Result.isFailure(disconnecting) ? tr("couldNotDisconnectAgent") : tr("couldNotConnectAgent")} description={tr("checkAgentConfig")} className="mt-5" />}
    {Result.isFailure(rows) ? <ErrorNotice title={tr("couldNotCheckConnections")} description={tr("connectionStatusUnavailable")} className="mt-5" />
      : !Result.isSuccess(rows) ? <ConnectionsSkeleton />
      : rows.value._tag === "Unavailable" ? <ErrorNotice title={tr("couldNotCheckConnections")} description={tr("cannotReadConnections")} className="mt-5" />
      : <HarnessConnections connections={rows.value.connections} busy={busy} canConnect={canConnect} models={models} defaultModel={defaultModel} platform={platform}
          onConnect={harness => connect({ harness, model: selectedModel, installSkill: true })} onDisconnect={harness => disconnect({ harness })} />}
  </>
}
/** The model row's status: the load stage in full while loading, the memory in use once loaded. */
const modelStatusText = (residency: ModelResidency, hardware: Option.Option<LocalInferenceHardware>): string => {
  switch (residency._tag) {
    case "Requested": return tr("preparingEllipsis")
    case "Loading": return describeModelLoadStage(residency.stage, residency.plannedAllocation, hardware)
    case "Ready": return `${tr("loaded")} · ${formatModelMemory(residency.allocation)}`
    case "Stopping": return tr("stoppingEllipsis")
    case "Unloaded":
    case "Stopped":
    case "Failed": return tr("notLoaded")
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
  if (Result.isInitial(models)) return <LoadingRegion label={tr("loadingModelStatus")} className="mt-5"><div className="flex h-12 items-center gap-3"><SkeletonLine className="h-6 w-64" /></div></LoadingRegion>
  if (Result.isSuccess(models) && !active) return <div className="mt-5 flex min-h-12 items-center"><p className="m-0 text-base text-slate-500 dark:text-slate-400"><T k="noModelLoaded" /></p></div>
  return <div className="mt-5">
    <div className="flex min-h-12 items-center justify-between gap-5">
      <div className="flex min-w-0 items-center gap-3">
        {active ? <ModelLogo model={active.model} className="size-6 shrink-0" /> : <CubeIcon aria-hidden="true" className="size-5 shrink-0 text-slate-400 dark:text-slate-500" />}
        <div className="flex min-w-0 items-baseline gap-2">
          <p title={active ? formatLocalModelDisplayName(active.model) : undefined} className={`m-0 min-w-0 truncate ${active ? "text-base font-medium text-slate-800 dark:text-slate-200" : "text-sm text-slate-500 dark:text-slate-400"}`}>{active ? formatLocalModelDisplayName(active.model) : presentation?.label ?? (Result.isFailure(models) ? tr("modelStatusUnavailable") : tr("readingModelStatus"))}</p>
          {active && <><span aria-hidden="true" className="text-slate-400 dark:text-slate-500">·</span><span className={`shrink-0 text-xs ${active.residency._tag === "Ready" ? "text-green-700 dark:text-green-400" : "text-slate-500 dark:text-slate-400"}`}>{modelStatusText(active.residency, Result.isSuccess(hardware) ? Option.some(hardware.value) : Option.none())}</span></>}
        </div>
      </div>
      {presentation?.canStop && <Button variant="outline" className="hover:border-red-300 hover:text-red-600 dark:hover:border-red-800 dark:hover:text-red-400" disabled={stopping.pending} onClick={() => stop()}><SquareIcon /><T k="stopModel" /></Button>}
    </div>
    {active && Option.match(loadFraction(active.residency), {
      onNone: () => null,
      onSome: fraction => <div className="mt-4 flex items-center gap-3">
        <Progress aria-label={tr("modelLoadingProgress")} className="flex-1" indicatorClassName="bg-blue-700 dark:bg-blue-500" value={fraction * 100} />
        <span className="w-9 shrink-0 text-right text-xs tabular-nums text-slate-500 dark:text-slate-400">{formatModelLoadPercentage(fraction)}</span>
      </div>,
    })}
    {Result.isFailure(models) && <ErrorNotice title={tr("couldNotReadModelStatus")} description={tr("cannotConfirmModelRunning")} className="mt-2" />}
    {Option.isSome(stopping.failure) && <ErrorNotice title={tr("couldNotStopModel")} description={stopping.failure.value} className="mt-2" />}
  </div>
}
function DownloadActivity() {
  const models = useLocalModels()
  const active = Result.isSuccess(models) ? models.value.models.filter((model): model is CatalogLocalModel => model._tag === "Catalog" && (acquiring(model.acquisitionState) || model.acquisitionState._tag === "Removing")) : []
  if (Result.isInitial(models)) return null
  if (Result.isSuccess(models) && active.length === 0) return null
  return <div className="mt-5 border-t border-slate-200 pt-5 dark:border-slate-700">
    {!Result.isSuccess(models) ? <p className="mt-3 text-sm text-slate-500">{Result.isFailure(models) ? tr("downloadActivityUnavailable") : tr("readingDownloadActivity")}</p> : <ul className="space-y-5">{active.map(model => <li key={model.modelId}><div className="flex items-center gap-3"><ModelLogo model={model} className="size-6" /><p className="text-sm">{formatLocalModelDisplayName(model)} · {model.acquisitionState._tag === "Removing" ? tr("removingFiles") : model.acquisitionState._tag === "Optimizing" ? tr("optimizing") : model.acquisitionState._tag === "Updating" ? tr("update") : tr("downloading")}</p></div><DownloadProgress modelName={formatLocalModelDisplayName(model)} acquisition={model.acquisitionState} /></li>)}</ul>}
  </div>
}
function Status({ observation }: { observation: ServiceObservation }) {
  const session = useSession()
  const retry = useAtomSet(session.retryService)
  const retrying = useAtomValue(session.retryService)
  const snapshot = Result.isSuccess(observation.service) ? observation.service.value : null
  const service = snapshot ?? undefined
  const tray = Option.getOrUndefined(observation.tray)
  const ready=service?._tag === "Ready"
  return <div className={pageLayout.statusStack}>
    <section aria-busy={!snapshot} aria-label={!snapshot ? tr("loadingServiceStatus") : undefined} className={pageLayout.statusHero}>
      <div className="flex items-center justify-between gap-4 border-b border-slate-200 pb-4 dark:border-slate-700">
        <h2 className="m-0 text-base font-medium text-slate-600 dark:text-slate-300"><T k="magnitudeService" /></h2>
        <div className={`flex h-8 items-center gap-2 rounded-full px-3 text-sm font-medium ${ready ? "bg-green-200/20 text-green-700 dark:bg-green-800/20 dark:text-green-400" : "text-slate-500 dark:text-slate-400"}`}>
          {ready ? <CheckCircleIcon aria-label={tr("serviceReady")} weight="fill" className="size-5 shrink-0" /> : <PulseIcon className="size-4 shrink-0" />}
          <span>{!snapshot ? <SkeletonLine className="h-4 w-16 text-xs" /> : ready ? tr("ready") : service?._tag === "CleanupFailed" ? tr("cleanupAttention") : service?._tag === "Failed" ? tr("unavailable") : tr("starting")}</span>
        </div>
      </div>
      {ready ? <ModelStatus /> : !snapshot ? <SkeletonLine className="mt-5 h-12 w-64" /> : service?._tag === "Failed" || service?._tag === "CleanupFailed" ? null : <p className="mt-5 text-sm text-slate-500"><T k="startingService" /></p>}
      {ready && <DownloadActivity />}
      {service && "message" in service && <ErrorNotice className="mt-5" title={service._tag === "CleanupFailed" ? tr("couldNotConfirmServiceStopped") : tr("serviceCouldNotStart")}
        description={service._tag === "CleanupFailed" ? tr("backgroundWorkMayRun") : translateKnownMessage(service.message)}
        actions={service._tag === "Failed" && observation.canRetry ? <NoticeAction disabled={retrying.waiting} onClick={() => retry()}><T k="retryService" /></NoticeAction> : undefined} />}
      {Result.isFailure(retrying) && !retrying.waiting && service?._tag !== "Failed" && service?._tag !== "Ready" && <ErrorNotice title={tr("couldNotRetryService")} className="mt-3" />}
    </section>
    <MemoryBreakdown />
    {ready && <StatusOverview />}
    {tray?._tag === "Unavailable" && <ErrorNotice severity="warning" title={tr("trayIconUnavailable")} description={tr("windowKeepsRunning")} />}
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
  const usage = useAtomValue(client.Models.GetServingUsage({ period: tr("today"), timeZone: Intl.DateTimeFormat().resolvedOptions().timeZone, model: Option.none() })).result
  const rows = useAtomValue(client.Connections.WatchHarnessConnections({})).result
  const network = useAtomValue(client.Configuration.GetNetworkAccess({})).result
  const today = Result.isSuccess(usage) && usage.value._tag === "Available" ? usage.value : null
  const installed = Result.isSuccess(rows) && rows.value._tag === "Ready" ? rows.value.connections.filter(row => row.installed) : null
  const connected = installed?.filter(row => row.inspection._tag === "Connected").length
  const address = Result.isSuccess(network) && network.value.enabled ? Option.getOrElse(network.value.bind, () => tr("allInterfaces")) : null
  const loading = <SkeletonLine className="h-4 text-xs" width="80px" />
  return <section aria-label={tr("activity")} className={`${pageLayout.card} grid grid-cols-1 gap-2 p-3 md:grid-cols-3`}>
    <StatusTile label={tr("today")} onClick={() => navigate("usage")}
      value={today ? `${compact.format(today.totalTokens)} tokens` : Result.isInitial(usage) ? loading : "Unavailable"}
      detail={today ? `${today.requests.toLocaleString()} ${today.requests === 1 ? "request" : "requests"}` : null} />
    <StatusTile label={tr("agents")} onClick={() => navigate("connections")}
      value={connected !== undefined ? `${connected} connected` : Result.isInitial(rows) ? loading : "Unavailable"}
      detail={installed ? `${installed.length} installed` : null} />
    <StatusTile label={tr("networkAccess")} onClick={() => navigate("settings")}
      value={Result.isSuccess(network) ? network.value.enabled ? "On" : "Off" : Result.isInitial(network) ? loading : "Unavailable"}
      detail={Result.isSuccess(network) ? address ?? tr("thisComputerOnly") : null} />
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
  return <div className={nested ? "bg-slate-50 py-2.5 pl-6 pr-4 dark:bg-slate-900/40 md:pl-10" : "px-4 py-3"}>
    <div className="flex flex-col items-start gap-2 md:flex-row md:items-center md:justify-between md:gap-6">
      <div className="min-w-0"><p className={nested ? "text-[13px] font-medium" : "text-sm font-medium"}>{label}</p>
        {hint && <div className="mt-0.5 text-xs text-slate-500">{hint}</div>}

      </div>
      {control && <div className="flex shrink-0 flex-wrap items-center gap-2 md:flex-nowrap">{control}</div>}
    </div>
    {alert && <div className="mt-2">{alert}</div>}
    {children}
  </div>
}
function ThemeRow() {
  const { t } = useI18n()
  const appearance = useAppearancePreference()
  const session = useSession()
  const saveAppearance = useMemo(() => Atom.fn((preference: AppearancePreference) => session.saveAppearance(preference).pipe(
    Effect.tap(() => Effect.sync(() => setAppearancePreference(preference))))), [session])
  const save = useAtomSet(saveAppearance)
  const saving = useAtomValue(saveAppearance)
  const readError = useAtomValue(appearanceReadError)
  return <SettingsRow label={t("theme")} alert={!saving.waiting && (Result.isFailure(saving) ? <ErrorNotice title={tr("themeNotSaved")} description={tr("previousAppearanceInUse")} /> : readError ? <ErrorNotice title={tr("couldNotReadTheme")} description={tr("systemAppearanceUsed")} /> : undefined)} control={
    <div className="inline-flex rounded-md border border-slate-300 p-0.5 dark:border-slate-700" role="group" aria-label={useI18n().t("theme")}>
      {(["system", "light", "dark"] as const).map(value => { const Icon = value === "system" ? MonitorIcon : value === "light" ? SunIcon : MoonIcon
        return <Button key={value} size="sm" variant={appearance === value ? "secondary" : "ghost"} aria-pressed={appearance === value} disabled={saving.waiting} onClick={() => save(value)}><Icon />{t(value)}</Button> })}
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
  const hint = current?._tag === "Unavailable" ? translateKnownMessage(current.message)
    : current?._tag === "RequiresApproval" ? tr("startupSettingsEnable")
    : tr("startsInBackground")
  const alert = !change.waiting && Result.isFailure(change) ? <ErrorNotice title={tr("couldNotUpdateLaunch")} description={tr("startupSettingsCheck")} /> : Result.isFailure(state) ? <ErrorNotice title={tr("couldNotCheckLaunch")} description={tr("cannotConfirmSignIn")} /> : undefined
  return <SettingsRow label={useI18n().t("launchAtLogin")} hint={Result.isInitial(state) ? <SkeletonLine className="h-4 text-xs" width="160px" /> : hint} alert={alert}
    control={current && <Switch aria-label={useI18n().t("launchAtLogin")} checked={enabled} disabled={current._tag === "Unavailable" || change.waiting} onCheckedChange={checked => set(checked)} />} />
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
  const refusal = failure ? Option.flatMap(Cause.failureOption(failure.cause), error => error instanceof ServerSettingsFailed ? Option.some(translateKnownMessage(error.message)) : Option.none()) : Option.none<string>()
  return <>
    <FolderPicker open={choosing} onOpenChange={setChoosing} initialPath={current ? Option.some(current.path) : Option.none()}
      title={tr("chooseModelsFolder")} description={tr("modelFolderDescription")} confirmLabel={tr("useThisFolder")}
      onChoose={path => save({ path: Option.some(path) })} />
    <SettingsRow label={useI18n().t("modelStorage")} alert={!busy && failure ? <ErrorNotice title={tr("modelFolderNotSaved")} description={Option.getOrElse(refusal, () => tr("checkFolderWritable"))} /> : Result.isFailure(settings) ? <ErrorNotice title={tr("couldNotReadFolderSetting")} description={tr("runningServiceFolderUnchanged")} /> : current && Option.isSome(current.warning) ? <ErrorNotice severity="warning" title={tr("savedFolderUnavailable")} description={tr("defaultFolderInstead")} /> : undefined}
      hint={current ? <span className="block truncate" title={current.path}>{tr("currentPathIs")} <span className="text-slate-700 dark:text-slate-300" data-testid="model-storage-path">{current.path}</span>{current.source === "Default" && ` (${tr("defaultLabel").toLowerCase()})`}</span>  : Result.isInitial(settings) ? <SkeletonLine className="h-4 text-xs" width="220px" /> : undefined}
      control={<>
        {(current?.source === "Configured" || (current && Option.isSome(current.warning))) && <Button size="sm" variant="ghost" disabled={busy} onClick={() => save({ path: Option.none() })}><T k="useDefault" /></Button>}
        <Button size="sm" variant="outline" disabled={!current || busy} onClick={() => setChoosing(true)}><FolderOpenIcon /><T k="change" /></Button>
      </>} />
  </>
}
function useRestartPending() {
  const client = useAgentClient()
  const storage = useAtomValue(client.Configuration.GetModelStorage({})).result
  const network = useAtomValue(client.Configuration.GetNetworkAccess({})).result
  const storageValue = Result.isSuccess(storage) ? storage.value : null
  return { storageValue, storagePending: storageValue !== null && storageValue.path !== storageValue.active, networkPending: Result.isSuccess(network) && network.value.pending }
}
function RestartRequiredToast() {
  const session = useSession()
  const client = useAgentClient()
  const platform = useServerPlatform()
  const { storageValue, storagePending, networkPending } = useRestartPending()
  const { supports } = useApplicationOwner()
  const restartApplication = useAtomSet(client.Application.RestartApplication)
  const disconnect = useDisconnectWarning()
  const relaunch = () => disconnect.warn("Restart", () => restartApplication({}))
  const relaunching = useAtomValue(client.Application.RestartApplication)
  if (!storagePending && !networkPending) return null
  const reason = storagePending && networkPending ? tr("previousModelNetworkSettings")
    : storagePending ? tr("magnitudeStillPreviousFolder") : tr("previousNetworkSettings")
  return <div className="fixed bottom-4 right-4 z-50 w-96 max-w-[calc(100vw-2rem)] rounded-lg bg-white shadow-md dark:bg-slate-850">
    {disconnect.dialog}
    <ErrorNotice severity={Result.isFailure(relaunching) && !relaunching.waiting ? "error" : "info"}
      title={Result.isFailure(relaunching) && !relaunching.waiting ? tr("magnitudeCouldNotRestart") : tr("restartApplyChanges")}
      description={supports("RestartService") ? reason : `${reason} Restart the Magnitude server to apply them.`}
      actions={supports("RestartService") ? <NoticeAction disabled={relaunching.waiting} onClick={() => relaunch()}><T k="restartMagnitude" /></NoticeAction> : undefined}>
      {storagePending && storageValue && <details className="mt-2 text-xs text-slate-600 dark:text-slate-400">
        <summary className="w-fit cursor-pointer rounded py-0.5 hover:underline focus-visible:outline-2 focus-visible:outline-blue-500"><T k="movingModels" /></summary>
        <p className="my-2">{tr("existingDownloadsPreviousFolder")}</p>
        <CopyCommand command={moveModelsCommand(platform, storageValue.active, storageValue.path)} label={tr("copyMoveCommand")} />
      </details>}
    </ErrorNotice>
  </div>
}

function NetworkAccessRows() {
  const client = useAgentClient()
  const settings = useAtomValue(client.Configuration.GetNetworkAccess({})).result
  const save = useAtomSet(client.Configuration.SetNetworkAccess)
  const update = (change: Partial<NetworkAccessChange>) => save({ enabled: Option.none(), bind: Option.none(), requireApiKey: Option.none(), ...change })
  const disconnect = useDisconnectWarning()
  const updating = useAtomValue(client.Configuration.SetNetworkAccess)
  const regenerate = useAtomSet(client.Configuration.RegenerateNetworkApiKey)
  const regenerating = useAtomValue(client.Configuration.RegenerateNetworkApiKey)
  const busy = updating.waiting || regenerating.waiting
  const current = Result.isSuccess(settings) ? settings.value : null
  const failure = firstFailure([updating, regenerating])
  const reachable = current?.enabled ? Option.getOrElse(current.bind, () => current.interfaces[0]?.address) : undefined
  return <>
    {disconnect.dialog}
    <SettingsRow label={useI18n().t("networkAccess")} hint={current ? <>{tr("networkInference")} <a href="https://docs.magnitude.dev/remote-server" target="_blank" rel="noreferrer" className="inline-flex items-center gap-0.5 font-medium text-slate-700 hover:underline dark:text-slate-300"><T k="remoteServerGuide" /><ArrowUpRightIcon aria-hidden="true" className="size-3" /></a></> : Result.isInitial(settings) ? <SkeletonLine className="h-4 text-xs" width="240px" /> : undefined}
      alert={!busy && failure ? <ErrorNotice title={tr("networkSettingsNotSaved")} description={tr("previousSavedSettings")} /> : Result.isFailure(settings) ? <ErrorNotice title={tr("couldNotReadNetworkSettings")} description={tr("runningServiceNetworkSettingsUnchanged")} /> : current && Option.isSome(current.warning) ? <ErrorNotice severity="warning" title={tr("savedAddressInvalid")} description={tr("chooseAddressBelow")} /> : undefined}
      control={current && <Switch aria-label={tr("networkAccess")} checked={current.enabled} disabled={busy} onCheckedChange={checked => checked ? update({ enabled: Option.some(true) }) : disconnect.warn("TurnOffNetworkAccess", () => update({ enabled: Option.some(false) }))} />} />
    {current?.enabled && <>
      <SettingsRow nested label={tr("address")} hint={current.interfaces.length === 0 ? tr("noNetworkInterfaces") : tr("networkAddressHint")}
        control={<Select items={[{ value: ALL_INTERFACES, label: tr("allInterfaces") }, ...current.interfaces.map(entry => ({ value: entry.address, label: `${entry.address} (${entry.kind === "tailscale" ? "Tailscale" : entry.name})` }))]}
          value={Option.getOrElse(current.bind, () => ALL_INTERFACES)} onValueChange={value => value === ALL_INTERFACES || value === null ? update({ bind: Option.some<NetworkBind>({ _tag: "AllInterfaces" }) }) : disconnect.warn("ChangeAddress", () => update({ bind: Option.some<NetworkBind>({ _tag: "Address", address: String(value) }) }))}>
          <SelectTrigger aria-label={tr("networkAddress")} className="min-w-56"><SelectValue /></SelectTrigger>
          <SelectContent>{[<SelectItem key={ALL_INTERFACES} value={ALL_INTERFACES}><T k="allInterfacesLabel" /></SelectItem>, ...current.interfaces.map(entry => <SelectItem key={entry.address} value={entry.address}>{entry.address} ({entry.kind === "tailscale" ? "Tailscale" : entry.name})</SelectItem>)]}</SelectContent>
        </Select>} />
      <SettingsRow nested label={tr("apiKey")} hint={current.requireApiKey ? tr("otherDevicesBearer") : tr("otherDevicesNoKey")}
        control={<><Button size="sm" variant="ghost" disabled={busy} onClick={() => disconnect.warn("RegenerateKey", () => regenerate({}))}><T k="regenerate" /></Button><span className="flex items-center gap-2 text-sm text-slate-600 dark:text-slate-400"><span aria-hidden="true"><T k="require" /></span><Switch aria-label={tr("requireApiKey")} checked={current.requireApiKey} disabled={busy} onCheckedChange={checked => checked ? update({ requireApiKey: Option.some(true) }) : disconnect.warn("StopRequiringKey", () => update({ requireApiKey: Option.some(false) }))} /></span></>}>
        {Option.isSome(current.apiKey) && current.requireApiKey && <div className="mt-2"><CopyCommand command={current.apiKey.value} label={tr("copyApiKey")} /></div>}
      </SettingsRow>
      {reachable && <SettingsRow nested label={tr("reachableAt")} hint={tr("openAiBaseUrlHint").replace("{keyHint}", current.requireApiKey ? tr("withApiKeyAbove") : "")}>
        <div className="mt-2"><CopyCommand command={inferenceUrl(reachable, current.port)} label={tr("copyBaseUrl")} /></div>
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
  if (Result.isSuccess(owner) && !supports(tr("automaticUpdates"))) return null
  const snapshot = Result.isSuccess(observation) ? observation.value : null
  const preference = snapshot?.preference
  const closed = snapshot?.transfer._tag === "Closed"
  return <SettingsRow label={useI18n().t("automaticUpdates")}
    hint={Result.isInitial(observation) ? <SkeletonLine className="h-4 text-xs" width="200px" /> : preference?._tag === "Known" ? tr("automaticUpdatesHint") : undefined}
    alert={Result.isFailure(saving) && !saving.waiting ? <ErrorNotice title={tr("updatesPreferencesNotSaved")} description={tr("previousPreferenceInUse")} /> : Result.isFailure(observation) || preference?._tag === "Unavailable" ? <ErrorNotice title={tr("couldNotReadUpdatePreferences")} description={tr("autoDownloadsUnavailable")} /> : undefined}
    control={snapshot && <Switch aria-label={useI18n().t("automaticUpdates")} checked={preference?._tag === "Known" && preference.autoDownload} disabled={preference?._tag !== "Known" || saving.waiting || closed} onCheckedChange={checked => setAutoDownload(checked)} />} />
}
const failedUpdatePhrase = { verify: tr("updateVerifyFailure"), install: tr("updateInstallFailure"), authorization: tr("updateAuthorizationFailure"), startup: tr("updateStartupFailure") } as const

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
  const disconnect = useDisconnectWarning()
  const restart = () => disconnect.warn("Restart", () => installUpdate({}))
  const checking = useAtomValue(client.Application.CheckApplicationUpdate)
  const downloading = useAtomValue(client.Application.DownloadApplicationUpdate)
  const restarting = useAtomValue(client.Application.InstallApplicationUpdate)
  const snapshot = Result.isSuccess(observation) ? observation.value : null
  const current: UpdateTransfer | undefined = snapshot?.transfer
  const pending = downloading.waiting || restarting.waiting || discarding.waiting
  const message = !current ? Result.isFailure(observation) ? tr("updateStatusUnavailable") : tr("readingUpdateStatus")
    : current._tag === "Idle" ? snapshot?.check._tag === "Succeeded" ? tr("upToDate") : tr("checksAutomatically")
    : current._tag === "Available" ? `Version ${current.version} is available · ${formatStorageSize(current.bytes)}${Option.match(current.failed, {
      onNone: () => "", onSome: failed => ` · It ${failedUpdatePhrase[failed]} on this computer, so it isn’t downloaded automatically.` })}`
    : current._tag === "Downloading" ? tr("downloadingVersion").replace("{version}", current.version).replace("{completed}", formatStorageSize(current.completed)).replace("{total}", formatStorageSize(current.total))
    : current._tag === "Cancelling" ? tr("stoppingAutomaticDownload")
    : current._tag === "Staging" ? tr("preparingVersion").replace("{version}", current.version)
    : current._tag === "Ready" ? tr("versionReadyRestart").replace("{version}", current.version)
    : current._tag === "Closed" ? tr("magnitudeQuitting") : current._tag === "Unavailable" ? tr("updatesUnavailableNow") : undefined
  const actionFailure = firstFailure([checking, downloading, restarting, discarding])
  const checkFailed = snapshot?.check._tag === "Failed"
  const installable = current?._tag === "Ready" || current?._tag === "InstallationFailed"
  const hasFailure = Boolean(actionFailure || checkFailed || current?._tag === "Failed" || current?._tag === "InstallationFailed")
  const busy = pending || checking.waiting || snapshot?.check._tag === "Checking"
  const actions = <>
    {installable ? <><NoticeAction disabled={busy} onClick={() => restart()}><T k="retryUpdate" /></NoticeAction><NoticeAction disabled={busy} onClick={() => discard()}><T k="discardDownload" /></NoticeAction></>
      : current?._tag === "Available" ? <NoticeAction disabled={busy} onClick={() => download()}><T k="downloadUpdate" /></NoticeAction>
      : current && !["Unavailable", "Closed"].includes(current._tag) ? <NoticeAction disabled={busy} onClick={() => check()}><T k="checkUpdates" /></NoticeAction> : null}
  </>
  return <>{disconnect.dialog}<SettingsRow label={version} hint={Result.isInitial(observation) ? <SkeletonLine className="h-4 text-xs" width="160px" /> : message}
    alert={hasFailure && !busy ? <ErrorNotice
      title={Result.isFailure(discarding) && !discarding.waiting ? tr("couldNotDiscardUpdate") : current?._tag === "InstallationFailed" ? tr("updateNotCompleted") : checkFailed ? tr("couldNotCheckUpdates") : tr("updateCouldNotFinish")}
      description={installable ? tr("preparedUpdateAvailable")
        : snapshot?.check._tag === "Failed" ? translateKnownMessage(snapshot.check.message)
        : current?._tag === "Failed" ? translateKnownMessage(current.message) : tr("checkConnectionRetry")}
      actions={actions} /> : undefined}
    control={hasFailure ? busy ? <span className="text-xs text-slate-500"><T k="working" /></span> : null : <>
      {installable && <Button size="sm" variant="ghost" disabled={pending} onClick={() => discard()}><T k="discardDownload" /></Button>}
      {installable ? <Button size="sm" disabled={pending} onClick={() => restart()}><T k="restartToUpdate" /></Button>
        : current?._tag === "Available" ? <Button size="sm" disabled={pending} onClick={() => download()}><T k="downloadUpdate" /></Button>
        : current && !["Unavailable", "Closed"].includes(current._tag) ? <Button size="sm" variant="outline" disabled={busy} onClick={() => check()}>{busy ? tr("checkingEllipsis") : tr("checkUpdates")}</Button>
        : null}
    </>} /></>
}
function SignOutRow() {
  const { origin } = useRemoteAccess()
  const action = useMemo(() => Atom.fn((_: void) => signOut(origin)), [origin])
  const run = useAtomSet(action)
  const signingOut = useAtomValue(action)
  return <SettingsRow label={tr("signedInFromAnotherDevice")} hint={tr("thisBrowserControlsAt").replace("{host}", new URL(origin).host)}
    alert={Result.isFailure(signingOut) && !signingOut.waiting ? <ErrorNotice title={tr("couldNotSignOut")} description={tr("checkDeviceReach")} /> : undefined}
    control={<Button size="sm" variant="outline" disabled={signingOut.waiting} onClick={() => run()}>{useI18n().t("signOut")}</Button>} />
}
function LanguageRow() {
  const { language, setLanguage, t } = useI18n()
  return <SettingsRow label={t("language")} hint={t("languageHint")}
    control={<Select value={language} onValueChange={value => { if (isLanguage(value)) setLanguage(value) }}>
      <SelectTrigger className="w-48" aria-label={t("language")}><SelectValue /></SelectTrigger>
      <SelectContent>
        <SelectItem value="en">{t("english")}</SelectItem>
        <SelectItem value="es">{t("spanish")}</SelectItem>
        <SelectItem value="zh-CN">{t("chineseSimplified")}</SelectItem>
        <SelectItem value="hi">{t("hindi")}</SelectItem>
        <SelectItem value="ur">{t("urdu")}</SelectItem>
        <SelectItem value="bn">{t("bengali")}</SelectItem>
        <SelectItem value="fr">Français</SelectItem>
        <SelectItem value="de">Deutsch</SelectItem>
        <SelectItem value="ja">日本語</SelectItem>
        <SelectItem value="pt">Português</SelectItem>
        <SelectItem value="it">Italiano</SelectItem>
        <SelectItem value="uk">Українська</SelectItem>
      </SelectContent>
    </Select>} />
}
function SettingsPage() {
  // A fresh mount observes hand edits; writes refresh in their own Effect actions.
  const client = useAgentClient()
  const { remote } = useRemoteAccess()
  const { storagePending, networkPending } = useRestartPending()
  const { t } = useI18n()
  useAtomMount(useMemo(() => Atom.make(Effect.all([Atom.refresh(client.Configuration.GetModelStorage({})), Atom.refresh(client.Configuration.GetNetworkAccess({}))], { discard: true })), [client]))
  return <>
    <SettingsGroup label={t("general")}><LanguageRow /><ThemeRow /><LaunchAtLoginRow /><ModelStorageRow /><NetworkAccessRows /><AutomaticUpdatesRow /></SettingsGroup>
    <SettingsGroup label={t("about")}><AboutRow /></SettingsGroup>
    {remote && <SettingsGroup label={t("thisBrowser")}><SignOutRow /></SettingsGroup>}
    {(storagePending || networkPending) && <div aria-hidden="true" className="h-44" />}
  </>
}
export function App() {
  const { t } = useI18n()
  const session = useSession()
  const observation = useServiceObservation()
  const page = useAtomValue(session.page)
  const navigate = useNavigate()
  const service = Result.isSuccess(observation.service) ? observation.service.value : null
  const platform = Option.getOrUndefined(Option.map(session.clientWindow, window => window.platform))
  return <><RestartRequiredToast /><HostNotices /><QuitFailureDialog /><AppShell page={page} navigate={navigate} platform={platform}>
      {service?._tag === "Unreachable" ? <CannotReachMagnitude />
      : service?._tag === "Reconnecting" ? <ReconnectingToMagnitude />
      : page === "status" ? Result.isFailure(observation.service) ? <ErrorNotice title={tr("couldNotReadServiceStatus")} description={tr("cannotConfirmServiceState")} className="mt-7" /> : <Status observation={observation} />
      : page === "usage" ? <ServingUsage />
      : page === "settings" ? <SettingsPage />
      : page === "connections" ? <Connections serviceReady={service?._tag === "Ready"} selectedModel={Option.none()} />
      : service?._tag !== "Ready" ? (service?._tag === "Failed" || service?._tag === "CleanupFailed" || Result.isFailure(observation.service) ? <>{page !== "discover" && <h1 className={pageLayout.pageTitle}>{localizedPageName(page, t)}</h1>}<ErrorNotice title={tr("magnitudeNeedsAttention")} description={tr("inferenceUnavailable")} className="mt-8" actions={<NoticeAction onClick={() => navigate("status")}><T k="openStatus" /></NoticeAction>} /></> : <ModelsSkeleton page={page} />)
      : page === "discover" || page === "catalog" || page === "models" ? <Models page={page} />
      : null}
  </AppShell></>
}
/** Shown while a browser's own retries run after its connection drops, as it does when Magnitude restarts. */
function ReconnectingToMagnitude() {
  return <ErrorNotice className="mt-8" severity="info" title={tr("reconnectingMagnitude")}
    description={tr("serviceRestartInterrupted")} />
}
/** A browser cannot restart the service; it keeps retrying in the background, and Reconnect tries at once. */
function CannotReachMagnitude() {
  const attempt = useReconnectAttempt()
  const reconnectAction = useMemo(() => Atom.fn((_: void) => attempt), [attempt])
  const reconnect = useAtomSet(reconnectAction)
  const reconnecting = useAtomValue(reconnectAction)
  return <ErrorNotice className="mt-8" title={tr("cantReachMagnitude")}
    description={tr("pageCantConnectService")}
    actions={<NoticeAction disabled={reconnecting.waiting} onClick={() => reconnect()}><T k="reconnect" /></NoticeAction>} />
}
/** Messages raised by Electron main while this window exists, such as a link that couldn't open. */
function HostNotices() {
  const session = useSession()
  const notices = useAtomValue(session.notices)
  const dismiss = useAtomSet(useMemo(() => Atom.fn((notice: HostNotice) => session.dismissNotice(notice)), [session]))
  if (notices.length === 0) return null
  return <div className="fixed bottom-4 left-4 z-50 flex w-96 max-w-[calc(100vw-2rem)] flex-col gap-2">
    {notices.map((notice, index) => <div key={index} className="rounded-lg bg-white shadow-md dark:bg-slate-850">
      <ErrorNotice title={translateKnownMessage(notice.title)} description={translateKnownMessage(notice.description)} actions={<NoticeAction onClick={() => dismiss(notice)}><T k="dismiss" /></NoticeAction>} />
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
        <AlertDialogTitle><T k="quittingTitle" /></AlertDialogTitle>
        <AlertDialogDescription>{tr("quitBackgroundProcesses")}</AlertDialogDescription>
      </AlertDialogHeader>
      <AlertDialogFooter>
        <AlertDialogCancel disabled={resolving.waiting} onClick={() => resolve("KeepOpen")}><T k="keepOpen" /></AlertDialogCancel>
        <Button variant="outline" disabled={resolving.waiting} onClick={() => resolve("ForceQuit")}><T k="forceQuitAction" /></Button>
        <AlertDialogAction disabled={resolving.waiting} onClick={() => resolve("RetryQuit")}><T k="retryQuitAction" /></AlertDialogAction>
      </AlertDialogFooter>
    </AlertDialogContent>
  </AlertDialog>
}
export function AppShell({ page, navigate, platform, children }: { page: ApplicationPage; navigate?: (page: ApplicationPage) => void; platform: string | undefined; children: ReactNode }) {
  const { t } = useI18n()
  const narrow = useNarrowViewport()
  const [open, setOpen] = useState(false)
  // Only the navigation chrome changes across `md`; the page stays mounted, so resizing keeps its state.
  return <div className={narrow ? "relative flex h-screen flex-col bg-slate-50 font-sans text-slate-900 dark:bg-slate-925 dark:text-slate-200" : "relative flex h-screen bg-slate-50 font-sans text-slate-900 dark:bg-slate-925 dark:text-slate-200"}>
    {narrow ? <NarrowBar open={open} onOpen={() => setOpen(true)} /> : <DockedSidebar page={page} navigate={navigate} platform={platform} />}
    <main key={page} className="min-w-0 flex-1 overflow-y-auto" inert={narrow && open}>
      <div data-page-content className={narrow ? "mx-auto w-full max-w-6xl px-4 pb-8 pt-5" : `mx-auto w-[calc(100vw-224px)] max-w-[min(100%,72rem)] px-10 pb-9 ${platform === "win32" ? "pt-14" : "pt-9"}`}>
        {page !== "catalog" && page !== "models" && <h1 className={pageLayout.pageTitle}>{localizedPageName(page, t)}</h1>}
        {children}
      </div>
    </main>
    {narrow && open && <NavigationDrawer page={page} navigate={navigate} onClose={() => setOpen(false)} />}
  </div>
}
function Navigation({ page, navigate, onNavigate }: { page: ApplicationPage; navigate?: (page: ApplicationPage) => void; onNavigate?: () => void }) {
  const { t } = useI18n()
  return <nav id="desktop-navigation" className="flex min-h-0 flex-1 flex-col gap-2 px-4">
    {(Object.keys(pageNames) as ApplicationPage[]).map(key => {
      const Icon = pageIcons[key]
      return <Button variant="ghost" key={key} disabled={!navigate} onClick={() => { navigate?.(key); onNavigate?.() }} aria-label={localizedPageName(key, t)} aria-current={page === key ? "page" : undefined} className={`h-10 gap-3 rounded-lg px-3 text-left text-sm font-medium justify-start ${key === "status" ? "mt-auto" : ""} ${page === key ? "bg-blue-50 text-blue-700 dark:bg-slate-800 dark:text-blue-400" : "hover:bg-slate-100 dark:hover:bg-slate-800"}`}>
        <Icon className="size-4 shrink-0" />{localizedPageName(key, t)}
      </Button>
    })}
  </nav>
}
/** Below `md`: a top bar that opens a navigation drawer over the page, closed by default. */
function NarrowBar({ open, onOpen }: { open: boolean; onOpen: () => void }) {
  return <header className="flex h-14 shrink-0 items-center gap-3 border-b border-slate-200 px-4 dark:border-slate-750" inert={open}>
    <button type="button" className="inline-flex size-10 items-center justify-center rounded-md text-slate-600 hover:bg-slate-100 focus-visible:outline-2 focus-visible:outline-blue-500 dark:text-slate-300 dark:hover:bg-slate-800" aria-label={useI18n().t("openNavigation")} aria-expanded={open} aria-controls="desktop-navigation" onClick={onOpen}>
      <SidebarSimpleIcon className="size-5" />
    </button>
    <span className="flex items-center gap-2 font-heading text-base font-semibold"><MagnitudeMark className="h-7 w-7 shrink-0" />Magnitude</span>
  </header>
}
function NavigationDrawer({ page, navigate, onClose }: { page: ApplicationPage; navigate?: (page: ApplicationPage) => void; onClose: () => void }) {
  return <div className="fixed inset-0 z-50 flex" onKeyDown={event => { if (event.key === "Escape") onClose() }}>
    <button type="button" aria-label={useI18n().t("closeNavigation")} className="absolute inset-0 cursor-default bg-black/45 dark:bg-black/70" onClick={onClose} />
    <aside aria-label={useI18n().t("navigation")} className="relative flex h-full w-64 max-w-[85vw] flex-col border-r border-slate-200 bg-slate-50 pb-6 pt-4 shadow-xl dark:border-slate-750 dark:bg-slate-925 motion-safe:animate-[slide-in-left_150ms_ease-out]">
      <div className="mb-6 flex h-10 items-center justify-between px-4">
        <span className="flex items-center gap-3 px-3 font-heading text-base font-semibold"><MagnitudeMark className="h-8 w-8 shrink-0" />Magnitude</span>
        <button type="button" autoFocus className="inline-flex size-10 items-center justify-center rounded-md text-slate-500 hover:bg-slate-100 focus-visible:outline-2 focus-visible:outline-blue-500 dark:hover:bg-slate-800" aria-label={useI18n().t("closeNavigation")} onClick={onClose}><XIcon className="size-5" /></button>
      </div>
      <Navigation page={page} navigate={navigate} onNavigate={onClose} />
    </aside>
  </div>
}
/** From `md` up: the collapsible sidebar with the window's integrated controls and drag regions. */
function DockedSidebar({ page, navigate, platform }: { page: ApplicationPage; navigate?: (page: ApplicationPage) => void; platform: string | undefined }) {
  const [collapsed, setCollapsed] = useState(false)
  const integratedControls = platform === "darwin" || platform === "win32"
  const sidebarWidth = collapsed ? 0 : 224
  return <>
    {integratedControls && <div aria-hidden="true" data-window-drag-region style={{ left: sidebarWidth }} className="absolute right-0 top-0 z-50 h-8 select-none transition-[left] duration-250 ease-in-out motion-reduce:transition-none [-webkit-app-region:drag]" />}
    <div data-window-drag-region={integratedControls ? "" : undefined} style={{ width: collapsed ? (platform === "darwin" ? 128 : 64) : sidebarWidth }} className={`absolute left-0 top-0 z-50 flex h-[42px] items-center justify-end px-4 transition-[width] duration-250 ease-in-out motion-reduce:transition-none ${integratedControls ? "select-none [-webkit-app-region:drag]" : ""}`}>
      <button type="button" className="inline-flex size-6 items-center justify-center rounded-sm text-slate-500 hover:text-slate-900 focus-visible:outline-2 focus-visible:outline-blue-500 dark:hover:text-slate-100 [-webkit-app-region:no-drag]" aria-label={useI18n().t(collapsed ? "expandSidebar" : "collapseSidebar")} aria-expanded={!collapsed} aria-controls="desktop-navigation" onClick={() => setCollapsed(value => !value)}>
        <SidebarSimpleIcon className="size-5" />
      </button>
    </div>
    <aside aria-hidden={collapsed} inert={collapsed} style={{ width: sidebarWidth }} className="shrink-0 overflow-hidden transition-[width] duration-250 ease-in-out motion-reduce:transition-none">
      <div className={`flex h-full w-56 flex-col border-r border-slate-200 pt-10 pb-8 transition-transform duration-250 ease-in-out motion-reduce:transition-none dark:border-slate-750 ${collapsed ? "-translate-x-full" : "translate-x-0"}`}>
      <div className="mb-10 mt-4 flex h-8 shrink-0 items-center gap-3 px-7 font-heading text-base font-semibold">
        <MagnitudeMark className="h-8 w-8 shrink-0" />Magnitude
      </div>
      <Navigation page={page} navigate={navigate} />
      </div>
    </aside>
  </>
}

