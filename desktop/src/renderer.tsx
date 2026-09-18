import { LoadingRegion, SkeletonLine, ModelsSkeleton, RecommendationsSkeleton, ConnectionsSkeleton, LoginSkeleton, UpdatesSkeleton } from "./page-skeletons"
import { pageLayout } from "./page-layout"
import { RecommendationPreference } from "./model-preference-slider"
import { ServingUsage } from "./serving-usage"
import { initializeAppearance, setAppearancePreference, useAppearancePreference, subscribeAppearance, getAppearancePreference } from "../../web/src/stores/appearance-store"
import { ActionTooltip, TooltipProvider } from "../../web/src/components/ui/tooltip"
import { Button } from "../../web/src/components/ui/button"
import { Input } from "../../web/src/components/ui/input"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "../../web/src/components/ui/select"
import { Progress } from "../../web/src/components/ui/progress"
import { MagnitudeMark } from "../../web/src/components/magnitude-mark"
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
} from "@phosphor-icons/react"
import { createRoot } from "react-dom/client"
import { useId, useMemo, useState, type ReactNode } from "react"
import { Atom, RegistryProvider, Result, useAtomValue, useAtomSet } from "@effect-atom/atom-react"
import { Cause, Effect, Exit, Layer, Option, Runtime, Schema, Scope, Stream } from "effect"
import { FetchHttpClient } from "@effect/platform"
import { MagnitudeClient, ProviderModelIdSchema, type ProviderModelId, type CatalogLocalModel } from "@magnitudedev/sdk"
import { ApplicationSnapshot, LoginStartupState } from "@magnitudedev/sdk/desktop-host"
import {
  DesktopApplicationInfo, DesktopUpdateState, DesktopConnectRequest, DesktopHostUnavailable, DesktopSession, DesktopConnectionsSnapshot, activeLocalModel, modelDownloadFailureMessage,
  createAgentClient, AgentClientProvider, useAgentClient, makeFirstPartyConnection,
  useCatalogModels, useLocalModelCommandStatus, useLocalModelMutations, useLocalModelStopStatus, useLocalModels, localModelFailureMessage, modelTrayPresentation, useLocalInferenceHardware, formatLocalModelDisplayName,
  formatStorageSize, formatTransferRate, formatMemorySize, localModelIsInstalled, localModelProviderModelId, rankedLocalModelOptions, featuredCatalogModels, targetPhysicalMemoryBytes,
  LOCAL_MODEL_RANKING_SCALE_VALUES,
} from "@magnitudedev/client-common"
import { HardwareOverview, ModelRadar } from "./discovery-visuals"
import { MemoryBreakdown } from "./memory-breakdown"
import { HarnessConnections } from "./harness-connections"
import { ModelLogo } from "./model-logo"
import type { DesktopApi, Page } from "./desktop-rpc"
import "@web-styles/tailwind.css"

initializeAppearance()
document.documentElement.dataset.desktopPlatform = window.__magnitudeDesktop.platform
const host = window.__magnitudeDesktop
class DesktopHostFailed extends Schema.TaggedError<DesktopHostFailed>()("DesktopHostFailed", { message: Schema.String }) {}
const hostCommand = (action: () => Promise<void>) => Effect.tryPromise({ try: action, catch: error => {
  const decoded = Schema.decodeUnknownEither(Schema.Struct({ message: Schema.String }))(error)
  return new DesktopHostFailed({ message: decoded._tag === "Right" ? decoded.right.message : "Magnitude could not complete this action. Try again or check Status." })
} })
const hostFailureMessage = (cause: Cause.Cause<unknown>) => {
  const failure = Cause.failureOption(cause)
  return Option.isSome(failure) && Schema.is(DesktopHostFailed)(failure.value)
    ? failure.value.message : "Magnitude could not complete this action. Try again or check Status."
}

const observation = Stream.asyncPush<typeof ApplicationSnapshot.Type, DesktopHostFailed>(emit => Effect.acquireRelease(
  Effect.sync(() => host.observe(encoded => {
    const value = Schema.decodeUnknownEither(ApplicationSnapshot)(encoded)
    if (value._tag === "Right") emit.single(value.right)
    else emit.fail(new DesktopHostFailed({ message: String(value.left) }))
  }, message => emit.fail(new DesktopHostFailed({ message })))), unsubscribe => Effect.sync(unsubscribe),
).pipe(Effect.asVoid))
const hostState = Atom.keepAlive(Atom.make(observation))
const pageNames: Record<Page, string> = { discover: "Discover", catalog: "Catalog", models: "My Models", connections: "Connections", usage: "Usage", status: "Status", settings: "Settings" }
const pageIcons = { discover: StackIcon, catalog: SquaresFourIcon, models: CubeIcon, connections: PlugIcon, usage: ChartBarIcon, status: PulseIcon, settings: SlidersIcon }

function ModelFit({ model }: { model: CatalogLocalModel }) {
  const serving = model.servingState
  if (serving._tag === "Assessing") return <p className="mt-3 text-sm text-slate-500">Checking compatibility with your machine…</p>
  if (serving._tag === "Failed") return <p className="mt-3 text-sm text-slate-500">Assessment unavailable: {serving.failure.message}</p>
  const assessment = serving.assessment
  if (assessment._tag === "Incompatible") return <p className="mt-3 text-sm text-slate-500">Not compatible: {assessment.failure.message}</p>
  if (assessment._tag === "DoesNotFit") return <p className="mt-3 text-sm text-slate-500">Doesn’t fit this machine · short by {formatMemorySize(assessment.deficitBytes, { rounding: "up" })} of memory.</p>
  return <p className="mt-3 text-sm text-blue-700 dark:text-blue-400">Fits your machine · {formatMemorySize(assessment.memory.totalRequiredBytes)} estimated memory</p>
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
      {serving._tag === "Assessed" && serving.assessment._tag === "Fits" && serving.assessment.performance.length > 0 && <div className="min-w-0">
        <table className="w-full text-left text-sm tabular-nums">
          <caption className="mb-3 text-left font-medium">Estimated speed on your machine</caption>
          <thead className="text-xs text-slate-500"><tr><th className="pb-2 font-normal">Context tokens</th><th className="pb-2 text-right font-normal">Tokens / sec</th></tr></thead>
          <tbody>{serving.assessment.performance.map(sample => <tr key={sample.contextTokens} className="border-t border-slate-200 dark:border-slate-750"><td className="py-2">{sample.contextTokens.toLocaleString()}</td><td className="py-2 text-right">{Math.round(sample.estimatedTokensPerSecond)}</td></tr>)}</tbody>
        </table>
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
function DownloadProgress({ acquisition, modelName, onCancel, pending = false }: { acquisition: CatalogLocalModel["acquisitionState"]; modelName: string; onCancel?: () => void; pending?: boolean }) {
  if (acquisition._tag !== "Installing" && acquisition._tag !== "Updating") return null
  const { progress } = acquisition
  const downloading = progress.stage === "downloading"
  const percent = progress.totalBytes > 0 ? progress.completedBytes / progress.totalBytes * 100 : null
  const rate = downloading ? Option.getOrNull(progress.bytesPerSecond) : null
  const eta = downloading && rate !== null && rate > 0 && progress.totalBytes > 0
    ? `About ${Math.max(1, Math.ceil((progress.totalBytes - progress.completedBytes) / rate / 60))} min`
    : downloading ? "Estimating…" : "—"
  const stages = { queued: "Queued", resolving: "Preparing download", checking_space: "Checking space", downloading: "Downloading", verifying: "Verifying download", publishing: "Finishing download" }
  const bytes = `${formatStorageSize(progress.completedBytes)} / ${progress.totalBytes > 0 ? formatStorageSize(progress.totalBytes) : "Unknown total"}`
  return <div className="w-full min-w-0" aria-label="Model download">
    <h3 className="mb-6 flex items-center gap-2 text-sm font-medium">
      {downloading ? <><span className="shrink-0">Downloading</span><span className="min-w-0 flex-1 truncate" title={modelName}>{modelName}</span></> : <span className="min-w-0 flex-1 truncate">{stages[progress.stage]}</span>}
      <CircleNotchIcon aria-hidden="true" className="size-3.5 shrink-0 text-blue-700 motion-safe:animate-spin dark:text-blue-400" />
    </h3>
    <Progress aria-label="Download progress" aria-valuetext={bytes} value={percent} trackClassName="h-2" indicatorClassName={`bg-blue-700 dark:bg-blue-500 ${percent === null ? "w-full motion-safe:animate-pulse" : ""}`} />
    <div className="mt-3 flex items-baseline justify-between gap-3 text-sm tabular-nums text-slate-600 dark:text-slate-300"><span>{bytes}</span><span>{percent !== null ? `${Math.floor(percent)}%` : "—"}</span></div>
    <dl className="mt-6 grid grid-cols-2 gap-4 text-sm"><div><dt className="text-xs text-slate-500 dark:text-slate-400">Download speed</dt><dd className="mt-1 font-medium tabular-nums text-slate-800 dark:text-slate-200">{rate !== null ? formatTransferRate(rate) : "—"}</dd></div><div className="text-right"><dt className="text-xs text-slate-500 dark:text-slate-400">Time remaining</dt><dd className="mt-1 font-medium tabular-nums text-slate-800 dark:text-slate-200">{eta}</dd></div></dl>
    {onCancel && <div className="mt-7 flex justify-center"><Button variant="ghost" className="hover:bg-transparent hover:text-red-600 dark:hover:bg-transparent dark:hover:text-red-400" disabled={pending} onClick={onCancel}><XIcon />Cancel download</Button></div>}
  </div>
}
function ModelControls({ model, replacing, children }: { model: CatalogLocalModel; replacing?: string; children?: ReactNode }) {
  const { install, load, stop, cancel, remove, dismissFailure: dismiss } = useLocalModelMutations()
  const command = useLocalModelCommandStatus(model.modelId)
  const stopping = useLocalModelStopStatus()
  const pending = command.pending || stopping.pending || model.acquisitionState._tag === "Removing"
  const acquisition = model.acquisitionState
  const installed = "residencyState" in acquisition
  const residency = installed ? acquisition.residencyState : undefined
  const canStop = residency !== undefined && ["Ready", "Loading", "Requested", "Stopping"].includes(residency._tag)
  const transferring = acquisition._tag === "Installing" || acquisition._tag === "Updating"
  return <div>
    <div className="flex flex-wrap items-center gap-2">{children}
      {transferring ? <DownloadProgress modelName={formatLocalModelDisplayName(model)} acquisition={acquisition} pending={command.pending} onCancel={() => cancel(model.modelId)} /> : !installed ? <Button disabled={pending || model.servingState._tag !== "Assessed" || model.servingState.assessment._tag !== "Fits"} onClick={() => { install(model.modelId) }}><DownloadSimpleIcon />Download ({formatStorageSize(model.storageBytes).replace(/\s/g, "")})</Button> : <>
        {canStop ? <Button className="min-w-28" variant="outline" disabled={stopping.pending} onClick={() => stop()}><SquareIcon />Stop model</Button> : <Button className="min-w-28" disabled={pending} onClick={() => { if (!replacing || window.confirm(`Loading ${formatLocalModelDisplayName(model)} will stop ${replacing}. Continue?`)) load(model.modelId) }}><PlayIcon />Load model</Button>}
        <Button variant="ghost" size="icon" aria-label={`Remove ${formatLocalModelDisplayName(model)}`} title="Remove download" disabled={pending} onClick={() => { if (window.confirm(`Remove the downloaded files for ${formatLocalModelDisplayName(model)}?`)) remove(model.modelId) }}><TrashIcon /></Button>
        {(acquisition._tag === "UpdateAvailable" || acquisition._tag === "UpdateFailed") && <Button variant="outline" disabled={pending} onClick={() => install(model.modelId)}>Update</Button>}
      </>}
      {(acquisition._tag === "InstallFailed" || acquisition._tag === "UpdateFailed") && <Button variant="outline" onClick={() => dismiss(model.modelId)}>Dismiss error</Button>}
    </div>
    {(model.servingState._tag !== "Assessed" || model.servingState.assessment._tag !== "Fits") && <ModelFit model={model} />}
    {"failure" in acquisition && <p role="alert" className="mt-3 text-sm">{acquisition._tag === "InstallFailed" || acquisition._tag === "UpdateFailed" ? modelDownloadFailureMessage(acquisition.failure) : acquisition.failure.message}</p>}
    {residency?._tag === "Failed" && <p role="alert" className="mt-3 text-sm">{residency.failure.message}</p>}
    {command.failures.map(message => <p key={message} role="alert" className="mt-3 text-sm">{message}</p>)}
    {residency?._tag === "Stopping" && Option.isSome(stopping.failure) && <p role="alert" className="mt-3 text-sm">{stopping.failure.value}</p>}
  </div>
}
function ModelCard({ model, showMemory = false, replacing }: { model: CatalogLocalModel; showMemory?: boolean; replacing?: string }) {
  const [detailsOpen, setDetailsOpen] = useState(false)
  const detailsId = useId()
  const acquisition = model.acquisitionState
  const residency = "residencyState" in acquisition ? acquisition.residencyState : undefined
  const statusLabel = acquisition._tag === "Removing" ? "Removing…" : acquisition._tag === "RemoveFailed" ? "Removal failed" : residency?._tag === "Ready" ? "Loaded" : residency?._tag === "Unloaded" ? "Downloaded" : residency?._tag ?? (acquisition._tag === "NotInstalled" ? "" : acquisition._tag)
  const status = (statusLabel || showMemory) && <div className="mt-1 flex flex-wrap items-center gap-x-3 text-sm text-slate-500">{statusLabel && <span className={residency?._tag === "Ready" ? "text-green-600 dark:text-green-400" : ""}>{statusLabel}</span>}{showMemory && model.servingState._tag === "Assessed" && model.servingState.assessment._tag === "Fits" && <><span aria-hidden="true">·</span><span>{formatMemorySize(model.servingState.assessment.memory.totalRequiredBytes)} memory</span></>}</div>
  return <article className={pageLayout.modelCard}>
    <div className={pageLayout.modelRow}>
      <div className="flex min-w-0 items-center gap-4"><ModelLogo model={model} /><div className="min-w-0"><h2 className="text-lg font-semibold">{formatLocalModelDisplayName(model)}</h2>{status}</div></div>
      <ModelControls model={model} {...(replacing ? { replacing } : {})}>
        <Button variant="ghost" aria-expanded={detailsOpen} aria-controls={detailsId} onClick={() => setDetailsOpen(value => !value)}>Details<CaretDownIcon aria-hidden="true" className={`size-4 ${detailsOpen ? "rotate-180" : ""}`} /></Button>
      </ModelControls>
    </div>
    <ModelDetails model={model} radar open={detailsOpen} contentId={detailsId} />
  </article>
}
function SelectedRecommendation({ model, active }: { model: CatalogLocalModel; active: ReturnType<typeof activeLocalModel> }) {
  const [view, setView] = useState<"profile" | "details">("profile")
  const { cancel } = useLocalModelMutations()
  const command = useLocalModelCommandStatus(model.modelId)
  const transferring = model.acquisitionState._tag === "Installing" || model.acquisitionState._tag === "Updating"
  return <div className={`relative ${pageLayout.recommendationPane}`} aria-label="Selected model profile">
    <div className={transferring ? "invisible" : undefined} inert={transferring} aria-hidden={transferring}>
    <div className={pageLayout.recommendationToolbar}>
    <div className="flex items-center gap-1" aria-label="Model information">
      <Button variant={view === "profile" ? "secondary" : "ghost"} aria-pressed={view === "profile"} onClick={() => setView("profile")}>Profile</Button>
      <Button variant={view === "details" ? "secondary" : "ghost"} aria-pressed={view === "details"} onClick={() => setView("details")}>Details</Button>
    </div>
      {transferring ? <Button disabled><DownloadSimpleIcon />Download ({formatStorageSize(model.storageBytes).replace(/\s/g, "")})</Button> : <ModelControls model={model} {...(Option.isSome(active) && active.value.model.modelId !== model.modelId ? { replacing: formatLocalModelDisplayName(active.value.model) } : {})} />}
    </div>
    <div className="grid min-h-72">
      <div className={`col-start-1 row-start-1 min-w-0 ${view === "profile" ? "" : "invisible"}`} aria-hidden={view !== "profile"}><ModelRadar model={model} /></div>
      <div className={`col-start-1 row-start-1 min-w-0 ${view === "details" ? "" : "invisible"}`} aria-hidden={view !== "details"}><ModelDetails model={model} compact open /></div>
    </div>
    </div>
    {transferring && <div className="absolute inset-5 flex items-center justify-center overflow-y-auto" aria-label="Download panel">
      <div className="w-full max-w-sm px-3 py-4">
        <DownloadProgress modelName={formatLocalModelDisplayName(model)} acquisition={model.acquisitionState} pending={command.pending} onCancel={() => cancel(model.modelId)} />
        {command.failures.map(message => <p key={message} role="alert" className="mt-3 text-sm">{message}</p>)}
      </div>
    </div>}
  </div>
}
function Recommendations({ models, active, preference }: { models: readonly CatalogLocalModel[]; preference: number; active: ReturnType<typeof activeLocalModel> }) {
  const [selection, setSelection] = useState<{ preference: number; modelId: CatalogLocalModel["modelId"] | null }>({ preference, modelId: null })
  if (selection.preference !== preference) setSelection({ preference, modelId: null })
  const selectedId = selection.preference === preference ? selection.modelId : null
  const downloads = models.filter(model => model.acquisitionState._tag === "Installing" || model.acquisitionState._tag === "Updating")
  const selectable = downloads.length > 0 ? downloads : models
  const selected = selectable.find(model => model.modelId === selectedId) ?? selectable[0]
  if (!selected) return null
  return <section aria-label="Top recommendations" className="mb-8">
    <div className={pageLayout.recommendations}>
      <div className={pageLayout.recommendationList} aria-label="Recommended models">{models.map((model, rank) => <button key={model.modelId} type="button" aria-pressed={model.modelId === selected.modelId} onClick={() => setSelection({ preference, modelId: model.modelId })} className={`${pageLayout.recommendationRow} transition-colors focus-visible:outline-2 focus-visible:outline-blue-500 ${model.modelId === selected.modelId ? "border-blue-300 bg-blue-50 dark:border-blue-700 dark:bg-slate-800" : "border-transparent hover:bg-slate-100 dark:hover:bg-slate-800"}`}>
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
      </button>)}</div>
      <SelectedRecommendation model={selected} active={active} />
    </div>
  </section>
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
  const [sort, setSort] = useState(installedOnly ? "name" : "recommended")
  const client = useAgentClient()
  const session = useMemo(() => client.runtime.atom(DesktopSession), [client])
  const preferenceAtom = useMemo(() => Atom.make(get => Result.map(get(session), service => get(service.rankingPreference))), [session])
  const preferenceResult = useAtomValue(preferenceAtom)
  const preference = Result.isSuccess(preferenceResult) ? preferenceResult.value : 2
  const setPreference = useAtomSet(useMemo(() => client.runtime.fn((index: number) => Effect.flatMap(DesktopSession, service => service.setRankingPreference(index))), [client]))
  if (Result.isFailure(catalog)) return <>{!discover && <h1 className={pageLayout.pageTitle}>{pageNames[page]}</h1>}<p role="alert" className="mt-5">{localModelFailureMessage(catalog.cause, "Could not read the model catalog. Check Status and try again.")}</p></>
  if (!Result.isSuccess(catalog) && !discover) return <ModelsSkeleton page={page} />
  const models = (Result.isSuccess(catalog) ? catalog.value.models : []).filter((model): model is CatalogLocalModel => model._tag === "Catalog")
  const ranked = !installedOnly && Result.isSuccess(hardware) ? rankedLocalModelOptions(models.map(model => ({ id: model.modelId, kind: localModelIsInstalled(model) ? "stored" as const : "downloadable" as const, model })), { fastToSmart: LOCAL_MODEL_RANKING_SCALE_VALUES[preference]!, memoryBudgetBytes: targetPhysicalMemoryBytes(hardware.value) }, models.length).flatMap(option => option.model._tag === "Catalog" ? [option.model] : []) : []
  const assessment = Result.isSuccess(catalog) ? catalog.value.preparation.assessment : undefined
  const recommendationsPending = !Result.isFailure(hardware) && (Result.isInitial(hardware) || !assessment?.complete)
  const rankedIds = new Set(ranked.map(model => model.modelId))
  const ordered = installedOnly ? models : [...ranked, ...models.filter(model => !rankedIds.has(model.modelId))]
  const library = ordered.filter(model => !installedOnly || model.acquisitionState._tag !== "NotInstalled")
  const visible = library.filter(model => {
    const acquisition = model.acquisitionState
    const matchesFilter = filter === "all"
      || filter === "fits" && model.servingState._tag === "Assessed" && model.servingState.assessment._tag === "Fits"
      || filter === "downloaded" && localModelIsInstalled(model)
      || filter === "downloading" && (acquisition._tag === "Installing" || acquisition._tag === "Updating")
    return matchesFilter && `${formatLocalModelDisplayName(model)} ${model.presentation.description}`.toLowerCase().includes(search.trim().toLowerCase())
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
          <Select items={sortOptions} value={sort} onValueChange={value => { if (value !== null) setSort(value) }}>
            <SelectTrigger aria-label="Sort models"><span className="text-slate-500">Sort:</span><SelectValue /></SelectTrigger>
            <SelectContent>{sortOptions.map(option => <SelectItem key={option.value} value={option.value}>{option.label}</SelectItem>)}</SelectContent>
          </Select>
        </div>
        <Input aria-label="Search models" placeholder="Search models…" className={pageLayout.modelSearch} value={search} onChange={event => setSearch(event.target.value)} />
      </div>
    </>}
    {discover && <HardwareOverview /> }
    {Option.isSome(stopResult.failure) && <p role="alert" className="mt-5 text-sm">{stopResult.failure.value}</p>}
    {discover && <RecommendationPreference value={preference} onChange={setPreference} />}
    {!discover && assessment && !assessment.complete && <p className="mb-4 text-sm text-slate-500">Assessing models · {assessment.settledModels} of {assessment.totalModels}</p>}
    {discover && (recommendationsPending
      ? <RecommendationsSkeleton assessment={assessment} waitingForHardware={Result.isInitial(hardware)} />
      : <Recommendations preference={preference} models={featuredCatalogModels(ranked, 5)} active={Option.fromNullable(active)} />)}
    {!discover && <>
    <div className="grid items-start gap-5">{visible.map(model => <ModelCard key={model.modelId} model={model} showMemory={installedOnly} {...(active && active.model.modelId !== model.modelId ? { replacing: formatLocalModelDisplayName(active.model) } : {})} />)}</div>
    {visible.length === 0 && <p className="py-8 text-slate-500">{search.trim() || filter !== "all" ? "No models match your search or filter." : installedOnly ? "No models downloaded yet. Find one in Discover." : "No models match this filter."}</p>}
    </>}
    {discover && ranked.length === 0 && !recommendationsPending && Result.isSuccess(hardware) && <p className="py-8 text-slate-500">No fitting recommendations right now. Explore Catalog for compatibility details.</p>}
  </>
}
function Connections({ serviceReady, selectedModel }: { serviceReady: boolean; selectedModel: Option.Option<ProviderModelId> }) {
  const client = useAgentClient()
  const session = useMemo(() => client.runtime.atom(DesktopSession), [client])
  const service = useAtomValue(session)
  return Result.isSuccess(service) ? <ConnectionsView service={service.value} serviceReady={serviceReady} selectedModel={selectedModel} /> : Result.isFailure(service) ? <p role="alert" className="mt-5">Could not prepare connections. {hostFailureMessage(service.cause)}</p> : <ConnectionsSkeleton />
}
function ConnectionsView({ service, serviceReady, selectedModel }: { service: DesktopSession; serviceReady: boolean; selectedModel: Option.Option<ProviderModelId> }) {
  const models = useLocalModels()
  const canConnect = serviceReady && Result.isSuccess(models) && models.value.models.some(model => Option.isSome(localModelProviderModelId(model)))
  const client = useAgentClient()
  const discover = useAtomSet(useMemo(() => client.runtime.fn(() => Effect.flatMap(DesktopSession, session => session.navigate("discover"))), [client]))
  const rows = useAtomValue(service.connections)
  const connect = useAtomSet(service.connect)
  const disconnect = useAtomSet(service.disconnect)
  const connecting = useAtomValue(service.connect)
  const disconnecting = useAtomValue(service.disconnect)
  const busy = connecting.waiting || disconnecting.waiting
  const error = [connecting, disconnecting].find(Result.isFailure)
  return <>
    {!serviceReady && <p className="mt-5 text-sm text-slate-500">Configuration checks are available. Start the service from Status before connecting a harness.</p>}
    {serviceReady && !canConnect && !Result.isInitial(models) && <div className="mt-5 flex flex-wrap items-center gap-3 text-sm text-slate-500"><p>{Result.isFailure(models) ? "Model availability could not be checked." : !Result.isSuccess(models) ? "Checking available models…" : "Download a compatible model before connecting a harness. It doesn’t need to be loaded."}</p><Button variant="outline" onClick={() => discover()}>Discover models</Button></div>}
    {error && Result.isFailure(error) && <p role="alert" className="mt-5 text-sm">{hostFailureMessage(error.cause)}</p>}
    {Result.isFailure(rows) ? <p role="alert" className="mt-5">Could not check connections. {hostFailureMessage(rows.cause)}</p>
      : !Result.isSuccess(rows) ? <ConnectionsSkeleton />
      : rows.value._tag === "Unavailable" ? <p role="alert" className="mt-5">Could not check connections. {rows.value.message}</p>
      : <HarnessConnections connections={rows.value.connections} busy={busy} canConnect={canConnect}
          onConnect={harness => connect({ harness, model: selectedModel })} onDisconnect={harness => disconnect(harness)} />}
  </>
}
function ModelStatus() {
  const models = useLocalModels()
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
          {active && <><span aria-hidden="true" className="text-slate-400 dark:text-slate-500">·</span><span className={`shrink-0 text-xs ${active.residency._tag === "Ready" ? "text-green-700 dark:text-green-400" : "text-slate-500 dark:text-slate-400"}`}>{active.residency._tag === "Ready" ? "Loaded" : active.residency._tag === "Requested" ? "Queued" : `${active.residency._tag}…`}</span></>}
        </div>
      </div>
      {presentation?.canStop && <Button variant="outline" className="hover:border-red-300 hover:text-red-600 dark:hover:border-red-800 dark:hover:text-red-400" disabled={stopping.pending} onClick={() => stop()}><SquareIcon />Stop model</Button>}
    </div>
    {active?.residency._tag === "Loading" && <div className="mt-4"><Progress aria-label="Model loading progress" indicatorClassName="bg-blue-700 dark:bg-blue-500" value={Option.match(active.residency.progress, { onNone: () => null, onSome: fraction => fraction * 100 })} /></div>}
    {Result.isFailure(models) && <p role="alert" className="mt-2 text-sm text-slate-500">{localModelFailureMessage(models.cause, "Could not read model status. Check Status and try again.")}</p>}
    {Option.isSome(stopping.failure) && <p role="alert" className="mt-2 text-sm">{stopping.failure.value}</p>}
  </div>
}
function DownloadActivity() {
  const models = useLocalModels()
  const active = Result.isSuccess(models) ? models.value.models.filter((model): model is CatalogLocalModel => model._tag === "Catalog" && (model.acquisitionState._tag === "Installing" || model.acquisitionState._tag === "Updating" || model.acquisitionState._tag === "Removing")) : []
  if (Result.isInitial(models)) return null
  if (Result.isSuccess(models) && active.length === 0) return null
  return <div className="mt-5 border-t border-slate-200 pt-5 dark:border-slate-700">
    {!Result.isSuccess(models) ? <p className="mt-3 text-sm text-slate-500">{Result.isFailure(models) ? "Download activity unavailable" : "Reading download activity…"}</p> : <ul className="space-y-5">{active.map(model => <li key={model.modelId}><div className="flex items-center gap-3"><ModelLogo model={model} className="size-6" /><p className="text-sm">{formatLocalModelDisplayName(model)} · {model.acquisitionState._tag === "Removing" ? "Removing files…" : model.acquisitionState._tag === "Updating" ? "Updating" : "Downloading"}</p></div><DownloadProgress modelName={formatLocalModelDisplayName(model)} acquisition={model.acquisitionState} /></li>)}</ul>}
  </div>
}
function Status({ snapshot }: { snapshot: typeof ApplicationSnapshot.Type | null }) {
  const service = snapshot?.service
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
      {ready ? <ModelStatus /> : !snapshot ? <SkeletonLine className="mt-5 h-12 w-64" /> : <p className="mt-5 text-sm text-slate-500">{service?._tag}</p>}
      {ready && <DownloadActivity />}
      {service && "message" in service && <p role="alert" className="mt-5 text-sm">{service.message}</p>}
      {service?._tag === "Failed" && <Button className="mt-5" variant="outline" onClick={() => host.retry()}>Retry service</Button>}
    </section>
    <MemoryBreakdown />
    <section className={pageLayout.card}><div className="flex items-center gap-3"><PlugIcon className="size-5 text-blue-600 dark:text-blue-400" /><h2 className="font-heading text-lg">Local connection</h2></div><p className="mt-2 text-sm text-slate-500">Your tools connect to Magnitude on this machine.</p><p className="mt-4 break-all rounded-lg bg-slate-50 p-4 font-mono text-sm dark:bg-slate-900">{snapshot?.endpoint ?? <SkeletonLine className="h-5 text-sm" width="200px" />}</p><div className="mt-5 flex items-start gap-3"><span className={`mt-1 size-2 shrink-0 rounded-full ${snapshot?.tray._tag === "Registered" ? "bg-blue-500" : "bg-slate-400"}`} /><div><p className="text-sm font-medium">Background activity</p><p className="mt-1 text-sm text-slate-500">{snapshot?.tray._tag === "Registered" ? "Magnitude keeps running when you close the window." : snapshot?.tray._tag === "Unavailable" ? snapshot.tray.message : snapshot?.tray._tag === "Closed" ? "Magnitude is quitting." : <SkeletonLine className="h-5 w-72 text-sm" />}</p></div></div></section>
  </div>
}
function ApplicationSettings() {
  const client = useAgentClient()
  const session = useMemo(() => client.runtime.atom(DesktopSession), [client])
  const service = useAtomValue(session)
  const info = useAtomValue(useMemo(() => Atom.make(get => Result.flatMap(get(session), value => get(value.applicationInfo))), [session]))
  return <section className={pageLayout.settingsCard}>
    <h2 className="font-heading text-lg">About Magnitude</h2>
    <p className="mt-2 text-sm text-slate-500">{Result.isSuccess(info) ? `Version ${info.value.version}` : Result.isFailure(info) ? "Version unavailable" : <SkeletonLine className="h-5 text-sm" width="96px" />}</p>
    {Result.isSuccess(service) ? <UpdateSettingsView service={service.value} /> : Result.isFailure(service) ? <p role="alert">Update settings unavailable.</p> : <UpdatesSkeleton />}
  </section>
}
function UpdateSettingsView({ service }: { service: DesktopSession }) {
  const observation = useAtomValue(service.updates)
  const check = useAtomSet(service.checkUpdate)
  const discard = useAtomSet(service.discardUpdate)
  const discarding = useAtomValue(service.discardUpdate)
  const download = useAtomSet(service.downloadUpdate)
  const restart = useAtomSet(service.restartUpdate)
  const setAutoDownload = useAtomSet(service.setAutoDownload)
  const checking = useAtomValue(service.checkUpdate)
  const downloading = useAtomValue(service.downloadUpdate)
  const restarting = useAtomValue(service.restartUpdate)
  const saving = useAtomValue(service.setAutoDownload)
  const snapshot = Result.isSuccess(observation) ? observation.value : null
  const current = snapshot?.transfer
  const pending = downloading.waiting || restarting.waiting || discarding.waiting
  const message = !current ? Result.isFailure(observation) ? "Update status unavailable." : "Reading update status…"
    : current._tag === "Idle" ? snapshot?.check._tag === "Succeeded" ? "You’re up to date." : "Magnitude checks for updates automatically."
    : current._tag === "Available" ? `Version ${current.version} is available · ${formatStorageSize(current.bytes)}`
    : current._tag === "Downloading" ? `Downloading version ${current.version} · ${formatStorageSize(current.completed)} of ${formatStorageSize(current.total)}`
    : current._tag === "Cancelling" ? "Stopping automatic download…"
    : current._tag === "Staging" ? `Preparing version ${current.version}…`
    : current._tag === "Ready" ? `Version ${current.version} is ready to install. Restart Magnitude to update.`
    : current._tag === "Closed" ? "Magnitude is quitting…" : current.message
  if (Result.isInitial(observation)) return <UpdatesSkeleton />
  return <div className="mt-5 border-t border-slate-200 pt-5 dark:border-slate-750">
    <h3 className="font-medium">Application updates</h3>
    {snapshot?.preference._tag === "Known" && <label className="mt-3 flex items-center gap-3 text-sm">
      <input type="checkbox" className="size-4 accent-sky-500" checked={snapshot.preference.autoDownload} disabled={saving.waiting || current?._tag === "Closed"} onChange={event => setAutoDownload(event.target.checked)} />
      Auto-download updates
    </label>}
    {snapshot?.preference._tag === "Unavailable" && current?._tag !== "Unavailable" && <div className="mt-2">
      <p role="alert" className="text-sm">{snapshot.preference.message}</p>
      <div className="mt-2 flex gap-2"><Button variant="outline" disabled={saving.waiting} onClick={() => setAutoDownload(true)}>Enable automatic downloads</Button>
        <Button variant="outline" disabled={saving.waiting} onClick={() => setAutoDownload(false)}>Use manual downloads</Button></div>
    </div>}
    <p className="mt-3 text-sm text-slate-500" role="status">{message}</p>
    {snapshot?.check._tag === "Failed" && <p className="mt-2 text-sm" role="alert">{snapshot.check.message}</p>}
    <div className="mt-3 flex flex-wrap gap-2">
      {current && !["Unavailable", "Closed"].includes(current._tag) && <Button variant="outline" disabled={checking.waiting || snapshot?.check._tag === "Checking"} onClick={() => check()}>{snapshot?.check._tag === "Checking" ? "Checking…" : "Check for updates"}</Button>}
      {current?._tag === "Available" && <Button disabled={pending} onClick={() => download()}>Download update</Button>}
      {(current?._tag === "Ready" || current?._tag === "InstallationFailed") && <Button disabled={pending} onClick={() => restart()}>{current._tag === "InstallationFailed" ? "Retry update" : "Restart to update"}</Button>}
      {(current?._tag === "Ready" || current?._tag === "InstallationFailed") && <Button variant="outline" disabled={pending} onClick={() => discard()}>Discard download</Button>}
    </div>
    {current?._tag === "Ready" && <p className="mt-2 text-sm text-slate-500">Restarting stops the running model and service.</p>}
    {[checking, downloading, restarting, discarding, saving].map((result, index) => Result.isFailure(result) ? <p key={index} role="alert" className="mt-2 text-sm">{hostFailureMessage(result.cause)}</p> : null)}
  </div>
}
function AppearanceSettings() {
  const appearance = useAppearancePreference()
  return <section aria-labelledby="appearance-heading" className="mt-8 overflow-hidden rounded-lg border border-slate-300 bg-white dark:border-slate-750 dark:bg-slate-850">
    <header className="border-b border-slate-200 px-5 py-4 dark:border-slate-800"><h2 id="appearance-heading" className="font-heading text-lg">Appearance</h2></header>
    <div className="flex flex-wrap items-center justify-between gap-6 px-5 py-5"><div><p className="font-medium">Theme</p><p className="mt-1 text-sm text-slate-500">Use your system appearance or choose a theme.</p></div>
      <div className="flex gap-2" role="group" aria-label="Theme">{(["system", "light", "dark"] as const).map(value => { const Icon = value === "system" ? MonitorIcon : value === "light" ? SunIcon : MoonIcon; return <Button key={value} variant={appearance === value ? "default" : "outline"} aria-pressed={appearance === value} onClick={() => setAppearancePreference(value)}><Icon />{value[0]!.toUpperCase() + value.slice(1)}</Button> })}</div>
    </div>
  </section>
}
function LoginSettings() {
  const client = useAgentClient()
  const session = useMemo(() => client.runtime.atom(DesktopSession), [client])
  const service = useAtomValue(session)
  return Result.isSuccess(service) ? <LoginSettingsView service={service.value} /> : Result.isFailure(service) ? <p role="alert" className="mt-6">Login settings unavailable.</p> : <LoginSkeleton />
}
function LoginSettingsView({ service }: { service: DesktopSession }) {
  const state = useAtomValue(service.loginStartup)
  const set = useAtomSet(service.setLoginStartup)
  const change = useAtomValue(service.setLoginStartup)
  const current = Result.isSuccess(state) ? state.value : null
  if (Result.isInitial(state)) return <LoginSkeleton />
  const enabled = current?._tag === "Enabled" || current?._tag === "RequiresApproval"
  return <section className={pageLayout.settingsCard}>
    <div className="flex items-center justify-between gap-6"><div><h2 className="font-heading text-lg">Launch at login</h2><p className="mt-2 text-sm text-slate-500">Start Magnitude in the background with its tray icon. The window stays closed.</p></div>
    <Button variant="outline" disabled={!current || current._tag === "Unavailable" || change.waiting} aria-pressed={enabled} onClick={() => set(!enabled)}>{enabled ? "Disable" : "Enable"}</Button></div>
    {current?._tag === "Unavailable" && <p className="mt-3 text-sm text-slate-500">{current.message}</p>}
    {current?._tag === "RequiresApproval" && <p className="mt-3 text-sm">Allow Magnitude in your system login settings to finish enabling startup.</p>}
    {Result.isFailure(state) && <p role="alert" className="mt-3 text-sm">Could not read login startup. {hostFailureMessage(state.cause)}</p>}
    {Result.isFailure(change) && <p role="alert" className="mt-3 text-sm">{hostFailureMessage(change.cause)}</p>}
  </section>
}
function App() {
  const state = useAtomValue(hostState)
  const client = useAgentClient()
  const session = useMemo(() => client.runtime.atom(DesktopSession), [client])
  const pageAtom = useMemo(() => Atom.make(get => Result.map(get(session), value => get(value.page))), [session])
  const navigate = useAtomSet(useMemo(() => client.runtime.fn((page: Page) => Effect.flatMap(DesktopSession, session => session.navigate(page))), [client]))
  const pageResult = useAtomValue(pageAtom)
  const page = Result.isSuccess(pageResult) ? pageResult.value : "discover"
  const service = Result.isSuccess(state) ? state.value.service : null
  return <DesktopShell page={page} navigate={navigate}>
      {page === "status" ? Result.isFailure(state) ? <p role="alert" className="mt-7">{hostFailureMessage(state.cause)}</p> : <Status snapshot={Result.isSuccess(state) ? state.value : null} />
      : page === "usage" ? <ServingUsage />
      : page === "settings" ? <><AppearanceSettings /><LoginSettings /><ApplicationSettings /></>
      : page === "connections" ? <Connections serviceReady={service?._tag === "Ready"} selectedModel={Option.none()} />
      : service?._tag !== "Ready" ? (service?._tag === "Failed" || service?._tag === "CleanupFailed" || Result.isFailure(state) ? <>{page !== "discover" && <h1 className={pageLayout.pageTitle}>{pageNames[page]}</h1>}<p role="alert" className="mt-8">The service needs attention. Open Status for details.</p></> : <ModelsSkeleton page={page} />)
      : page === "discover" || page === "catalog" || page === "models" ? <Models page={page} />
      : null}
  </DesktopShell>
}
function DesktopShell({ page, navigate, children }: { page: Page; navigate?: (page: Page) => void; children: ReactNode }) {
  const platform = window.__magnitudeDesktop.platform
  const [collapsed, setCollapsed] = useState(false)
  const integratedControls = platform === "darwin" || platform === "win32"
  const sidebarWidth = collapsed ? 0 : 224
  return <div className="relative flex h-screen bg-slate-50 font-sans text-slate-900 dark:bg-slate-925 dark:text-slate-200">
    {platform === "win32" && <div aria-hidden="true" data-window-drag-region style={{ left: sidebarWidth }} className="absolute right-0 top-0 z-50 h-8 select-none transition-[left] duration-250 ease-in-out motion-reduce:transition-none [-webkit-app-region:drag]" />}
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
        {(Object.keys(pageNames) as Page[]).map(key => {
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

const root = createRoot(document.getElementById("root")!)
root.render(<DesktopShell page="discover"><ModelsSkeleton page="discover" /></DesktopShell>)
const boot = Effect.gen(function* () {
  const initial = yield* observation.pipe(Stream.take(1), Stream.runHead, Effect.flatMap(value => value._tag === "Some" ? Effect.succeed(value.value) : Effect.fail(new DesktopHostUnavailable())))
  const scope = yield* Scope.make()
  const updateNativeTheme = () => { void host.appearance(getAppearancePreference()).catch(console.error) }
  updateNativeTheme()
  const unsubscribeAppearance = subscribeAppearance(updateNativeTheme)
  yield* Scope.addFinalizer(scope, Effect.sync(unsubscribeAppearance))
  const runtime = yield* Effect.runtime<never>()
  window.addEventListener("beforeunload", () => { Runtime.runFork(runtime)(Scope.close(scope, Exit.void)) }, { once: true })
  const connection = yield* makeFirstPartyConnection(MagnitudeClient.layer({ origin: initial.endpoint, autoStart: false }).pipe(Layer.provide(FetchHttpClient.layer))).pipe(Effect.provideService(Scope.Scope, scope))
  const client = createAgentClient(connection.client, { desktopBridge: {
    loginStartup: Stream.asyncPush(emit => Effect.acquireRelease(Effect.sync(() => host.loginStartup(state => {
      const decoded = Schema.decodeUnknownEither(LoginStartupState)(state)
      if (decoded._tag === "Right") emit.single(decoded.right)
      else emit.fail(new DesktopHostFailed({ message: String(decoded.left) }))
    }, message => emit.fail(new DesktopHostFailed({ message })))), unsubscribe => Effect.sync(unsubscribe)).pipe(Effect.asVoid)),
    memory: Stream.asyncPush(emit => Effect.acquireRelease(Effect.sync(() => host.memory(value => emit.single(value), message => emit.fail(new DesktopHostFailed({ message })))), unsubscribe => Effect.sync(unsubscribe)).pipe(Effect.asVoid)),
    machineIdentity: Effect.tryPromise(() => host.machineIdentity()),
    applicationInfo: Effect.tryPromise(() => host.applicationInfo()).pipe(Effect.flatMap(Schema.decodeUnknown(DesktopApplicationInfo))),
    updates: Stream.asyncPush(emit => Effect.acquireRelease(Effect.sync(() => host.updates(state => {
      const decoded = Schema.decodeUnknownEither(DesktopUpdateState)(state)
      if (decoded._tag === "Right") emit.single(decoded.right)
      else emit.fail(new DesktopHostFailed({ message: String(decoded.left) }))
    }, message => emit.fail(new DesktopHostFailed({ message })))), unsubscribe => Effect.sync(unsubscribe)).pipe(Effect.asVoid)),
    setAutoDownload: enabled => hostCommand(() => host.setAutoDownload(enabled)),
    checkUpdate: hostCommand(() => host.checkUpdate()),
    discardUpdate: hostCommand(() => host.discardUpdate()),
    downloadUpdate: hostCommand(() => host.downloadUpdate()),
    restartUpdate: hostCommand(() => host.restartUpdate()),
    setLoginStartup: enabled => hostCommand(() => host.setLoginStartup(enabled)),
    connections: Stream.asyncPush(emit => Effect.acquireRelease(Effect.sync(() => host.connections(rows => {
      const decoded = Schema.decodeUnknownEither(DesktopConnectionsSnapshot)(rows)
      if (decoded._tag === "Right") emit.single(decoded.right)
      else emit.fail(new DesktopHostFailed({ message: String(decoded.left) }))
    }, message => emit.fail(new DesktopHostFailed({ message })))), unsubscribe => Effect.sync(unsubscribe)).pipe(Effect.asVoid)),
    connect: input => hostCommand(() => host.connect(Schema.encodeSync(DesktopConnectRequest)(input))),
    disconnect: harness => hostCommand(() => host.disconnect(harness)),
    actions: Stream.asyncPush(emit => Effect.acquireRelease(Effect.sync(() => host.actions(action => emit.single(action))), unsubscribe => Effect.sync(unsubscribe)).pipe(Effect.asVoid)),
    presentModel: value => Effect.tryPromise(() => host.presentModel(value)),
  } })
  root.render(<RegistryProvider><AgentClientProvider tag={client}><App /></AgentClientProvider></RegistryProvider>)
})
Effect.runPromise(boot).catch(error => root.render(<p role="alert">Unable to open Magnitude: {String(error)}</p>))
