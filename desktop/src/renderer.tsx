import { ModelPreferenceSlider } from "./model-preference-slider"
import { ServingUsage } from "./serving-usage"
import { initializeAppearance, setAppearancePreference, useAppearancePreference, subscribeAppearance, getAppearancePreference } from "../../web/src/stores/appearance-store"
import { ActionTooltip, TooltipProvider } from "../../web/src/components/ui/tooltip"
import { Button } from "../../web/src/components/ui/button"
import { Input } from "../../web/src/components/ui/input"
import { Progress } from "../../web/src/components/ui/progress"
import { MagnitudeMark } from "../../web/src/components/magnitude-mark"
import { ChevronDown, Eye, ArrowUpRight, Layers3, Library, HardDrive, Plug, Activity, BarChart3, Check, Settings2, Download, Play, Square, Trash2, X, Monitor, Sun, Moon } from "lucide-react"
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
  formatStorageSize, formatMemorySize, localModelIsInstalled, localModelProviderModelId, rankedLocalModelOptions, featuredCatalogModels, targetPhysicalMemoryBytes,
  LOCAL_MODEL_RANKING_SCALE_LABELS, LOCAL_MODEL_RANKING_SCALE_VALUES,
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
const pageIcons = { discover: Layers3, catalog: Library, models: HardDrive, connections: Plug, usage: BarChart3, status: Activity, settings: Settings2 }

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
          {serving._tag === "Assessed" && serving.capabilities.vision && <div className="self-end"><TooltipProvider><ActionTooltip label="Supports vision" trigger={<button type="button" aria-label="Supports vision" className="rounded p-1 text-slate-500 hover:text-slate-800 focus-visible:outline-2 focus-visible:outline-blue-500 dark:hover:text-slate-200"><Eye aria-hidden="true" className="size-4" /></button>} /></TooltipProvider></div>}
        </dl>
        {model.presentation.sourceUrls.length > 0 && <div><p className="mb-2 text-xs text-slate-500">Sources</p><div className="flex flex-wrap gap-x-4 gap-y-2">{model.presentation.sourceUrls.map(url => {
          const source = new URL(url)
          const label = source.hostname === "huggingface.co" ? `Hugging Face · ${source.pathname.split("/")[1]}` : source.hostname.replace(/^www\./, "")
          return <a className="inline-flex items-center gap-1 text-sm text-slate-600 hover:underline dark:text-slate-300" key={url} href={url} title={url} target="_blank" rel="noreferrer">{label}<ArrowUpRight aria-hidden="true" className="size-3.5" /></a>
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
    <summary className="flex cursor-pointer list-none items-center justify-end gap-1 rounded py-1 text-slate-600 focus-visible:outline-2 focus-visible:outline-blue-500 dark:text-slate-300 [&::-webkit-details-marker]:hidden">Model details<ChevronDown aria-hidden="true" className="size-4 group-open:rotate-180" /></summary>
    {content}
  </details>
}
function DownloadProgress({ acquisition }: { acquisition: CatalogLocalModel["acquisitionState"] }) {
  if (acquisition._tag !== "Installing" && acquisition._tag !== "Updating") return null
  return <div className="mt-4"><p className="mb-2 text-sm">{acquisition.progress.stage === "downloading" ? `${formatStorageSize(acquisition.progress.completedBytes)} of ${formatStorageSize(acquisition.progress.totalBytes)}` : acquisition.progress.stage.replaceAll("_", " ")}</p><Progress aria-label="Download progress" indicatorClassName="bg-blue-700 dark:bg-blue-500" value={acquisition.progress.totalBytes ? acquisition.progress.completedBytes / acquisition.progress.totalBytes * 100 : null} /></div>
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
      {transferring ? <Button variant="outline" onClick={() => cancel(model.modelId)}><X />Cancel download</Button> : !installed ? <Button disabled={pending || model.servingState._tag !== "Assessed" || model.servingState.assessment._tag !== "Fits"} onClick={() => { install(model.modelId) }}><Download />Download ({formatStorageSize(model.storageBytes).replace(/\s/g, "")})</Button> : <>
        {canStop ? <Button className="min-w-28" variant="outline" disabled={stopping.pending} onClick={() => stop()}><Square />Stop model</Button> : <Button className="min-w-28" disabled={pending} onClick={() => { if (!replacing || window.confirm(`Loading ${formatLocalModelDisplayName(model)} will stop ${replacing}. Continue?`)) load(model.modelId) }}><Play />Load model</Button>}
        <Button variant="ghost" size="icon" aria-label={`Remove ${formatLocalModelDisplayName(model)}`} title="Remove download" disabled={pending} onClick={() => { if (window.confirm(`Remove the downloaded files for ${formatLocalModelDisplayName(model)}?`)) remove(model.modelId) }}><Trash2 /></Button>
        {(acquisition._tag === "UpdateAvailable" || acquisition._tag === "UpdateFailed") && <Button variant="outline" disabled={pending} onClick={() => install(model.modelId)}>Update</Button>}
      </>}
      {(acquisition._tag === "InstallFailed" || acquisition._tag === "UpdateFailed") && <Button variant="outline" onClick={() => dismiss(model.modelId)}>Dismiss error</Button>}
    </div>
    {(model.servingState._tag !== "Assessed" || model.servingState.assessment._tag !== "Fits") && <ModelFit model={model} />}
    <DownloadProgress acquisition={acquisition} />
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
  return <article className="rounded-2xl border border-slate-200 bg-white p-5 dark:border-slate-750 dark:bg-slate-850">
    <div className="grid items-center gap-4 lg:grid-cols-[minmax(0,1fr)_auto]">
      <div className="flex min-w-0 items-center gap-4"><ModelLogo model={model} /><div className="min-w-0"><h2 className="text-lg font-semibold">{formatLocalModelDisplayName(model)}</h2>{status}</div></div>
      <ModelControls model={model} {...(replacing ? { replacing } : {})}>
        <Button variant="ghost" aria-expanded={detailsOpen} aria-controls={detailsId} onClick={() => setDetailsOpen(value => !value)}>Details<ChevronDown aria-hidden="true" className={`size-4 ${detailsOpen ? "rotate-180" : ""}`} /></Button>
      </ModelControls>
    </div>
    <ModelDetails model={model} radar open={detailsOpen} contentId={detailsId} />
  </article>
}
function SelectedRecommendation({ model, active }: { model: CatalogLocalModel; active: ReturnType<typeof activeLocalModel> }) {
  const [view, setView] = useState<"profile" | "details">("profile")
  return <div className="min-w-0 border-t border-slate-200 p-5 dark:border-slate-750 lg:border-l lg:border-t-0" aria-label="Selected model profile">
    <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
    <div className="flex items-center gap-1" aria-label="Model information">
      <Button variant={view === "profile" ? "secondary" : "ghost"} aria-pressed={view === "profile"} onClick={() => setView("profile")}>Profile</Button>
      <Button variant={view === "details" ? "secondary" : "ghost"} aria-pressed={view === "details"} onClick={() => setView("details")}>Details</Button>
    </div>
      <ModelControls model={model} {...(Option.isSome(active) && active.value.model.modelId !== model.modelId ? { replacing: formatLocalModelDisplayName(active.value.model) } : {})} />
    </div>
    <div className="grid min-h-72">
      <div className={`col-start-1 row-start-1 min-w-0 ${view === "profile" ? "" : "invisible"}`} aria-hidden={view !== "profile"}><ModelRadar model={model} /></div>
      <div className={`col-start-1 row-start-1 min-w-0 ${view === "details" ? "" : "invisible"}`} aria-hidden={view !== "details"}><ModelDetails model={model} compact open /></div>
    </div>
  </div>
}
function Recommendations({ models, active }: { models: readonly CatalogLocalModel[]; active: ReturnType<typeof activeLocalModel> }) {
  const [selectedId, setSelectedId] = useState<CatalogLocalModel["modelId"] | null>(null)
  const selected = models.find(model => model.modelId === selectedId) ?? models[0]
  if (!selected) return null
  return <section aria-label="Top recommendations" className="mb-8">
    <div className="grid overflow-hidden rounded-2xl border border-slate-200 bg-white dark:border-slate-750 dark:bg-slate-850 lg:grid-cols-[minmax(0,0.9fr)_minmax(0,1.1fr)]">
      <div className="space-y-2 p-3" aria-label="Recommended models">{models.map((model, rank) => <button key={model.modelId} type="button" aria-pressed={model.modelId === selected.modelId} onClick={() => setSelectedId(model.modelId)} className={`flex min-h-16 w-full items-center gap-3 rounded-xl border px-3 py-4 text-left transition-colors focus-visible:outline-2 focus-visible:outline-blue-500 ${model.modelId === selected.modelId ? "border-blue-300 bg-blue-50 dark:border-blue-700 dark:bg-slate-800" : "border-transparent hover:bg-slate-100 dark:hover:bg-slate-800"}`}>
        <span className="w-4 shrink-0 text-sm tabular-nums text-slate-500">{rank + 1}</span>
        <ModelLogo model={model} className="size-7" />
        <span className="min-w-0 text-sm font-medium">{formatLocalModelDisplayName(model)}</span>
      </button>)}</div>
      <SelectedRecommendation key={selected.modelId} model={selected} active={active} />
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
  const [fitOnly, setFitOnly] = useState(false)
  const client = useAgentClient()
  const session = useMemo(() => client.runtime.atom(DesktopSession), [client])
  const preferenceAtom = useMemo(() => Atom.make(get => Result.map(get(session), service => get(service.rankingPreference))), [session])
  const preferenceResult = useAtomValue(preferenceAtom)
  const preference = Result.isSuccess(preferenceResult) ? preferenceResult.value : 2
  const setPreference = useAtomSet(useMemo(() => client.runtime.fn((index: number) => Effect.flatMap(DesktopSession, service => service.setRankingPreference(index))), [client]))
  if (Result.isFailure(catalog)) return <p role="alert">{localModelFailureMessage(catalog.cause, "Could not read the model catalog. Check Status and try again.")}</p>
  if (!Result.isSuccess(catalog)) return <p>Reading the curated catalog…</p>
  const models = catalog.value.models.filter((model): model is CatalogLocalModel => model._tag === "Catalog")
  const ranked = !installedOnly && Result.isSuccess(hardware) ? rankedLocalModelOptions(models.map(model => ({ id: model.modelId, kind: localModelIsInstalled(model) ? "stored" as const : "downloadable" as const, model })), { fastToSmart: LOCAL_MODEL_RANKING_SCALE_VALUES[preference]!, memoryBudgetBytes: targetPhysicalMemoryBytes(hardware.value) }, models.length).flatMap(option => option.model._tag === "Catalog" ? [option.model] : []) : []
  const rankedIds = new Set(ranked.map(model => model.modelId))
  const ordered = installedOnly ? models : [...ranked, ...models.filter(model => !rankedIds.has(model.modelId))]
  const visible = ordered.filter(model => (!installedOnly || model.acquisitionState._tag !== "NotInstalled") && (!fitOnly || model.servingState._tag === "Assessed" && model.servingState.assessment._tag === "Fits") && `${formatLocalModelDisplayName(model)} ${model.presentation.description}`.toLowerCase().includes(search.toLowerCase()))
  return <>
    <p className="mt-2 text-slate-500">{installedOnly ? "Your downloads and installed models, in one place." : discover ? "Your best local models, matched to your machine." : "Explore every model in the curated catalog."}</p>
    {discover && <HardwareOverview /> }
    {Option.isSome(stopResult.failure) && <p role="alert" className="mt-5 text-sm">{stopResult.failure.value}</p>}
    {discover && <section aria-label="Recommendation preference" className="my-6"><div className="flex items-end justify-between gap-4"><div><h2 className="font-heading text-xl">Find your balance</h2><p className="mt-2 text-sm text-slate-500">Quick responses or deeper thinking. Choose what matters to you.</p></div><span className="rounded-full bg-blue-50 px-4 py-2 text-sm font-medium text-blue-700 dark:bg-slate-800 dark:text-blue-400">{LOCAL_MODEL_RANKING_SCALE_LABELS[preference]}</span></div><ModelPreferenceSlider value={preference} onChange={setPreference} /></section>}
    {!catalog.value.preparation.assessment.complete && <p className="mb-4 text-sm text-slate-500">Assessing models · {catalog.value.preparation.assessment.settledModels} of {catalog.value.preparation.assessment.totalModels}</p>}
    {discover && <Recommendations key={preference} models={featuredCatalogModels(ranked, 5)} active={Option.fromNullable(active)} />}
    {!discover && <>
    <div className="mb-5 mt-8 flex flex-wrap items-center justify-between gap-4"><h2 className="font-heading text-xl">{installedOnly?"Your library":"Explore the catalog"}</h2><Input aria-label="Search models" placeholder="Find a model…" className="max-w-sm" value={search} onChange={event=>setSearch(event.target.value)} /></div>
    {!installedOnly && <div className="mb-5 flex items-center justify-between text-sm text-slate-500"><label className="flex items-center gap-2"><input type="checkbox" checked={fitOnly} onChange={event => setFitOnly(event.target.checked)} className="accent-blue-600" />Fits my machine</label><span>{visible.length} configurations</span></div>}
    <div className="grid items-start gap-5">{visible.map(model => <ModelCard key={model.modelId} model={model} showMemory={installedOnly} {...(active && active.model.modelId !== model.modelId ? { replacing: formatLocalModelDisplayName(active.model) } : {})} />)}</div>
    {visible.length === 0 && <p className="py-8 text-slate-500">{search ? "No matching models." : installedOnly ? "No models downloaded yet. Find one in Discover." : "No models match this filter."}</p>}
    </>}
    {discover && ranked.length === 0 && catalog.value.preparation.assessment.complete && <p className="py-8 text-slate-500">No fitting recommendations right now. Explore Catalog for compatibility details.</p>}
  </>
}
function Connections({ serviceReady, selectedModel }: { serviceReady: boolean; selectedModel: Option.Option<ProviderModelId> }) {
  const client = useAgentClient()
  const session = useMemo(() => client.runtime.atom(DesktopSession), [client])
  const service = useAtomValue(session)
  return Result.isSuccess(service) ? <ConnectionsView service={service.value} serviceReady={serviceReady} selectedModel={selectedModel} /> : <p className="mt-5">Preparing connections…</p>
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
    <p className="mt-2 text-slate-500">Use your local models in the tools you already work with. Connect installs the configuration and Magnitude instructions.</p>
    {!serviceReady && <p className="mt-5 text-sm text-slate-500">Configuration checks are available. Start the service from Status before connecting a harness.</p>}
    {serviceReady && !canConnect && <div className="mt-5 flex flex-wrap items-center gap-3 text-sm text-slate-500"><p>{Result.isFailure(models) ? "Model availability could not be checked." : !Result.isSuccess(models) ? "Checking available models…" : "Download a compatible model before connecting a harness. It doesn’t need to be loaded."}</p><Button variant="outline" onClick={() => discover()}>Discover models</Button></div>}
    {error && Result.isFailure(error) && <p role="alert" className="mt-5 text-sm">{hostFailureMessage(error.cause)}</p>}
    {Result.isFailure(rows) ? <p role="alert" className="mt-5">Could not check connections. {hostFailureMessage(rows.cause)}</p>
      : !Result.isSuccess(rows) ? <p className="mt-5">Checking harness configuration…</p>
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
  return <div className="mt-3">
    <div className="flex items-center gap-3">{active && <ModelLogo model={active.model} className="size-7" />}<p>{presentation?.label ?? (Result.isFailure(models) ? "Model status unavailable" : "Reading model status…")}</p></div>
    {active?.residency._tag === "Loading" && <div className="mt-4"><Progress aria-label="Model loading progress" indicatorClassName="bg-blue-700 dark:bg-blue-500" value={Option.match(active.residency.progress, { onNone: () => null, onSome: fraction => fraction * 100 })} /></div>}
    {Result.isFailure(models) && <p role="alert" className="mt-2 text-sm text-slate-500">{localModelFailureMessage(models.cause, "Could not read model status. Check Status and try again.")}</p>}
    {presentation?.canStop && <Button className="mt-4" variant="outline" disabled={stopping.pending} onClick={() => stop()}><Square />Stop model</Button>}
    {Option.isSome(stopping.failure) && <p role="alert" className="mt-2 text-sm">{stopping.failure.value}</p>}
  </div>
}
function DownloadActivity() {
  const models = useLocalModels()
  const active = Result.isSuccess(models) ? models.value.models.filter((model): model is CatalogLocalModel => model._tag === "Catalog" && (model.acquisitionState._tag === "Installing" || model.acquisitionState._tag === "Updating" || model.acquisitionState._tag === "Removing")) : []
  if (Result.isSuccess(models) && active.length === 0) return null
  return <div className="mt-5 border-t border-slate-200 pt-5 dark:border-slate-700">
    {!Result.isSuccess(models) ? <p className="mt-3 text-sm text-slate-500">{Result.isFailure(models) ? "Download activity unavailable" : "Reading download activity…"}</p> : <ul className="space-y-5">{active.map(model => <li key={model.modelId}><div className="flex items-center gap-3"><ModelLogo model={model} className="size-6" /><p className="text-sm">{formatLocalModelDisplayName(model)} · {model.acquisitionState._tag === "Removing" ? "Removing files…" : model.acquisitionState._tag === "Updating" ? "Updating" : "Downloading"}</p></div><DownloadProgress acquisition={model.acquisitionState} /></li>)}</ul>}
  </div>
}
function Status({ snapshot }: { snapshot: typeof ApplicationSnapshot.Type | null }) {
  const service = snapshot?.service
  const ready=service?._tag === "Ready"
  return <div className="mt-7 space-y-6">
    <section className="relative overflow-hidden rounded-2xl border border-blue-200 bg-gradient-to-br from-blue-50 to-white p-7 dark:border-slate-700 dark:from-slate-800 dark:to-slate-900">
      <div className="flex items-center gap-5"><div className={`flex size-16 shrink-0 items-center justify-center rounded-full border bg-white dark:bg-slate-900 ${ready ? "border-green-300 text-green-600 dark:border-green-800 dark:text-green-400" : "border-blue-300 text-blue-600 dark:border-blue-800 dark:text-blue-400"}`}>{ready ? <Check aria-label="Service ready" className="size-7 text-green-600 dark:text-green-400" /> : <Activity className="size-7" />}</div><div><p className="text-xs font-medium uppercase tracking-widest text-slate-500">Magnitude service</p><h2 className="mt-2 font-heading text-2xl">{ready ? "Ready when you are" : service?._tag === "CleanupFailed" ? "Cleanup needs attention" : service?._tag === "Failed" ? "Let’s get you running" : "Starting your local engine"}</h2>{ready ? <ModelStatus /> : <p className="mt-2 text-sm text-slate-500">{service?._tag ?? "Connecting"}</p>}</div></div>
      {ready && <DownloadActivity />}
      {service && "message" in service && <p role="alert" className="mt-5 text-sm">{service.message}</p>}
      {service?._tag === "Failed" && <Button className="mt-5" variant="outline" onClick={() => host.retry()}>Retry service</Button>}
    </section>
    <MemoryBreakdown />
    <section className="rounded-2xl border border-slate-200 bg-white p-6 dark:border-slate-750 dark:bg-slate-850"><div className="flex items-center gap-3"><Plug className="size-5 text-blue-600 dark:text-blue-400" /><h2 className="font-heading text-lg">Local connection</h2></div><p className="mt-2 text-sm text-slate-500">Your tools connect to Magnitude on this machine.</p><p className="mt-4 break-all rounded-lg bg-slate-50 p-4 font-mono text-sm dark:bg-slate-900">{snapshot?.endpoint ?? "Endpoint unavailable"}</p><div className="mt-5 flex items-start gap-3"><span className={`mt-1 size-2 shrink-0 rounded-full ${snapshot?.tray._tag === "Registered" ? "bg-blue-500" : "bg-slate-400"}`} /><div><p className="text-sm font-medium">Background activity</p><p className="mt-1 text-sm text-slate-500">{snapshot?.tray._tag === "Registered" ? "Magnitude keeps running when you close the window." : snapshot?.tray._tag === "Unavailable" ? snapshot.tray.message : snapshot?.tray._tag === "Closed" ? "Magnitude is quitting." : "Checking tray availability…"}</p></div></div></section>
  </div>
}
function ApplicationSettings() {
  const client = useAgentClient()
  const session = useMemo(() => client.runtime.atom(DesktopSession), [client])
  const service = useAtomValue(session)
  const info = useAtomValue(useMemo(() => Atom.make(get => Result.flatMap(get(session), value => get(value.applicationInfo))), [session]))
  return <section className="mt-6 rounded-lg border border-slate-300 bg-white px-5 py-5 dark:border-slate-750 dark:bg-slate-850">
    <h2 className="font-heading text-lg">About Magnitude</h2>
    <p className="mt-2 text-sm text-slate-500">{Result.isSuccess(info) ? `Version ${info.value.version}` : Result.isFailure(info) ? "Version unavailable" : "Reading app version…"}</p>
    {Result.isSuccess(service) && <UpdateSettingsView service={service.value} />}
  </section>
}
function UpdateSettingsView({ service }: { service: DesktopSession }) {
  const observation = useAtomValue(service.updates)
  const check = useAtomSet(service.checkUpdate)
  const download = useAtomSet(service.downloadUpdate)
  const restart = useAtomSet(service.restartUpdate)
  const checking = useAtomValue(service.checkUpdate)
  const downloading = useAtomValue(service.downloadUpdate)
  const restarting = useAtomValue(service.restartUpdate)
  const current = Result.isSuccess(observation) ? observation.value : null
  const pending = checking.waiting || downloading.waiting || restarting.waiting
  const message = !current ? Result.isFailure(observation) ? "Update status unavailable." : "Reading update status…"
    : current._tag === "Idle" ? "Check for a new version of Magnitude."
    : current._tag === "Checking" ? "Checking for updates…"
    : current._tag === "Current" ? "You’re up to date."
    : current._tag === "Available" ? `Version ${current.version} is available · ${formatStorageSize(current.bytes)}`
    : current._tag === "Downloading" ? `Downloading version ${current.version} · ${formatStorageSize(current.completed)} of ${formatStorageSize(current.total)}`
    : current._tag === "Staging" ? `Preparing version ${current.version}…`
    : current._tag === "Ready" ? `Version ${current.version} is ready. Restart now, or it will be applied when you quit Magnitude.`
    : current._tag === "Closed" ? "Magnitude is quitting…" : current.message
  return <div className="mt-5 border-t border-slate-200 pt-5 dark:border-slate-750">
    <h3 className="font-medium">Application updates</h3>
    <p className="mt-2 text-sm text-slate-500" role="status">{message}</p>
    <div className="mt-3 flex flex-wrap gap-2">
      {current && ["Idle", "Current", "Available", "Failed"].includes(current._tag) && <Button variant="outline" disabled={pending} onClick={() => check()}>Check for updates</Button>}
      {current?._tag === "Available" && <Button disabled={pending} onClick={() => download()}>Download update</Button>}
      {current?._tag === "Ready" && <Button disabled={pending} onClick={() => restart()}>Restart to update</Button>}
    </div>
    {current?._tag === "Ready" && <p className="mt-2 text-sm text-slate-500">Restarting stops the running model and service.</p>}
    {[checking, downloading, restarting].map((result, index) => Result.isFailure(result) ? <p key={index} role="alert" className="mt-2 text-sm">{hostFailureMessage(result.cause)}</p> : null)}
  </div>
}
function AppearanceSettings() {
  const appearance = useAppearancePreference()
  return <section aria-labelledby="appearance-heading" className="mt-8 overflow-hidden rounded-lg border border-slate-300 bg-white dark:border-slate-750 dark:bg-slate-850">
    <header className="border-b border-slate-200 px-5 py-4 dark:border-slate-800"><h2 id="appearance-heading" className="font-heading text-lg">Appearance</h2></header>
    <div className="flex flex-wrap items-center justify-between gap-6 px-5 py-5"><div><p className="font-medium">Theme</p><p className="mt-1 text-sm text-slate-500">Use your system appearance or choose a theme.</p></div>
      <div className="flex gap-2" role="group" aria-label="Theme">{(["system", "light", "dark"] as const).map(value => { const Icon = value === "system" ? Monitor : value === "light" ? Sun : Moon; return <Button key={value} variant={appearance === value ? "default" : "outline"} aria-pressed={appearance === value} onClick={() => setAppearancePreference(value)}><Icon />{value[0]!.toUpperCase() + value.slice(1)}</Button> })}</div>
    </div>
  </section>
}
function LoginSettings() {
  const client = useAgentClient()
  const session = useMemo(() => client.runtime.atom(DesktopSession), [client])
  const service = useAtomValue(session)
  return Result.isSuccess(service) ? <LoginSettingsView service={service.value} /> : null
}
function LoginSettingsView({ service }: { service: DesktopSession }) {
  const state = useAtomValue(service.loginStartup)
  const set = useAtomSet(service.setLoginStartup)
  const change = useAtomValue(service.setLoginStartup)
  const current = Result.isSuccess(state) ? state.value : null
  const enabled = current?._tag === "Enabled" || current?._tag === "RequiresApproval"
  return <section className="mt-6 rounded-lg border border-slate-300 bg-white px-5 py-5 dark:border-slate-750 dark:bg-slate-850">
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
  return <div className="flex h-screen bg-slate-50 font-sans text-slate-900 dark:bg-slate-925 dark:text-slate-200">
    <aside className="flex w-56 shrink-0 flex-col border-r border-slate-200 px-4 py-8 dark:border-slate-750"><div className="mb-10 flex items-center gap-3 px-3 font-heading text-base font-semibold"><MagnitudeMark className="h-8 w-8" />Magnitude</div><nav className="flex min-h-0 flex-1 flex-col gap-2">{(Object.keys(pageNames) as Page[]).map(key => <Button variant="ghost" key={key} onClick={() => navigate(key)} aria-current={page === key ? "page" : undefined} className={`h-10 justify-start gap-3 rounded-lg px-3 text-left text-sm font-medium ${key === "status" ? "mt-auto" : ""} ${page === key ? "bg-blue-50 text-blue-700 dark:bg-slate-800 dark:text-blue-400" : "hover:bg-slate-100 dark:hover:bg-slate-800"}`}>{(() => { const Icon = pageIcons[key]; return <Icon className="size-4" /> })()}{pageNames[key]}</Button>)}</nav></aside>
    <main key={page} className="min-w-0 flex-1 overflow-y-auto px-10 py-9"><h1 className="font-heading text-[28px] font-semibold tracking-tight">{pageNames[page]}</h1>
      {page === "status" ? <Status snapshot={Result.isSuccess(state) ? state.value : null} />
      : page === "usage" ? <ServingUsage />
      : page === "settings" ? <><AppearanceSettings /><LoginSettings /><ApplicationSettings /></>
      : page === "connections" ? <Connections serviceReady={service?._tag === "Ready"} selectedModel={Option.none()} />
      : service?._tag !== "Ready" ? <p className="mt-8">{service?._tag === "Failed" ? "The service needs attention. Open Status for details." : "Starting the Magnitude service…"}</p>
      : page === "discover" || page === "catalog" || page === "models" ? <Models page={page} />
      : null}
    </main>
  </div>
}
const root = createRoot(document.getElementById("root")!)
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
    checkUpdate: hostCommand(() => host.checkUpdate()),
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
