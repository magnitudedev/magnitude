import { ModelPreferenceSlider } from "./model-preference-slider"
import { ServingUsage } from "./serving-usage"
import { initializeAppearance, setAppearancePreference, useAppearancePreference, subscribeAppearance, getAppearancePreference } from "../../web/src/stores/appearance-store"
import { Button } from "../../web/src/components/ui/button"
import { Input } from "../../web/src/components/ui/input"
import { Progress } from "../../web/src/components/ui/progress"
import { MagnitudeMark } from "../../web/src/components/magnitude-mark"
import { Layers3, Library, HardDrive, Plug, Activity, BarChart3, Check, Settings2, Download, Play, Square, Trash2, X, Monitor, Sun, Moon } from "lucide-react"
import { createRoot } from "react-dom/client"
import { useMemo, useState } from "react"
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
import { HarnessLogo } from "./harness-logo"
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
function ModelDetails({ model, radar = false }: { model: CatalogLocalModel; radar?: boolean }) {
  const serving = model.servingState
  return <details className="mt-4 text-sm">
    <summary className="cursor-pointer rounded py-1 text-slate-600 focus-visible:outline-2 focus-visible:outline-blue-500 dark:text-slate-300">Model details</summary>
    <div className="mt-3 space-y-3 border-t border-slate-200 pt-4 dark:border-slate-750">
      {radar && <div className="max-w-sm"><ModelRadar model={model} /></div>}
      <p>License: {Option.getOrElse(model.presentation.license, () => "Not supplied by the catalog")}</p>
      {serving._tag === "Assessed" && <>
        <p>Context: {serving.assessment.profile.contextLength.toLocaleString()} tokens</p>
        <p>Capabilities: {[
          serving.capabilities.tools && "Tool use", serving.capabilities.vision && "Vision",
          serving.capabilities.structuredOutput && "Structured output", serving.capabilities.reasoning.supported && "Reasoning",
        ].filter(Boolean).join(" · ") || "Text generation"}</p>
        {serving.assessment._tag === "Fits" && serving.assessment.performance.length > 0 && <div>
          <p className="font-medium">Estimated generation speed</p>
          <ul className="mt-1 space-y-1 text-slate-500">{serving.assessment.performance.map(sample => <li key={sample.contextTokens}>{Math.round(sample.estimatedTokensPerSecond)} tokens/s at {sample.contextTokens.toLocaleString()} context tokens</li>)}</ul>
          <p className="mt-2 text-xs text-slate-500">Estimates from your machine’s assessment; actual speed varies with workload.</p>
        </div>}
      </>}
      {model.presentation.sourceUrls.length > 0 && <div className="flex flex-col items-start gap-2"><p className="font-medium">Sources</p>{model.presentation.sourceUrls.map(url => <a className="break-all text-blue-700 underline underline-offset-2 dark:text-blue-400" key={url} href={url} target="_blank" rel="noreferrer">{url}</a>)}</div>}
    </div>
  </details>
}
function DownloadProgress({ acquisition }: { acquisition: CatalogLocalModel["acquisitionState"] }) {
  if (acquisition._tag !== "Installing" && acquisition._tag !== "Updating") return null
  return <div className="mt-4"><p className="mb-2 text-sm">{acquisition.progress.stage === "downloading" ? `${formatStorageSize(acquisition.progress.completedBytes)} of ${formatStorageSize(acquisition.progress.totalBytes)}` : acquisition.progress.stage.replaceAll("_", " ")}</p><Progress aria-label="Download progress" indicatorClassName="bg-blue-700 dark:bg-blue-500" value={acquisition.progress.totalBytes ? acquisition.progress.completedBytes / acquisition.progress.totalBytes * 100 : null} /></div>
}
function ModelCard({ model, recommended = false, visual = false, replacing }: { model: CatalogLocalModel; visual?: boolean; recommended?: boolean; replacing?: string }) {
  const { install, load, stop, cancel, remove, dismissFailure: dismiss } = useLocalModelMutations()
  const command = useLocalModelCommandStatus(model.modelId)
  const stopping = useLocalModelStopStatus()
  const pending = command.pending || stopping.pending || model.acquisitionState._tag === "Removing"
  const acquisition = model.acquisitionState
  const installed = "residencyState" in acquisition
  const residency = installed ? acquisition.residencyState : undefined
  const transferring = acquisition._tag === "Installing" || acquisition._tag === "Updating"
  return <article className={`relative rounded-2xl border bg-white p-7 dark:bg-slate-850 ${recommended ? "border-blue-300 shadow-sm dark:border-blue-700" : "border-slate-200 dark:border-slate-750"}`}>
    <div className={visual ? "grid items-center gap-8 lg:grid-cols-[minmax(0,1fr)_16rem]" : ""}>
    <div>
    {visual && <p className="mb-3 text-xs font-medium uppercase tracking-wider text-blue-700 dark:text-blue-400">{recommended ? "Best match" : "Also worth a look"}</p>}
    <div className="flex items-start justify-between gap-4"><div className="flex min-w-0 items-center gap-4"><ModelLogo model={model} /><div><h2 className="text-lg font-semibold">{formatLocalModelDisplayName(model)}</h2><p className="mt-1 text-sm text-slate-500">Download size: {formatStorageSize(model.storageBytes)}</p></div></div><span className="text-sm text-slate-500">{acquisition._tag === "Removing" ? "Removing…" : acquisition._tag === "RemoveFailed" ? "Removal failed" : residency?._tag === "Ready" ? "Loaded" : acquisition._tag === "NotInstalled" ? "" : residency?._tag === "Unloaded" ? "Downloaded" : residency?._tag ?? acquisition._tag}</span></div>
    <p className="mt-4 max-w-xl text-sm leading-6 text-slate-500">{model.presentation.description}</p>
    <ModelFit model={model} />
    <DownloadProgress acquisition={acquisition} />
    {"failure" in acquisition && <p role="alert" className="mt-3 text-sm">{acquisition._tag === "InstallFailed" || acquisition._tag === "UpdateFailed" ? modelDownloadFailureMessage(acquisition.failure) : acquisition.failure.message}</p>}
    {residency?._tag === "Failed" && <p role="alert" className="mt-3 text-sm">{residency.failure.message}</p>}
    {command.failures.map(message => <p key={message} role="alert" className="mt-3 text-sm">{message}</p>)}
    {residency?._tag === "Stopping" && Option.isSome(stopping.failure) && <p role="alert" className="mt-3 text-sm">{stopping.failure.value}</p>}
    <div className="mt-5 flex flex-wrap gap-2">
      {transferring ? <Button variant="outline" onClick={() => cancel(model.modelId)}><X />Cancel download</Button> : !installed ? <Button disabled={pending || model.servingState._tag !== "Assessed" || model.servingState.assessment._tag !== "Fits"} onClick={() => { install(model.modelId) }}><Download />Download</Button> : <>
        <Button disabled={pending || residency?._tag === "Ready" || residency?._tag === "Loading" || residency?._tag === "Requested" || residency?._tag === "Stopping"} onClick={() => { if (!replacing || window.confirm(`Loading ${formatLocalModelDisplayName(model)} will stop ${replacing}. Continue?`)) { load(model.modelId) } }}><Play />Load model</Button>
        <Button variant="outline" disabled={stopping.pending || residency?._tag === "Unloaded" || residency?._tag === "Failed"} onClick={() => stop()}><Square />Stop model</Button>
        <Button variant="outline" disabled={pending} onClick={() => { if (window.confirm(`Remove the downloaded files for ${formatLocalModelDisplayName(model)}?`)) remove(model.modelId) }}><Trash2 />Remove</Button>
        {(acquisition._tag === "UpdateAvailable" || acquisition._tag === "UpdateFailed") && <Button variant="outline" disabled={pending} onClick={() => install(model.modelId)}>Update</Button>}
      </>}
      {(acquisition._tag === "InstallFailed" || acquisition._tag === "UpdateFailed") && <Button variant="outline" onClick={() => dismiss(model.modelId)}>Dismiss error</Button>}
    </div>
    </div>
    {visual && <ModelRadar model={model} />}
    </div>
    <ModelDetails model={model} radar={!visual} />
  </article>
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
    {discover && <section aria-label="Top recommendations" className="mb-10"><div className="mb-5 flex items-baseline justify-between"><h2 className="font-heading text-xl">Your top picks</h2><span className="text-xs text-slate-500">Best configuration for each model</span></div><div className="flex flex-col gap-6">{featuredCatalogModels(ranked).map((model,index)=><ModelCard key={model.modelId} model={model} visual recommended={index===0} {...(active&&active.model.modelId!==model.modelId?{replacing:formatLocalModelDisplayName(active.model)}:{})} />)}</div><p className="mt-4 text-xs text-slate-500">Profiles use catalog scores and your machine’s assessment. Speed is estimated; memory shows footprint, not free memory.</p></section>}
    {!discover && <>
    <div className="mb-5 mt-8 flex flex-wrap items-center justify-between gap-4"><h2 className="font-heading text-xl">{installedOnly?"Your library":"Explore the catalog"}</h2><Input aria-label="Search models" placeholder="Find a model…" className="max-w-sm" value={search} onChange={event=>setSearch(event.target.value)} /></div>
    {!installedOnly && <div className="mb-5 flex items-center justify-between text-sm text-slate-500"><label className="flex items-center gap-2"><input type="checkbox" checked={fitOnly} onChange={event => setFitOnly(event.target.checked)} className="accent-blue-600" />Fits my machine</label><span>{visible.length} configurations</span></div>}
    <div className="grid items-start gap-5 xl:grid-cols-2">{visible.map(model => <ModelCard key={model.modelId} model={model} {...(active && active.model.modelId !== model.modelId ? { replacing: formatLocalModelDisplayName(active.model) } : {})} />)}</div>
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
      : <div className="mt-7 grid items-start gap-5 xl:grid-cols-2">{rows.value.connections.map(row => <article key={row.id} className="rounded-xl border border-slate-200 bg-white p-5 dark:border-slate-750 dark:bg-slate-850">
        <div className="flex flex-wrap items-center justify-between gap-5"><div className="flex items-center gap-4"><HarnessLogo id={row.id} name={row.name} /><div><h2 className="text-lg font-semibold">{row.name}</h2><p className="mt-1 text-sm text-slate-500">{row.inspection._tag === "Connected" ? "Connected" : row.inspection._tag === "Unavailable" ? "Could not verify" : "Disconnected"}{!row.installed ? " · Harness not installed" : ""}</p></div></div>
          <div className="flex gap-2"><Button disabled={busy || !row.installed || !canConnect} onClick={() => connect({ harness: row.id, model: selectedModel })}>{row.inspection._tag === "Connected" ? "Reconnect" : row.managed ? "Repair connection" : "Connect"}</Button>{row.managed && <Button variant="outline" disabled={busy} onClick={() => disconnect(row.id)}>Disconnect</Button>}</div>
        </div>
        {row.inspection._tag !== "Connected" && <p className="mt-3 text-sm text-slate-500">{row.inspection.reason}</p>}
        {row.plugin._tag === "Some" && <p className="mt-3 text-sm text-slate-500">Includes {row.plugin.value.name}.</p>}
        <details className="mt-4 text-sm text-slate-500"><summary className="cursor-pointer">Configuration files</summary><ul className="mt-2 space-y-1">{row.configurationFiles.map(file => <li key={file} className="break-all font-mono text-xs">{file}</li>)}</ul></details>
      </article>)}</div>}
  </>
}
function ModelStatus() {
  const models = useLocalModels()
  const { stop } = useLocalModelMutations()
  const stopping = useLocalModelStopStatus()
  const presentation = Result.isSuccess(models) ? modelTrayPresentation(models.value) : null
  const active = Result.isSuccess(models) ? Option.getOrUndefined(activeLocalModel(models.value)) : undefined
  return <section className="rounded-xl border border-slate-200 bg-white p-6 dark:border-slate-750 dark:bg-slate-850">
    <h2 className="font-heading text-lg">Model activity</h2>
    <div className="mt-3 flex items-center gap-3">{active && <ModelLogo model={active.model} className="size-7" />}<p>{presentation?.label ?? (Result.isFailure(models) ? "Model status unavailable" : "Reading model status…")}</p></div>
    {active?.residency._tag === "Loading" && <div className="mt-4"><Progress aria-label="Model loading progress" indicatorClassName="bg-blue-700 dark:bg-blue-500" value={Option.match(active.residency.progress, { onNone: () => null, onSome: fraction => fraction * 100 })} /></div>}
    {Result.isFailure(models) && <p role="alert" className="mt-2 text-sm text-slate-500">{localModelFailureMessage(models.cause, "Could not read model status. Check Status and try again.")}</p>}
    {presentation?.canStop && <Button className="mt-4" variant="outline" disabled={stopping.pending} onClick={() => stop()}><Square />Stop model</Button>}
    {Option.isSome(stopping.failure) && <p role="alert" className="mt-2 text-sm">{stopping.failure.value}</p>}
  </section>
}
function DownloadActivity() {
  const models = useLocalModels()
  const active = Result.isSuccess(models) ? models.value.models.filter((model): model is CatalogLocalModel => model._tag === "Catalog" && (model.acquisitionState._tag === "Installing" || model.acquisitionState._tag === "Updating" || model.acquisitionState._tag === "Removing")) : []
  return <section className="rounded-xl border border-slate-200 bg-white p-6 dark:border-slate-750 dark:bg-slate-850">
    <h2 className="font-heading text-lg">Downloads</h2>
    {!Result.isSuccess(models) ? <p className="mt-3 text-sm text-slate-500">{Result.isFailure(models) ? "Download activity unavailable" : "Reading download activity…"}</p> : active.length === 0 ? <p className="mt-3 text-sm text-slate-500">No downloads in progress.</p> : <ul className="mt-4 space-y-5">{active.map(model => <li key={model.modelId}><div className="flex items-center gap-3"><ModelLogo model={model} className="size-6" /><p className="text-sm">{formatLocalModelDisplayName(model)} · {model.acquisitionState._tag === "Removing" ? "Removing files…" : model.acquisitionState._tag === "Updating" ? "Updating" : "Downloading"}</p></div><DownloadProgress acquisition={model.acquisitionState} /></li>)}</ul>}
  </section>
}
function Status({ snapshot }: { snapshot: typeof ApplicationSnapshot.Type | null }) {
  const service = snapshot?.service
  const ready=service?._tag === "Ready"
  return <div className="mt-7 space-y-6">
    <section className="relative overflow-hidden rounded-2xl border border-blue-200 bg-gradient-to-br from-blue-50 to-white p-7 dark:border-slate-700 dark:from-slate-800 dark:to-slate-900">
      <div className="flex items-center gap-5"><div className={`flex size-16 items-center justify-center rounded-full border bg-white dark:bg-slate-900 ${ready ? "border-green-300 text-green-600 dark:border-green-800 dark:text-green-400" : "border-blue-300 text-blue-600 dark:border-blue-800 dark:text-blue-400"}`}>{ready ? <Check aria-label="Service ready" className="size-7 text-green-600 dark:text-green-400" /> : <Activity className="size-7" />}</div><div><p className="text-xs font-medium uppercase tracking-widest text-slate-500">Magnitude service</p><h2 className="mt-2 font-heading text-2xl">{ready ? "Ready when you are" : service?._tag === "CleanupFailed" ? "Cleanup needs attention" : service?._tag === "Failed" ? "Let’s get you running" : "Starting your local engine"}</h2><p className="mt-2 text-sm text-slate-500">{ready ? "Your service is running. Model activity is shown separately below." : service?._tag ?? "Connecting"}</p></div></div>
      {service && "message" in service && <p role="alert" className="mt-5 text-sm">{service.message}</p>}
      {service?._tag === "Failed" && <Button className="mt-5" variant="outline" onClick={() => host.retry()}>Retry service</Button>}
    </section>
    <MemoryBreakdown />
    <div className="grid items-start gap-5 lg:grid-cols-2">
      {ready ? <ModelStatus /> : <section className="rounded-2xl border border-slate-200 bg-white p-6 dark:border-slate-750 dark:bg-slate-850"><h2 className="font-heading text-lg">Model activity</h2><p className="mt-4 text-sm text-slate-500">Model status will return when the service is ready.</p></section>}
      {ready && <DownloadActivity />}
    </div>
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
