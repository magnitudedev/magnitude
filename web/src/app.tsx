import { ChatPage } from "./components/chat"
import { ErrorNotice, NoticeAction } from "./components/error-notice"
import { SkeletonLine, ConnectionsSkeleton } from "./components/page-skeletons"
import { MachineCardSkeleton, ModelsSkeleton } from "./components/models/skeletons"
import { Skeleton } from "./components/ui/skeleton"
import { pageLayout } from "./components/page-layout"
import { ServingUsage } from "./components/serving-usage"
import { setAppearancePreference, useAppearancePreference } from "./stores/appearance-store"
import { Button } from "./components/ui/button"
import { Switch } from "./components/ui/switch"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "./components/ui/select"
import { MagnitudeMark } from "./components/magnitude-mark"
import {
  ArrowLeftIcon,
  ChatCircleIcon,
  SidebarSimpleIcon,
  ArrowUpRightIcon,
  CubeIcon,
  PlugIcon,
  ChartBarIcon,
  GearSixIcon,
  HardDrivesIcon,
  SlidersIcon,
  WifiHighIcon,
  InfoIcon,
  XIcon,
  MonitorIcon,
  SunIcon,
  MoonIcon,
  FolderOpenIcon,
  BookOpenIcon,
  type Icon,
} from "@phosphor-icons/react"
import { CopyCommand } from "./components/copy-command"
import { useEffect, useMemo, useState, type ReactNode } from "react"
import { Atom, Result, useAtomValue, useAtomSet, useAtomMount } from "@effect-atom/atom-react"
import { Cause, Effect, Equivalence, Option } from "effect"
import { type LocalModelsState, type ProviderModelId } from "@magnitudedev/sdk"
import type { AppearancePreference } from "@magnitudedev/sdk/desktop-host"
import { ServerSettingsFailed, type NetworkAccessChange, type NetworkBind } from "@magnitudedev/sdk"
import { FolderPicker, choosePath, useNativeOpenPath } from "./components/folder-picker"
import { SettingsGroup, SettingsRow } from "./components/settings-row"
import {
  activeLocalModel,
  deriveHardwareRoles,
  useAgentClient,
  type ApplicationPage,
  type MainPage,
  useLocalModels,
  useLocalModelsSelector,
  useLocalInferenceHardware,
  formatStorageSize,
  localModelConnectionOptions,
} from "@magnitudedev/client-common"
import { appearanceReadError } from "./appearance"
import { useNavigate, useServerPlatform, useSession } from "./session"
import { useReconnectAttempt, useServiceObservation, type ServiceObservation, type ServiceView } from "./service-view"
import { signOut, useDisconnectWarning, useRemoteAccess, viewerPlatform } from "./remote-access"
import { useNarrowViewport } from "./lib/viewport"
import { AlertDialog, AlertDialogAction, AlertDialogCancel, AlertDialogContent, AlertDialogDescription, AlertDialogFooter, AlertDialogHeader, AlertDialogTitle } from "./components/ui/alert-dialog"
import { ActionTooltip } from "./components/ui/tooltip"
import { AppModal, ModalBody } from "./components/app-modal"
import { SettingsStatus } from "./components/settings-status"
import { HarnessConnections, type CommandModel } from "./components/harness-connections"
import { HarnessSetupList } from "./components/harness-setup"
import { OtherApps } from "./components/other-apps"
import { ModelLocationsRow, ModelLocationsView } from "./components/model-locations"
import { HuggingFaceRow } from "./components/hugging-face"
import { HostNotices } from "./components/host-notices"
import { ModelsScreen } from "./components/models/models-screen"
import { NoModelsYet } from "./components/models/no-models"
import { MachineCard } from "./components/models/parts"
import { machineView } from "./components/models/machine"
import { Explore } from "./components/models/explore"
import { WhatRuns } from "./components/models/explore-view"
import { ExploreModelsButton } from "./components/models/your-models"
import { OnboardingScreen } from "./components/models/onboarding-screen"
import { onThisComputer } from "./components/models/data"
import { onboardingCompletedHere, recordOnboardingCompleted } from "./components/models/onboarding-record"
import { useRecommendations } from "./components/models/recommendations"
import { ChatModelSettings } from "./components/chat/model-settings"
import { ChatTabStatus, SidebarChats } from "./components/chat/sidebar-list"
import { SidebarUpdatePill, SidebarUpdateProvider, automaticUpdates, useApplicationOwner, useApplicationUpdateActions, useApplicationUpdateObservation, useSidebarUpdate, type UpdateTransfer } from "./components/application-update"

const ALL_INTERFACES = "all"
const inferenceUrl = (address: string, port: number) => `http://${address}:${port}/inference/v1`
/** The exact command that moves an existing store; the root holds only hub/, locks/ and one JSON file. */
const moveModelsCommand = (platform: string, from: string, to: string) => platform === "win32"
  ? `robocopy "${from}" "${to}" /E /MOVE`
  : `mv "${from}/"* "${to}/"`

/** The computer Magnitude runs on: its photo, name, memory and free disk. */
function ThisMachine() {
  const client = useAgentClient()
  const hardware = useLocalInferenceHardware()
  const machine = useAtomValue(client.Configuration.GetServerMachine({})).result
  return Result.isSuccess(hardware) ? <MachineCard machine={machineView(hardware.value, Result.isSuccess(machine) ? machine.value.identity : null)} /> : <MachineCardSkeleton />
}

/**
 * The Models page: the user's models, or what is recommended for this machine until there are any, with Explore models
 * beside its title. Explore models takes the page's place with a back arrow; a Hugging Face link opens it on that
 * repository.
 */
function ModelsPage({ observation }: { observation: ServiceObservation }) {
  const session = useSession()
  const linked = useAtomValue(session.addModelRequest)
  const clearLinked = useAtomSet(useMemo(() => Atom.fn((_: void) => session.clearAddModelRequest), [session]))
  const hardware = useLocalInferenceHardware()
  const models = useLocalModels()
  const [exploring, setExploring] = useState(false)
  const [typed, setTyped] = useState("")
  const memory = Result.isSuccess(hardware) ? deriveHardwareRoles(hardware.value).servingMemory : "Device"
  const recommendations = useRecommendations(false, memory)
  const service = Result.isSuccess(observation.service) ? observation.service.value : null
  const ready = service?._tag === "Ready"
  const linkedQuery = Option.map(linked, request => request._tag === "HuggingFace" ? request.repository : request.query)
  const back = () => { if (Option.isSome(linked)) clearLinked(); setExploring(false) }
  const explore = (query: string) => { setTyped(query); setExploring(true) }
  if ((ready && Option.isSome(linkedQuery)) || exploring) return <>
    <div className="mb-5 flex min-w-0 items-center gap-2">
      <Button variant="ghost" size="icon-lg" aria-label="Back" className="-ml-2.5 shrink-0" onClick={back}><ArrowLeftIcon className="size-5" /></Button>
      <h1 className={`${pageLayout.pageTitle} min-w-0 flex-1 truncate`}>Explore models</h1>
      <div className="shrink-0"><WhatRuns families={recommendations.families} /></div>
    </div>
    <Explore query={Option.getOrElse(linkedQuery, () => typed)} memory={memory} families={recommendations.families}
      owned={new Set(Result.isSuccess(models) ? models.value.models.filter(onThisComputer).map(model => String(model.modelId)) : [])}
      setQuery={next => { if (Option.isSome(linked)) clearLinked(); setTyped(next) }} />
  </>
  return <>
    <div className="mb-5 flex items-center justify-between gap-3">
      <h1 className={pageLayout.pageTitle}>Models</h1>
      {ready ? <ExploreModelsButton onClick={() => explore("")} /> : <Skeleton aria-hidden="true" className="h-8 w-40" />}
    </div>
    <div>
      {ready ? <ModelsScreen onExplore={explore} empty={<NoModelsYet memory={memory} />} />
        : Result.isFailure(observation.service) ? <ErrorNotice title="Magnitude needs your attention" description="The inference service is unavailable." />
        : <ModelsSkeleton startingService />}
    </div>
  </>
}
function Connections({ serviceReady, selectedModel }: { serviceReady: boolean; selectedModel: Option.Option<ProviderModelId> }) {
  const service = useSession()
  const models = useLocalModels()
  const commandModels = Result.isSuccess(models) ? localModelConnectionOptions(models.value.models) : []
  const canConnect = serviceReady && commandModels.length > 0
  const navigate = useNavigate()
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
  const active = Result.isSuccess(models) ? Option.getOrUndefined(activeLocalModel(models.value))?.model.modelId : undefined
  const defaultModel = commandModels.find(model => model.id === active)?.id ?? commandModels[0]?.id
  // One rhythm for the page's blocks, whichever are shown: each is separated from the next by the same gap.
  return <div className="flex flex-col gap-6">
    {!serviceReady && <p className="text-sm text-slate-500">{desktop ? "Configuration checks are available. Check the service in Settings before connecting a harness." : "Check the service in Settings before setting up an agent."}</p>}
    {serviceReady && !canConnect && !Result.isInitial(models) && <ErrorNotice severity={Result.isFailure(models) ? "error" : "info"} title={Result.isFailure(models) ? "Couldn’t check available models" : "Download a model to connect an agent"} description={Result.isFailure(models) ? "Check the service in Settings." : "Choose a compatible model. It doesn’t need to be loaded."} actions={!Result.isFailure(models) && <Button variant="outline" size="sm" className="mt-1" onClick={() => navigate("models")}>Open Models</Button>} />}
    {desktop
      ? <DesktopHarnessConnections canConnect={canConnect} models={commandModels} defaultModel={defaultModel} platform={platform} selectedModel={selectedModel} />
      : Option.isSome(apiOrigin) && <HarnessSetupList defaultModel={defaultModel} origin={apiOrigin.value} remote={remote} platform={viewerPlatform()} />}
    {Option.isSome(apiOrigin) && <OtherApps origin={apiOrigin.value} model={defaultModel} platform={remote ? viewerPlatform() : platform} remote={remote} networkAccess={Result.isSuccess(network) && network.value.enabled} onOpenSettings={() => navigate("settings")} />}
  </div>
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
    {error && Result.isFailure(error) && <ErrorNotice title={Result.isFailure(disconnecting) ? "Couldn’t disconnect this agent" : "Couldn’t connect this agent"} description="Check the agent’s configuration before trying again. Some changes may not have completed." />}
    {Result.isFailure(rows) ? <ErrorNotice title="Couldn’t check your connections" description="Connection status is unavailable. Magnitude will check again automatically." />
      : !Result.isSuccess(rows) ? <ConnectionsSkeleton />
      : rows.value._tag === "Unavailable" ? <ErrorNotice title="Couldn’t check your connections" description="Magnitude can’t read the saved connection information. Check that its configuration is accessible." />
      : <HarnessConnections connections={rows.value.connections} busy={busy} canConnect={canConnect} models={models} defaultModel={defaultModel} platform={platform}
          onConnect={harness => connect({ harness, model: selectedModel, installSkill: true })} onDisconnect={harness => disconnect({ harness })} />}
  </>
}
const firstFailure = (results: ReadonlyArray<Result.Result<unknown, unknown>>) => results.find((result): result is Result.Failure<unknown, unknown> => Result.isFailure(result) && !result.waiting)
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
    control={current && <Switch aria-label="Launch at login" checked={enabled} disabled={current._tag === "Unavailable" || change.waiting} onCheckedChange={checked => set(checked)} />} />
}
function ModelStorageRow() {
  const client = useAgentClient()
  const settings = useAtomValue(client.Configuration.GetModelStorage({})).result
  const save = useAtomSet(client.Configuration.SetModelStorage)
  const saving = useAtomValue(client.Configuration.SetModelStorage)
  const [choosing, setChoosing] = useState(false)
  const busy = saving.waiting
  const current = Result.isSuccess(settings) ? settings.value : null
  const native = useNativeOpenPath()
  const chooseFolder = choosePath(native, { title: "Choose a folder for downloaded models", defaultPath: current ? Option.some(current.path) : Option.none() }, path => save({ path: Option.some(path) }), () => setChoosing(true))
  const failure = firstFailure([saving])
  const refusal = failure ? Option.flatMap(Cause.failureOption(failure.cause), error => error instanceof ServerSettingsFailed ? Option.some(error.message) : Option.none()) : Option.none<string>()
  return <>
    <FolderPicker open={choosing} onOpenChange={setChoosing} initialPath={current ? Option.some(current.path) : Option.none()}
      title="Choose a folder for downloaded models" description="Folders on the computer running Magnitude." confirmLabel="Use this folder"
      onChoose={path => save({ path: Option.some(path) })} />
    <SettingsRow label="Model storage" alert={!busy && failure ? <ErrorNotice title="The model folder wasn’t saved" description={Option.getOrElse(refusal, () => "Check that the folder is available and writable, then try again.")} /> : Result.isFailure(settings) ? <ErrorNotice title="Couldn’t read the model folder setting" description="The folder used by the running service has not been changed." /> : current && Option.isSome(current.warning) ? <ErrorNotice severity="warning" title="The saved model folder can’t be used" description="Magnitude is using the default folder instead. Choose a folder the Magnitude service can write to." /> : undefined}
      hint={current ? <span className="block truncate" title={current.path}>Current path is <span className="text-slate-700 dark:text-slate-300" data-testid="model-storage-path">{current.path}</span>{current.source === "Default" && " (default)"}</span>  : Result.isInitial(settings) ? <SkeletonLine className="h-4 text-xs" width="220px" /> : undefined}
      control={<>
        {(current?.source === "Configured" || (current && Option.isSome(current.warning))) && <Button size="sm" variant="ghost" disabled={busy} onClick={() => save({ path: Option.none() })}>Use default</Button>}
        <Button size="sm" variant="outline" disabled={!current || busy} onClick={chooseFolder}><FolderOpenIcon />Change…</Button>
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
  const reason = storagePending && networkPending ? "Magnitude is still using the previous model folder and network settings."
    : storagePending ? "Magnitude is still using the previous model folder." : "Magnitude is still using the previous network settings."
  return <div className="fixed bottom-4 right-4 z-50 w-96 max-w-[calc(100vw-2rem)] rounded-lg bg-white shadow-md dark:bg-slate-850">
    {disconnect.dialog}
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
  const disconnect = useDisconnectWarning()
  const updating = useAtomValue(client.Configuration.SetNetworkAccess)
  const regenerate = useAtomSet(client.Configuration.RegenerateNetworkApiKey)
  const regenerating = useAtomValue(client.Configuration.RegenerateNetworkApiKey)
  const current = Result.isSuccess(settings) ? settings.value : null
  // A container always serves on every interface with a key, set by its environment; Settings only shows it.
  const managed = current?.container === true
  const busy = updating.waiting || regenerating.waiting || managed
  const failure = firstFailure([updating, regenerating])
  const reachable = current?.enabled ? Option.getOrElse(current.bind, () => current.interfaces[0]?.address) : undefined
  return <>
    {disconnect.dialog}
    <SettingsRow label="Network access" hint={current ? <>{managed ? "Set by the container: it accepts connections on every interface with an API key. MAGNITUDE_NETWORK_KEY sets the key and MAGNITUDE_ALLOWED_HOSTS adds host names. " : "Let other devices on your network use Magnitude for inference. "}<a href={managed ? "https://docs.magnitude.dev/installation/docker" : "https://docs.magnitude.dev/remote-server"} target="_blank" rel="noreferrer" className="inline-flex items-center gap-0.5 font-medium text-slate-700 hover:underline dark:text-slate-300">{managed ? "Docker guide" : "Remote server guide"}<ArrowUpRightIcon aria-hidden="true" className="size-3" /></a></> : Result.isInitial(settings) ? <SkeletonLine className="h-4 text-xs" width="240px" /> : undefined}
      alert={!busy && failure ? <ErrorNotice title="Network settings weren’t saved" description="Your previous saved settings are still in use." /> : Result.isFailure(settings) ? <ErrorNotice title="Couldn’t read network settings" description="The running service’s network settings have not been changed." /> : current && Option.isSome(current.warning) ? <ErrorNotice severity="warning" title="The saved network address is invalid" description="All interfaces are selected. Choose an address below to save a valid setting." /> : undefined}
      control={current && <Switch aria-label="Network access" checked={current.enabled} disabled={busy} onCheckedChange={checked => checked ? update({ enabled: Option.some(true) }) : disconnect.warn("TurnOffNetworkAccess", () => update({ enabled: Option.some(false) }))} />} />
    {current?.enabled && <>
      {!managed && <SettingsRow nested label="Address" hint={current.interfaces.length === 0 ? "No network interfaces were found." : "Which of this computer's addresses accepts connections."}
        control={<Select items={[{ value: ALL_INTERFACES, label: "All interfaces" }, ...current.interfaces.map(entry => ({ value: entry.address, label: `${entry.address} (${entry.kind === "tailscale" ? "Tailscale" : entry.name})` }))]}
          value={Option.getOrElse(current.bind, () => ALL_INTERFACES)} onValueChange={value => value === ALL_INTERFACES || value === null ? update({ bind: Option.some<NetworkBind>({ _tag: "AllInterfaces" }) }) : disconnect.warn("ChangeAddress", () => update({ bind: Option.some<NetworkBind>({ _tag: "Address", address: String(value) }) }))}>
          <SelectTrigger aria-label="Network address" className="min-w-56"><SelectValue /></SelectTrigger>
          <SelectContent>{[<SelectItem key={ALL_INTERFACES} value={ALL_INTERFACES}>All interfaces</SelectItem>, ...current.interfaces.map(entry => <SelectItem key={entry.address} value={entry.address}>{entry.address} ({entry.kind === "tailscale" ? "Tailscale" : entry.name})</SelectItem>)]}</SelectContent>
        </Select>} />}
      <SettingsRow nested label="API key" hint={current.requireApiKey ? "Other devices must send this key as a Bearer token." : "Other devices can connect without a key. Only do this on a network you trust."}
        control={<>{!managed && <Button size="sm" variant="ghost" disabled={busy} onClick={() => disconnect.warn("RegenerateKey", () => regenerate({}))}>Regenerate</Button>}<span className="flex items-center gap-2 text-sm text-slate-600 dark:text-slate-400"><span aria-hidden="true">Require</span><Switch aria-label="Require API key" checked={current.requireApiKey} disabled={busy} onCheckedChange={checked => checked ? update({ requireApiKey: Option.some(true) }) : disconnect.warn("StopRequiringKey", () => update({ requireApiKey: Option.some(false) }))} /></span></>}>
        {Option.isSome(current.apiKey) && current.requireApiKey && <div className="mt-2"><CopyCommand command={current.apiKey.value} label="Copy API key" /></div>}
      </SettingsRow>
      {reachable && !managed && <SettingsRow nested label="Reachable at" hint={`Use this as the OpenAI-compatible base URL on other devices${current.requireApiKey ? ", with the API key above" : ""}. Anthropic-compatible apps use /inference/anthropic on the same address.`}>
        <div className="mt-2"><CopyCommand command={inferenceUrl(reachable, current.port)} label="Copy base URL" /></div>
      </SettingsRow>}
    </>}
  </>
}
function AutomaticUpdatesRow() {
  const client = useAgentClient()
  const { owner, supports, observation, snapshot } = useApplicationUpdateObservation()
  const saveAutoDownload = useAtomSet(client.Application.SetApplicationAutoDownload)
  const setAutoDownload = (enabled: boolean) => saveAutoDownload({ enabled })
  const saving = useAtomValue(client.Application.SetApplicationAutoDownload)
  if (Result.isSuccess(owner) && !supports("Updates")) return null
  const status = automaticUpdates(observation)
  const closed = snapshot?.transfer._tag === "Closed"
  return <SettingsRow label="Automatic updates"
    hint={status._tag === "Reading" ? <SkeletonLine className="h-4 text-xs" width="200px" />
      : status._tag === "Known" ? "Download updates in the background when they are available."
      : status._tag === "NotInThisBuild" ? `Automatic updates aren’t available in this build. ${status.reason}`
      : undefined}
    alert={Result.isFailure(saving) && !saving.waiting ? <ErrorNotice title="Update preferences weren’t saved" description="Your previous preference is still in use." /> : status._tag === "ReadFailed" ? <ErrorNotice title="Couldn’t read update preferences" description="Automatic downloads are unavailable until your preference can be read." />
      : status._tag === "SetupFailed" ? <ErrorNotice title="Automatic updates couldn’t be set up" description={status.reason} />
      : undefined}
    control={snapshot && <Switch aria-label="Automatic updates" checked={status._tag === "Known" && status.autoDownload} disabled={status._tag !== "Known" || saving.waiting || closed} onCheckedChange={checked => setAutoDownload(checked)} />} />
}
const failedUpdatePhrase = { verify: "couldn’t be verified", install: "failed to install", authorization: "wasn’t authorized to install", startup: "failed to start" } as const

function AboutRow() {
  const service = useSession()
  const client = useAgentClient()
  const health = useAtomValue(client.Connection.Health({})).result
  const version = Result.isSuccess(health) ? `Magnitude ${health.value.version}` : Result.isFailure(health) ? "Magnitude" : <SkeletonLine className="h-5 text-sm" width="120px" />
  const { observation, snapshot } = useApplicationUpdateObservation()
  const { check, discard, download, restart, checking, downloading, restarting, discarding, pending, dialog } = useApplicationUpdateActions()
  const current: UpdateTransfer | undefined = snapshot?.transfer
  const message = !current ? Result.isFailure(observation) ? "Update status unavailable." : "Reading update status…"
    : current._tag === "Idle" ? snapshot?.check._tag === "Succeeded" ? "You’re up to date." : "Checks for updates automatically."
    : current._tag === "Available" ? `Version ${current.version} is available · ${formatStorageSize(current.bytes)}${Option.match(current.failed, {
      onNone: () => "", onSome: failed => ` · It ${failedUpdatePhrase[failed]} on this computer, so it isn’t downloaded automatically.` })}`
    : current._tag === "Downloading" ? `Downloading version ${current.version} · ${formatStorageSize(current.completed)} of ${formatStorageSize(current.total)}`
    : current._tag === "Cancelling" ? "Stopping automatic download…"
    : current._tag === "Staging" ? `Preparing version ${current.version}…`
    : current._tag === "Ready" ? `Version ${current.version} is ready. Restarting stops the running model and service.`
    : current._tag === "ImageAvailable" ? `Version ${current.version} is available. Pull the new image and recreate the container.`
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
  return <>{dialog}<SettingsRow label={version} hint={Result.isInitial(observation) ? <SkeletonLine className="h-4 text-xs" width="160px" /> : message}
    children={current?._tag === "ImageAvailable" && <div className="mt-2"><CopyCommand command={`docker pull ${current.image}`} label="Copy pull command" /></div>}
    alert={hasFailure && !busy ? <ErrorNotice
      title={Result.isFailure(discarding) && !discarding.waiting ? "Couldn’t discard the update" : current?._tag === "InstallationFailed" ? "The update wasn’t completed" : checkFailed ? "Couldn’t check for updates" : "The update couldn’t finish"}
      description={installable ? "The prepared update is still available. Retry it, or discard its download."
        : snapshot?.check._tag === "Failed" ? snapshot.check.message
        : current?._tag === "Failed" ? current.message : "Check your connection before trying again."}
      actions={actions} /> : undefined}
    control={hasFailure ? busy ? <span className="text-xs text-slate-500">Working…</span> : null : <>
      {installable && <Button size="sm" variant="ghost" disabled={pending} onClick={() => discard()}>Discard download</Button>}
      {installable ? <Button size="sm" disabled={pending} onClick={() => restart()}>Restart to update</Button>
        : current?._tag === "Available" ? <Button size="sm" disabled={pending} onClick={() => download()}>Download update</Button>
        : current && !["Unavailable", "Closed"].includes(current._tag) ? <Button size="sm" variant="outline" disabled={busy} onClick={() => check()}>{busy ? "Checking…" : "Check for updates"}</Button>
        : null}
    </>} /></>
}
function SignOutRow() {
  const { origin } = useRemoteAccess()
  const action = useMemo(() => Atom.fn((_: void) => signOut(origin)), [origin])
  const run = useAtomSet(action)
  const signingOut = useAtomValue(action)
  return <SettingsRow label="Signed in from another device" hint={`This browser controls Magnitude at ${new URL(origin).host}.`}
    alert={Result.isFailure(signingOut) && !signingOut.waiting ? <ErrorNotice title="Couldn’t sign out" description="Check that this device can still reach Magnitude, then try again." /> : undefined}
    control={<Button size="sm" variant="outline" disabled={signingOut.waiting} onClick={() => run()}>Sign out</Button>} />
}
type SettingsTab = "general" | "chat" | "storage" | "network" | "usage" | "about"
const settingsTabs: ReadonlyArray<{ readonly key: SettingsTab; readonly label: string; readonly icon: Icon }> = [
  { key: "general", label: "General", icon: SlidersIcon },
  { key: "chat", label: "Chat", icon: ChatCircleIcon },
  { key: "storage", label: "Storage", icon: HardDrivesIcon },
  { key: "network", label: "Network", icon: WifiHighIcon },
  { key: "usage", label: "Usage", icon: ChartBarIcon },
  { key: "about", label: "About", icon: InfoIcon },
]
const settingsTabFor = (page: ApplicationPage): SettingsTab => page === "usage" ? page : "general"

/**
 * Settings in its modal: its sections listed on the left, or in a menu above them in a narrow window, and the chosen
 * one beside them. Model locations replaces the section, with a back arrow over it.
 */
function SettingsModal({ observation, initial, onClose }: { observation: ServiceObservation; initial: SettingsTab; onClose: () => void }) {
  // A fresh mount observes hand edits; writes refresh in their own Effect actions.
  const client = useAgentClient()
  const { remote } = useRemoteAccess()
  const { storagePending, networkPending } = useRestartPending()
  const narrow = useNarrowViewport()
  const [tab, setTab] = useState(initial)
  const [locations, setLocations] = useState(false)
  useAtomMount(useMemo(() => Atom.make(Effect.all([Atom.refresh(client.Configuration.GetModelStorage({})), Atom.refresh(client.Configuration.GetNetworkAccess({})), Atom.refresh(client.Models.ListModelLocations({}))], { discard: true })), [client]))
  const choose = (next: SettingsTab) => { setLocations(false); setTab(next) }
  return <AppModal title="Settings" onClose={onClose}>
    <div className={narrow ? "flex min-h-0 flex-1 flex-col" : "flex min-h-0 flex-1"}>
      {narrow
        ? <div className="shrink-0 border-b border-slate-200 px-6 py-3 dark:border-slate-750">
            <Select items={settingsTabs.map(({ key, label }) => ({ value: key, label }))} value={tab} onValueChange={next => { if (next) choose(next as SettingsTab) }}>
              <SelectTrigger aria-label="Settings section" className="w-full text-[13px]"><SelectValue /></SelectTrigger>
              <SelectContent>{settingsTabs.map(({ key, label }) => <SelectItem key={key} value={key}>{label}</SelectItem>)}</SelectContent>
            </Select>
          </div>
        : <nav aria-label="Settings sections" className="flex w-44 shrink-0 flex-col gap-0.5 border-r border-slate-200 p-3 dark:border-slate-750">
            {settingsTabs.map(({ key, label, icon: TabIcon }) =>
              <button key={key} type="button" aria-current={tab === key ? "page" : undefined} onClick={() => choose(key)}
                className={`flex h-8 items-center gap-2.5 rounded-md px-2.5 text-left text-[13px] font-medium ${tab === key ? "bg-slate-100 text-slate-900 dark:bg-slate-800 dark:text-slate-100" : "text-slate-600 hover:bg-slate-100 dark:text-slate-300 dark:hover:bg-slate-800"}`}>
                <TabIcon className="size-4 shrink-0" />{label}
              </button>)}
          </nav>}
      <div className="flex min-h-0 min-w-0 flex-1 flex-col">
        {locations ? <>
          <div className="flex shrink-0 items-center gap-2 border-b border-slate-200 px-6 py-3 dark:border-slate-750">
            <Button variant="ghost" size="icon-sm" aria-label="Back" className="-ml-2" onClick={() => setLocations(false)}><ArrowLeftIcon /></Button>
            <h2 className="truncate text-sm font-semibold">Model locations</h2>
          </div>
          <ModalBody><ModelLocationsView /></ModalBody>
        </> : <ModalBody>
          {tab === "general" && <><ThisMachine /><div className="mt-4" /><SettingsStatus observation={observation} /><SettingsGroup><ThemeRow /><LaunchAtLoginRow /><AutomaticUpdatesRow /><HuggingFaceRow /></SettingsGroup></>}
          {tab === "chat" && <ChatModelSettings />}
          {tab === "storage" && <SettingsGroup><ModelStorageRow /><ModelLocationsRow onManage={() => setLocations(true)} /></SettingsGroup>}
          {tab === "network" && <SettingsGroup><NetworkAccessRows /></SettingsGroup>}
          {tab === "usage" && <ServingUsage />}
          {tab === "about" && <><SettingsGroup><AboutRow /></SettingsGroup>{remote && <SettingsGroup label="This browser"><SignOutRow /></SettingsGroup>}</>}
          {(storagePending || networkPending) && <div aria-hidden="true" className="h-44" />}
        </ModalBody>}
      </div>
    </div>
  </AppModal>
}
export function App() {
  const session = useSession()
  const observation = useServiceObservation()
  const page = useAtomValue(session.page)
  const main = useAtomValue(session.mainPage)
  const visit = useAtomValue(session.visit)
  const known = useAtomValue(session.onboardingPending)
  useEffect(() => { if (Option.isSome(known)) recordOnboardingCompleted(!known.value) }, [known])
  const onboarding = Option.getOrElse(known, () => !onboardingCompletedHere())
  const navigate = useNavigate()
  const service = Result.isSuccess(observation.service) ? observation.service.value : null
  const platform = Option.getOrUndefined(Option.map(session.clientWindow, window => window.platform))
  return <><RestartRequiredToast /><HostNotices /><QuitFailureDialog />{onboarding && <OnboardingScreen />}<SidebarUpdateProvider><AppShell page={main} navigate={navigate} platform={platform}>
      {service?._tag === "Unreachable" ? <CannotReachMagnitude />
      : service?._tag === "Reconnecting" ? <ReconnectingToMagnitude />
      : service?._tag === "Failed" || service?._tag === "CleanupFailed" ? <MagnitudeCouldNotStart service={service} canRetry={observation.canRetry} onOpenStatus={() => navigate("settings")} />
      : main === "chat" ? <ChatPage />
      : main === "connections" ? <><h1 className={`${pageLayout.pageTitle} mb-6`}>Connections</h1><Connections serviceReady={service?._tag === "Ready"} selectedModel={Option.none()} /></>
      : <ModelsPage key={visit} observation={observation} />}
    </AppShell>
    {page !== main && <SettingsModal key={page} observation={observation} initial={settingsTabFor(page)} onClose={() => navigate(main)} />}
  </SidebarUpdateProvider></>
}
/** Shown while a browser's own retries run after its connection drops, as it does when Magnitude restarts. */
function ReconnectingToMagnitude() {
  return <ErrorNotice className="mt-8" severity="info" title="Reconnecting to Magnitude…"
    description="The connection to the Magnitude service was interrupted, as it is when Magnitude restarts." />
}
/**
 * In place of the page when the desktop's service failed to start: every request would fail the same way, so one
 * notice says why, with the first line of the failure (Settings → General shows all of it), and offers a retry.
 */
function MagnitudeCouldNotStart({ service, canRetry, onOpenStatus }: {
  service: Extract<ServiceView, { readonly _tag: "Failed" | "CleanupFailed" }>
  canRetry: boolean
  onOpenStatus: () => void
}) {
  const session = useSession()
  const retry = useAtomSet(session.retryService)
  const retrying = useAtomValue(session.retryService)
  const reason = service.message.split("\n").map(line => line.trim()).find(line => line !== "") ?? ""
  return <ErrorNotice className="mt-8"
    title={service._tag === "CleanupFailed" ? "Magnitude couldn’t confirm its service stopped" : "Magnitude’s service couldn’t start"}
    description={service._tag === "CleanupFailed" ? "Some background work may still be running. Quit Magnitude to retry cleanup." : reason.length > 240 ? `${reason.slice(0, 240)}…` : reason || undefined}
    actions={<>
      {service._tag === "Failed" && canRetry && <NoticeAction disabled={retrying.waiting} onClick={() => retry()}>{retrying.waiting ? "Retrying…" : "Retry"}</NoticeAction>}
      <NoticeAction onClick={onOpenStatus}>See status</NoticeAction>
    </>} />
}
/** A browser cannot restart the service; it keeps retrying in the background, and Reconnect tries at once. */
function CannotReachMagnitude() {
  const attempt = useReconnectAttempt()
  const reconnectAction = useMemo(() => Atom.fn((_: void) => attempt), [attempt])
  const reconnect = useAtomSet(reconnectAction)
  const reconnecting = useAtomValue(reconnectAction)
  return <ErrorNotice className="mt-8" title="Can’t reach Magnitude"
    description="This page can’t connect to the Magnitude service and keeps trying. Check that Magnitude is running on that computer and that this device can reach it."
    actions={<NoticeAction disabled={reconnecting.waiting} onClick={() => reconnect()}>Reconnect</NoticeAction>} />
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
export function AppShell({ page, navigate, platform, children }: { page: MainPage; navigate?: (page: ApplicationPage) => void; platform: string | undefined; children: ReactNode }) {
  const narrow = useNarrowViewport()
  const [open, setOpen] = useState(false)
  // Only the navigation chrome changes across `md`; the page stays mounted, so resizing keeps its state.
  return <div className={narrow ? "relative flex h-screen flex-col bg-slate-50 font-sans text-slate-900 dark:bg-slate-925 dark:text-slate-200" : "relative flex h-screen bg-slate-50 font-sans text-slate-900 dark:bg-slate-925 dark:text-slate-200"}>
    {narrow ? <NarrowBar open={open} onOpen={() => setOpen(true)} /> : <DockedSidebar page={page} navigate={navigate} platform={platform} />}
    <main key={page} className="min-w-0 flex-1 overflow-y-auto" inert={narrow && open}>
      <div data-page-content className={page === "chat" ? `mx-auto flex h-full w-full max-w-[min(100%,72rem)] flex-col px-4 pb-0 pt-5 md:px-10 ${platform === "win32" ? "md:pt-14" : "md:pt-9"}` : narrow ? "mx-auto w-full max-w-6xl px-4 pb-8 pt-5" : `mx-auto w-[calc(100vw-224px)] max-w-[min(100%,72rem)] px-10 pb-9 ${platform === "win32" ? "pt-14" : "pt-9"}`}>
        {children}
      </div>
    </main>
    {narrow && open && <NavigationDrawer page={page} navigate={navigate} onClose={() => setOpen(false)} />}
  </div>
}
/** Models on this computer: the user's own, and catalog models installed or on their way. */
const modelsOnComputer = (state: LocalModelsState) =>
  state.models.filter(model => model._tag === "Custom" || model.acquisitionState._tag !== "NotInstalled").length
/** A page tab at the top of the sidebar, as main's sidebar drew them: icon and name, highlighted while its page shows. */
function NavTab({ label, name, icon, current, onClick, status }: { label: string; name?: string; icon: ReactNode; current: boolean; onClick?: () => void; status?: ReactNode }) {
  return <Button variant="ghost" disabled={!onClick} onClick={onClick} aria-label={name} aria-current={current ? "page" : undefined}
    className={`group h-10 justify-start gap-3 rounded-lg px-3 text-left text-sm font-medium ${current ? "bg-blue-50 text-blue-700 dark:bg-slate-800 dark:text-blue-400" : "hover:bg-slate-100 dark:hover:bg-slate-800"}`}>
    {icon}{label}
    {status !== undefined && <span className="ml-auto inline-flex size-4 shrink-0 items-center justify-center text-blue-500">{status}</span>}
  </Button>
}
/** The Chat tab, with room at its end for its status; before the window reaches the service it shows none. */
function ChatTab({ page, onClick }: { page: MainPage; onClick?: (() => void) | undefined }) {
  return <NavTab label="Chat" icon={<ChatCircleIcon className="size-4 shrink-0" />} current={page === "chat"} {...(onClick ? { onClick } : {})}
    status={onClick ? <ChatTabStatus page={page} /> : null} />
}
/** The Models tab, with how many models are on this computer; before the window reaches the service it shows without a count and can't be opened yet. */
function ModelsTab({ current, onClick }: { current: boolean; onClick?: () => void }) {
  return onClick ? <CountedModelsTab current={current} onClick={onClick} /> : <ModelsTabView current={current} count={Option.none()} />
}
function CountedModelsTab({ current, onClick }: { current: boolean; onClick: () => void }) {
  return <ModelsTabView current={current} count={useLocalModelsSelector(modelsOnComputer, Equivalence.number)} onClick={onClick} />
}
function ModelsTabView({ current, count, onClick }: { current: boolean; count: Option.Option<number>; onClick?: () => void }) {
  return <NavTab label="Models" name={Option.match(count, { onNone: () => "Models", onSome: value => value > 0 ? `Models, ${value} on this computer` : "Models" })} current={current} onClick={onClick} icon={
    <span className="relative inline-flex">
      <CubeIcon className="size-4 shrink-0" />
      {Option.isSome(count) && count.value > 0 && <span aria-hidden="true" className={`absolute -top-1.5 -right-2 min-w-4 rounded-full bg-slate-200 px-1 text-center text-[10px] leading-4 font-semibold tabular-nums text-slate-700 ring-2 dark:bg-slate-700 dark:text-slate-100 ${current ? "ring-blue-50 dark:ring-slate-800" : "ring-slate-50 group-hover:ring-slate-100 dark:ring-slate-925 dark:group-hover:ring-slate-800"}`}>{count.value}</span>}
    </span>} />
}
function BarButton({ label, icon: BarIcon, onClick }: { label: string; icon: Icon; onClick?: () => void }) {
  return <ActionTooltip label={label} trigger={
    <button type="button" aria-label={label} disabled={!onClick} onClick={onClick}
      className="inline-flex size-8 shrink-0 items-center justify-center rounded-md text-slate-500 hover:bg-slate-100 hover:text-slate-900 focus-visible:outline-2 focus-visible:outline-blue-500 disabled:opacity-50 dark:hover:bg-slate-800 dark:hover:text-slate-100">
      <BarIcon className="size-[18px]" />
    </button>} />
}
/** Docs, a plain link in the sidebar's bar: hovering changes its text's colour and underlines it. */
function DocsLink({ onNavigate }: { onNavigate?: () => void }) {
  return <a href="https://docs.magnitude.dev" target="_blank" rel="noreferrer" onClick={onNavigate}
    className="group inline-flex items-center gap-2 rounded-sm px-2 text-[13px] font-medium focus-visible:outline-2 focus-visible:outline-blue-500">
    <BookOpenIcon className="size-4 shrink-0 text-slate-500 dark:text-slate-400" />
    <span className="inline-flex items-center gap-1 border-b border-transparent text-slate-600 group-hover:border-current group-hover:text-slate-400 dark:text-slate-300 dark:group-hover:text-slate-100">Docs<ArrowUpRightIcon aria-hidden="true" className="size-3.5" /></span>
  </a>
}
/** The sidebar: the Models, Connections and Chat tabs, the recent chats, the update when there is one, then a bar with Docs and Settings. */
function Navigation({ page, navigate, onNavigate }: { page: MainPage; navigate?: (page: ApplicationPage) => void; onNavigate?: () => void }) {
  const open = navigate && ((target: ApplicationPage) => () => { navigate(target); onNavigate?.() })
  return <nav id="desktop-navigation" className="flex min-h-0 flex-1 flex-col">
    <div className="flex shrink-0 flex-col gap-2 px-4">
      <ModelsTab current={page === "models"} onClick={open?.("models")} />
      <NavTab label="Connections" icon={<PlugIcon className="size-4 shrink-0" />} current={page === "connections"} onClick={open?.("connections")} />
      <ChatTab page={page} onClick={open?.("chat")} />
    </div>
    <div className="mt-2 flex min-h-0 flex-1 flex-col px-4">{navigate && <SidebarChats page={page} navigate={navigate} onNavigate={onNavigate} />}</div>
    <div className="shrink-0 px-4 pb-2.5 empty:hidden"><SidebarUpdatePill navigate={navigate} onNavigate={onNavigate} /></div>
    <div className="flex shrink-0 items-center justify-between gap-1 border-t border-slate-200 px-3 pt-2.5 dark:border-slate-750">
      <DocsLink onNavigate={onNavigate} />
      <BarButton label="Settings" icon={GearSixIcon} onClick={open?.("settings")} />
    </div>
  </nav>
}
/** Below `md`: a top bar that opens a navigation drawer over the page, closed by default. */
function NarrowBar({ open, onOpen }: { open: boolean; onOpen: () => void }) {
  return <header className="flex h-14 shrink-0 items-center gap-3 border-b border-slate-200 px-4 dark:border-slate-750" inert={open}>
    <button type="button" className="inline-flex size-10 items-center justify-center rounded-md text-slate-600 hover:bg-slate-100 focus-visible:outline-2 focus-visible:outline-blue-500 dark:text-slate-300 dark:hover:bg-slate-800" aria-label="Open navigation" aria-expanded={open} aria-controls="desktop-navigation" onClick={onOpen}>
      <SidebarSimpleIcon className="size-5" />
    </button>
    <span className="flex items-center gap-2 font-heading text-base font-semibold"><MagnitudeMark className="h-7 w-7 shrink-0" />Magnitude</span>
  </header>
}
function NavigationDrawer({ page, navigate, onClose }: { page: MainPage; navigate?: (page: ApplicationPage) => void; onClose: () => void }) {
  return <div className="fixed inset-0 z-50 flex" onKeyDown={event => { if (event.key === "Escape") onClose() }}>
    <button type="button" aria-label="Close navigation" className="absolute inset-0 cursor-default bg-black/45 dark:bg-black/70" onClick={onClose} />
    <aside aria-label="Navigation" className="relative flex h-full w-64 max-w-[85vw] flex-col border-r border-slate-200 bg-slate-50 pb-3 pt-4 shadow-xl dark:border-slate-750 dark:bg-slate-925 motion-safe:animate-[slide-in-left_150ms_ease-out]">
      <div className="mb-6 flex h-10 items-center justify-between px-4">
        <span className="flex items-center gap-3 px-3 font-heading text-base font-semibold"><MagnitudeMark className="h-8 w-8 shrink-0" />Magnitude</span>
        <button type="button" autoFocus className="inline-flex size-10 items-center justify-center rounded-md text-slate-500 hover:bg-slate-100 focus-visible:outline-2 focus-visible:outline-blue-500 dark:hover:bg-slate-800" aria-label="Close navigation" onClick={onClose}><XIcon className="size-5" /></button>
      </div>
      <Navigation page={page} navigate={navigate} onNavigate={onClose} />
    </aside>
  </div>
}
/** From `md` up: the collapsible sidebar with the window's integrated controls and drag regions. */
function DockedSidebar({ page, navigate, platform }: { page: MainPage; navigate?: (page: ApplicationPage) => void; platform: string | undefined }) {
  const [collapsed, setCollapsed] = useState(false)
  const updatePending = useSidebarUpdate().update._tag !== "None"
  const integratedControls = platform === "darwin" || platform === "win32"
  const sidebarWidth = collapsed ? 0 : 224
  return <>
    {integratedControls && <div aria-hidden="true" data-window-drag-region style={{ left: sidebarWidth }} className="absolute right-0 top-0 z-50 h-8 select-none transition-[left] duration-250 ease-in-out motion-reduce:transition-none [-webkit-app-region:drag]" />}
    <div data-window-drag-region={integratedControls ? "" : undefined} style={{ width: collapsed ? (platform === "darwin" ? 128 : 64) : sidebarWidth }} className={`absolute left-0 top-0 z-50 flex h-[42px] items-center justify-end px-4 transition-[width] duration-250 ease-in-out motion-reduce:transition-none ${integratedControls ? "select-none [-webkit-app-region:drag]" : ""}`}>
      <button type="button" className="relative inline-flex size-6 items-center justify-center rounded-sm text-slate-500 hover:text-slate-900 focus-visible:outline-2 focus-visible:outline-blue-500 dark:hover:text-slate-100 [-webkit-app-region:no-drag]" aria-label={collapsed ? (updatePending ? "Expand sidebar, update available" : "Expand sidebar") : "Collapse sidebar"} aria-expanded={!collapsed} aria-controls="desktop-navigation" onClick={() => setCollapsed(value => !value)}>
        <SidebarSimpleIcon className="size-5" />
        {collapsed && updatePending && <span data-update-dot aria-hidden="true" className="absolute -right-0.5 -top-0.5 size-2 rounded-full bg-blue-500 ring-2 ring-slate-50 dark:ring-slate-925" />}
      </button>
    </div>
    <aside aria-hidden={collapsed} inert={collapsed} style={{ width: sidebarWidth }} className="shrink-0 overflow-hidden transition-[width] duration-250 ease-in-out motion-reduce:transition-none">
      <div className={`flex h-full w-56 flex-col border-r border-slate-200 pt-10 pb-3 transition-transform duration-250 ease-in-out motion-reduce:transition-none dark:border-slate-750 ${collapsed ? "-translate-x-full" : "translate-x-0"}`}>
      <div className="mb-8 mt-4 flex h-8 shrink-0 items-center gap-3 px-7 font-heading text-base font-semibold">
        <MagnitudeMark className="h-8 w-8 shrink-0" />Magnitude
      </div>
      <Navigation page={page} navigate={navigate} />
      </div>
    </aside>
  </>
}
