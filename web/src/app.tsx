/**
 * App root component — spec §9.2
 *
 * Wraps the component tree in DisplayViewControllerProvider.
 * Wires the display view controller, session list, composer, and panels.
 *
 * Cold RPCs use useAgentClient().query() / .mutation() (effect-atom).
 * StreamDisplayView uses the display view store (spec §6.1).
 * Local UI state uses plain atoms (spec §6.3).
 */
import {
  useCallback,
  useMemo,
  useRef,
  useSyncExternalStore,
  type ReactNode,
} from "react"
import { Menu } from "lucide-react"
import { Option, Effect } from "effect"
import {
  useAtomValue,
  useAtomSet,
  useAtomMount,
  Atom,
  Result,
} from "@effect-atom/atom-react"
import {
  type CommandContext,
  DisplayViewControllerProvider,
  useDisplayState,
  useDisplayViewController,
  useDisplayConnectionError,
  useSelectedSessionId,
  usePlatform,
  useAgentClient,
  useComposerState,
  useSessionPreload,
  useSessionActions,
  usePaginatedSessions,
  useOnboardingModelSetup,
  useLocalModels,
  useModelSlots,
  useModelConfig,
  useProviderModelCatalog,
  deriveCurrentLocalModel,
  installedLocalModels,
  formatLocalModelDisplayName,
  modelSlotResidentAllocation,
  selectedSlotModel,
  reasoningEffortControl,
  formatReasoningEffort,
  useActiveSessionStatusesSubscription,
  activeSessionStatusesAtom,
} from "@magnitudedev/client-common"
import { SessionsSidebar } from "./components/sessions-sidebar"
import { ChatTimeline } from "./components/chat-timeline"
import { WorkStatusBar } from "./components/work-status-bar"
import { Composer } from "./components/composer"
import { FooterBar } from "./components/footer-bar"
import { FileViewerPanel } from "./components/file-viewer-panel"
import { WorkerDetailPanel } from "./components/worker-detail-panel"
import { WorkStatusBarSkeleton } from "./components/work-status-bar-skeleton"
import { ContextUsageIndicator } from "./components/context-usage-indicator"
import { SettingsCenter } from "./components/model-center"
import { LocalModelOnboarding } from "./components/local-model-onboarding"
import { formatBytes } from "./components/local-inference-format"
import { ChatColumnPage } from "./components/chat-column-page"
import {
  selectedCwdAtom,
  selectedFilePathAtom,
  bashModeAtom,
  nextEscWillKillAllAtom,
} from "@magnitudedev/client-common"
import {
  sidebarSearchAtom,
  sidebarCwdFilterAtom,
  sidebarVisibleAtom,
  settingsTabAtom,
} from "./state/web-atoms"
import { useMenuActions } from "./hooks/use-menu-actions"
import { DaemonConnectionError } from "./components/daemon-connection-error"
import { ToastContainer } from "./components/toast"
import { showToast } from "./stores/toast-store"
import { subscribeResponsive, getIsNarrow } from "./stores/responsive-store"
import {
  useSlotProfiles,
  findSlotProfile,
  type SlotProfile,
  type SlotProfiles,
} from "@magnitudedev/client-common"
import {
  isRoleId,
  PRIMARY_SLOT_ID,
  ProviderIdSchema,
  ROLE_TO_SLOT,
  SECONDARY_SLOT_ID,
} from "@magnitudedev/sdk"
import type {
  DisplayActor,
  ReadFileResult,
  SessionCwdSummary,
  SessionMetadata,
} from "@magnitudedev/sdk"
import type { SlotId } from "@magnitudedev/sdk"
import { registerWebCommands } from "./commands/register"
registerWebCommands()
function formatRoleLabel(role: string | null | undefined): string {
  if (!role) return "Leader"
  return role.charAt(0).toUpperCase() + role.slice(1)
}

/**
 * Look up a slot profile for a given actor role.
 * Maps role → slot via ROLE_TO_SLOT, then finds the profile for that slot.
 */
function findSlotProfileForRole(
  profiles: SlotProfiles | null,
  role: string | null | undefined
): SlotProfile | null {
  if (!profiles || !role || !isRoleId(role)) return null
  const slotId =
    ROLE_TO_SLOT[role] === "primary" ? PRIMARY_SLOT_ID : SECONDARY_SLOT_ID
  return Option.getOrNull(findSlotProfile(profiles, slotId))
}
function useRootSlotProfile(slotProfiles: SlotProfiles | null): {
  roleId: string
  roleLabel: string
  profile: SlotProfile | null
} {
  const rootRole = useDisplayState(
    (state) => state.actors["root"]?.role ?? null
  )
  const roleId = rootRole ?? "leader"
  return {
    roleId,
    roleLabel: formatRoleLabel(roleId),
    profile: findSlotProfileForRole(slotProfiles, roleId),
  }
}

/** Sessions sidebar container — ListSessions query + shared session actions */
function SessionsSidebarContainer(props?: {
  overlay?: boolean
  onCloseOverlay?: () => void
}): ReactNode {
  const client = useAgentClient()
  const { startNewSession, resumeSession } = useSessionActions()
  const cwdFilter = useAtomValue(sidebarCwdFilterAtom)
  const setCwdFilter = useAtomSet(sidebarCwdFilterAtom)
  const searchQuery = useAtomValue(sidebarSearchAtom)
  const activeSessionStatuses = useAtomValue(activeSessionStatusesAtom)
  const settingsTab = useAtomValue(settingsTabAtom)
  const setSettingsTab = useAtomSet(settingsTabAtom)
  const trimmedSearchQuery = searchQuery.trim()
  const sessionPage = usePaginatedSessions({
    ...(cwdFilter
      ? {
          cwd: cwdFilter,
        }
      : {}),
    ...(trimmedSearchQuery
      ? {
          query: trimmedSearchQuery,
        }
      : {}),
    pageSize: 50,
  })

  // Listen for __magnitude:focus-search custom event → focus the search input
  const focusSearchAtom = useMemo(
    () =>
      Atom.make(
        Effect.gen(function* () {
          const handler = () => {
            const input = document.getElementById("sidebar-search-input")
            if (input) input.focus()
          }
          window.addEventListener("__magnitude:focus-search", handler)
          yield* Effect.addFinalizer(() =>
            Effect.sync(() =>
              window.removeEventListener("__magnitude:focus-search", handler)
            )
          )
        })
      ),
    []
  )
  useAtomMount(focusSearchAtom)
  const cwdOptionsResult = useAtomValue(
    client.rpc.query(
      "ListSessionCwds",
      {},
      {
        reactivityKeys: ["sessions"],
      }
    )
  )
  const cwdOptions = Result.match(cwdOptionsResult, {
    onInitial: () => [] as string[],
    onFailure: () => [] as string[],
    onSuccess: (result) =>
      (result.value as SessionCwdSummary[]).map((summary) => summary.cwd),
  })
  const handleNewSession = () => {
    setSettingsTab(null)
    startNewSession({
      cwd: cwdFilter,
    })
    if (props?.overlay && props.onCloseOverlay) props.onCloseOverlay()
  }
  return (
    <SessionsSidebar
      loading={sessionPage.loading}
      sessions={sessionPage.sessions.map((s) => {
        const liveStatus = activeSessionStatuses[s.id]
        const statusFields = liveStatus
          ? {
              updatedAt: liveStatus.lastMessageAt,
              workStatus: liveStatus.workStatus,
              activeWorkerCount: liveStatus.activeWorkerCount,
            }
          : {
              updatedAt: s.timestamp,
              workStatus: "idle" as const,
              activeWorkerCount: 0,
            }
        return {
          sessionId: s.id,
          title: s.title,
          cwd: s.workingDirectory,
          messageCount: s.messageCount,
          ...statusFields,
        }
      })}
      cwdFilter={cwdFilter}
      cwdOptions={cwdOptions}
      loadingMore={sessionPage.loadingMore}
      hasMore={sessionPage.hasMore}
      onCwdFilterChange={setCwdFilter}
      onLoadMore={sessionPage.loadMore}
      onSelectSession={(id) => {
        setSettingsTab(null)
        resumeSession(id)
      }}
      onNewSession={handleNewSession}
      onOpenSettings={() => {
        setSettingsTab("models")
      }}
      settingsTab={settingsTab}
      onSettingsTabChange={(tab) => {
        setSettingsTab(tab)
        props?.onCloseOverlay?.()
      }}
      onCloseSettings={() => setSettingsTab(null)}
      overlay={props?.overlay}
      onCloseOverlay={props?.onCloseOverlay}
    />
  )
}

/** FileViewerPanel container — ReadFile query */
function FileViewerPanelContainer(): ReactNode {
  const filePath = useAtomValue(selectedFilePathAtom)
  const setFilePath = useAtomSet(selectedFilePathAtom)
  const client = useAgentClient()
  const selectedCwd = useAtomValue(selectedCwdAtom)

  // Determine format based on file extension — images need base64
  const isImageFile = filePath
    ? ["png", "jpg", "jpeg", "gif", "webp", "svg"].includes(
        filePath.split(".").pop() ?? ""
      )
    : false

  // Only query when we have a real path + cwd.
  // When no file is selected, use a static idle atom so the hook count stays stable.
  // P1: reactivityKeys: ["files"] so the query refreshes when files change.
  const readFileAtom = useMemo(
    () =>
      filePath && selectedCwd
        ? client.rpc.query(
            "ReadFile",
            {
              cwd: selectedCwd,
              path: filePath,
              format: isImageFile ? "base64" : "text",
            },
            {
              reactivityKeys: ["files"],
            }
          )
        : Atom.make(() => null),
    [client, selectedCwd, filePath, isImageFile]
  )
  const result = useAtomValue(readFileAtom)

  // P2: Handle loading and error states explicitly
  // result is null when no file is selected (idle atom), so guard for that.
  const loading =
    !!filePath && !!selectedCwd && result !== null && Result.isInitial(result)
  const errorMsg =
    filePath && selectedCwd && result !== null && Result.isFailure(result)
      ? "Failed to read file. The file may not exist or is not accessible."
      : null
  const content =
    filePath && result !== null && Result.isSuccess(result)
      ? (result.value as ReadFileResult).content
      : null
  return (
    <FileViewerPanel
      filePath={filePath}
      content={content}
      loading={loading}
      error={errorMsg}
      onClose={() => setFilePath(null)}
    />
  )
}

/** WorkerDetailPanel container — read-only worker timeline */
function WorkerDetailPanelContainer({
  slotProfiles,
}: {
  slotProfiles: SlotProfiles | null
}): ReactNode {
  const { topForkId } = useDisplayViewController()
  const actors = useDisplayState((state) => state.actors)
  const tasks = useDisplayState((state) => state.tasks)
  const actor = topForkId ? actors[topForkId] ?? null : null
  const worker = topForkId ? deriveWorkerInfo(topForkId, actors) : null
  const taskTitle = actor?.taskId
    ? tasks?.byId[actor.taskId]?.title ?? null
    : null
  const profile = findSlotProfileForRole(slotProfiles, actor?.role)
  const modelDisplayName = profile?.modelDisplayName ?? null
  return (
    <WorkerDetailPanel
      forkId={topForkId}
      worker={worker}
      loadingTitle={taskTitle ?? undefined}
      loadingSubtitle={modelDisplayName}
    />
  )
}
function WorkerDetailPageContainer({
  slotProfiles,
}: {
  slotProfiles: SlotProfiles | null
}): ReactNode {
  const { topForkId, popFork } = useDisplayViewController()
  const actors = useDisplayState((state) => state.actors)
  const actor = topForkId ? actors[topForkId] ?? null : null
  const worker = topForkId ? deriveWorkerInfo(topForkId, actors) : null
  const profile = findSlotProfileForRole(slotProfiles, actor?.role)
  const title = worker
    ? `${formatRoleLabel(worker.role)}: ${worker.name}`
    : "Worker"
  return (
    <ChatColumnPage
      title={title}
      backLabel="Back to session"
      onBack={popFork}
      actions={
        actor ? (
          <ContextUsageIndicator
            context={actor.context}
            tokenCap={profile?.contextWindow ?? null}
            size={20}
            strokeWidth={2}
            showTokenLabel
            tooltip="native"
          />
        ) : null
      }
    >
      <WorkerDetailPanelContainer slotProfiles={slotProfiles} />
    </ChatColumnPage>
  )
}
function deriveWorkerInfo(
  forkId: string,
  actors: Record<string, DisplayActor>
): {
  forkId: string
  role: string
  name: string
} | null {
  const actor = actors[forkId]
  if (!actor || actor.kind !== "worker") return null
  return {
    forkId,
    role: actor.role,
    name: actor.name,
  }
}

/** Work status container — timer + active task table above composer */
function WorkStatusBarContainer({
  slotProfiles,
}: {
  slotProfiles: SlotProfiles | null
}): ReactNode {
  const rootActor = useDisplayState((state) => state.actors["root"] ?? null)
  const actors = useDisplayState((state) => state.actors)
  const tasks = useDisplayState((state) => state.tasks)
  const selectedSessionId = useSelectedSessionId()
  const { pushFork } = useDisplayViewController()

  // While a session is selected but display state hasn't populated yet
  // (root actor not yet received from the stream), show the skeleton to
  // reserve layout space.
  if (rootActor === null && selectedSessionId !== null) {
    return <WorkStatusBarSkeleton />
  }
  return (
    <WorkStatusBar
      rootActor={rootActor}
      actors={actors}
      tasks={tasks}
      slotProfiles={slotProfiles}
      onWorkerClick={pushFork}
    />
  )
}
function ComposerContainer({
  docked = false,
}: {
  docked?: boolean
}): ReactNode {
  const platform = usePlatform()
  const setBashMode = useAtomSet(bashModeAtom)
  const setSettingsTab = useAtomSet(settingsTabAtom)
  const setFilePath = useAtomSet(selectedFilePathAtom)
  const sidebarVisible = useAtomValue(sidebarVisibleAtom)
  const setSidebarVisible = useAtomSet(sidebarVisibleAtom)
  const { startNewSession } = useSessionActions()
  const sendRef = useRef<(text: string) => void>(() => {})
  const slotsResult = useModelSlots()
  const onboarding = useOnboardingModelSetup()
  const slots = Option.getOrNull(Result.value(slotsResult))
  const primary = slots?.slots.primary ?? null
  const modelReady =
    primary?._tag === "ConfiguredLocal" &&
    primary.availability._tag === "Available" &&
    primary.residency._tag === "Ready"
  const disabledReason = modelReady
    ? null
    : Result.isFailure(slotsResult)
    ? "Model runtime state is unavailable"
    : "Load a local model before sending"
  const commandContext: CommandContext = useMemo(
    () => ({
      resetConversation: () => startNewSession(),
      showSystemMessage: (message: string) => showToast("info", message),
      exitApp: () => {
        if (platform.quit) platform.quit()
      },
      openRecentChats: () => {
        if (getIsNarrow() && !sidebarVisible) {
          setSidebarVisible(true)
        }
        window.dispatchEvent(new CustomEvent("__magnitude:focus-search"))
      },
      enterBashMode: () => setBashMode(true),
      activateSkill: (
        skillName: string,
        _skillPath: string | undefined,
        args: string
      ) => {
        const content = args.trim()
          ? `/${skillName} ${args.trim()}`
          : `/${skillName}`
        sendRef.current(content)
      },
      initProject: () => {
        showToast(
          "info",
          "Project initialization is not available in the web app yet."
        )
      },
      openSettings: () => setSettingsTab("models"),
      openSetup: onboarding.open,
      openModelMenu: (menu) => {
        if (menu === "models" || menu === "catalog" || menu === "hardware") {
          setSettingsTab(menu)
        }
      },
      toggleAutopilot: () => {
        showToast("info", "Autopilot mode is not yet available in the web app.")
      },
    }),
    [
      startNewSession,
      platform,
      sidebarVisible,
      setSidebarVisible,
      setBashMode,
      setSettingsTab,
      onboarding.open,
    ]
  )
  const composer = useComposerState(commandContext)
  sendRef.current = (text: string) => composer.handleSend(text)
  const handleMentionConfirm = useCallback(
    (item: { path: string }) => {
      setFilePath(item.path)
    },
    [setFilePath]
  )
  return (
    <Composer
      role={composer.roleLabel}
      isStreaming={composer.isStreaming}
      bashMode={composer.bashMode}
      onSend={(text, mentions) => {
        void composer.handleSend(text, {
          mentions,
        })
      }}
      onInterrupt={composer.handleInterrupt}
      onRunBash={composer.handleRunBash}
      onSlashCommand={composer.handleSlashCommand}
      onToggleBashMode={() => composer.setBashMode((prev: boolean) => !prev)}
      onMentionConfirm={handleMentionConfirm}
      mentionClient={composer.mentionClient}
      cwd={composer.cwd}
      docked={docked}
      disabledReason={disabledReason}
      onDisabledAction={() => setSettingsTab("models")}
    />
  )
}

/** FooterBar container */
function FooterBarContainer({
  slotProfiles,
}: {
  slotProfiles: SlotProfiles | null
}): ReactNode {
  const client = useAgentClient()
  const selectedSessionId = useSelectedSessionId()
  const selectedCwd = useAtomValue(selectedCwdAtom)
  const context = useDisplayState(
    (state) => state.actors["root"]?.context ?? null
  )
  const { profile } = useRootSlotProfile(slotProfiles)
  const tokenCap = profile?.contextWindow ?? null
  const bashMode = useAtomValue(bashModeAtom)
  const nextEscWillKillAll = useAtomValue(nextEscWillKillAllAtom)
  const { displayMode } = useDisplayViewController()
  const setSettingsTab = useAtomSet(settingsTabAtom)
  const localModelsResult = useLocalModels()
  const slotsResult = useModelSlots()
  const catalogResult = useProviderModelCatalog()
  const modelConfig = useModelConfig()
  const slots = Option.getOrNull(Result.value(slotsResult))
  const currentModel = deriveCurrentLocalModel(
    Option.fromNullable(slots?.slots.primary)
  )
  const allocation = slots
    ? modelSlotResidentAllocation(slots.slots.primary)
    : Option.none()
  const residentBytes = Option.match(allocation, {
    onNone: () => null,
    onSome: ({ memoryDomains }) =>
      memoryDomains.reduce(
        (total, domain) =>
          total +
          domain.modelBytes +
          domain.contextBytes +
          domain.computeBytes +
          domain.auxiliaryBytes,
        0
      ),
  })
  const selectedSessionAtom = useMemo(
    () =>
      selectedSessionId
        ? client.rpc.query(
            "GetSession",
            {
              sessionId: selectedSessionId,
            },
            {
              reactivityKeys: ["sessions"],
            }
          )
        : Atom.make(() => null),
    [client, selectedSessionId]
  )
  const selectedSessionResult = useAtomValue(selectedSessionAtom)
  const sessionCwd =
    selectedSessionResult !== null && Result.isSuccess(selectedSessionResult)
      ? (selectedSessionResult.value as SessionMetadata).cwd
      : null
  const cwd = sessionCwd ?? selectedCwd
  const selectedModel = Option.flatMap(
    Option.all({
      catalog: Result.value(catalogResult),
      slots: Result.value(slotsResult),
    }),
    ({ catalog, slots }) => selectedSlotModel(catalog, slots, PRIMARY_SLOT_ID)
  )
  const thinkingOptions = Option.match(selectedModel, {
    onNone: () => [],
    onSome: ({ model }) => {
      const control = reasoningEffortControl(model)
      return control._tag === "Available" ? control.options : []
    },
  })
  const thinkingLevel = profile?.reasoningEffort
    ? formatReasoningEffort(profile.reasoningEffort)
    : null
  const openHardware = useCallback(() => {
    setSettingsTab("hardware")
  }, [setSettingsTab])
  const modelOptions = Option.match(Result.value(localModelsResult), {
    onNone: () => [],
    onSome: (state) =>
      installedLocalModels(state)
        .flatMap((model) => {
          if (
            model.servingState._tag !== "Assessed" ||
            model.servingState.availabilityState._tag !== "Selectable"
          ) {
            return []
          }
          return [
            {
              value: model.servingState.availabilityState.providerModelId,
              label: formatLocalModelDisplayName(model),
            },
          ]
        })
        .sort((left, right) => left.label.localeCompare(right.label)),
  })
  const primarySlot = slots?.slots.primary
  const selectedModelId =
    primarySlot && primarySlot._tag !== "Unassigned"
      ? primarySlot.selection.providerModelId
      : null
  const modelLabel =
    currentModel._tag === "NoSelection"
      ? "Choose model"
      : currentModel.displayName
  const modelResidency =
    currentModel._tag === "NoSelection"
      ? null
      : currentModel._tag === "Running"
      ? ("ready" as const)
      : currentModel._tag === "Loading" || currentModel._tag === "Stopping"
      ? ("loading" as const)
      : ("not-ready" as const)
  const modelLoadingPercentage =
    currentModel._tag === "Loading" ? currentModel.percentage : null
  const memoryLabel =
    currentModel._tag === "Running" && residentBytes !== null
      ? `${formatBytes(residentBytes)} mem`
      : null
  return (
    <FooterBar
      context={context}
      tokenCap={tokenCap}
      cwd={cwd}
      model={modelLabel}
      modelResidency={modelResidency}
      modelLoadingPercentage={modelLoadingPercentage}
      thinkingLevel={thinkingLevel}
      memoryLabel={memoryLabel}
      thinkingEffort={profile?.reasoningEffort ?? null}
      thinkingOptions={thinkingOptions}
      modelOptions={modelOptions}
      selectedModelId={selectedModelId}
      onModelSelect={(providerModelId) => {
        modelConfig.updateSlotModel(
          PRIMARY_SLOT_ID,
          ProviderIdSchema.make("local"),
          providerModelId
        )
      }}
      onThinkingSelect={(effort) => {
        modelConfig.updateSlotReasoning(PRIMARY_SLOT_ID, effort)
      }}
      onMemoryClick={openHardware}
      bashMode={bashMode}
      nextEscWillKillAll={nextEscWillKillAll}
      transcriptMode={displayMode === "transcript"}
    />
  )
}
function BottomDockContainer({
  slotProfiles,
}: {
  slotProfiles: SlotProfiles | null
}): ReactNode {
  return (
    <div className="[margin:14px_12px_14px] flex flex-col [gap:8px] shrink-0">
      <WorkStatusBarContainer slotProfiles={slotProfiles} />
      <ComposerContainer docked />
      <FooterBarContainer slotProfiles={slotProfiles} />
    </div>
  )
}
function ChatTitleBar({
  onOpenSidebar,
}: {
  onOpenSidebar?: () => void
}): ReactNode {
  const client = useAgentClient()
  const selectedSessionId = useSelectedSessionId()
  const displaySession = useDisplayState((state) => state.session)
  const selectedSessionAtom = useMemo(
    () =>
      selectedSessionId
        ? client.rpc.query(
            "GetSession",
            {
              sessionId: selectedSessionId,
            },
            {
              reactivityKeys: ["sessions"],
            }
          )
        : Atom.make(() => null),
    [client, selectedSessionId]
  )
  const selectedSessionResult = useAtomValue(selectedSessionAtom)
  const metadataTitle =
    selectedSessionResult !== null && Result.isSuccess(selectedSessionResult)
      ? (selectedSessionResult.value as SessionMetadata).title
      : null
  const streamedTitle =
    displaySession.sessionId === selectedSessionId ? displaySession.title : null
  const title = selectedSessionId
    ? (streamedTitle ?? metadataTitle)?.trim() || "Untitled session"
    : "New session"
  return (
    <div
      className="mac:[-webkit-app-region:drag] h-11 shrink-0 flex items-center px-4 bg-slate-50 dark:bg-slate-925 border-b border-slate-200 dark:border-slate-800 select-none"
      title={title}
    >
      {onOpenSidebar && (
        <button
          type="button"
          className="appearance-none min-h-8 rounded-[7px] px-3 inline-flex items-center justify-center gap-1.5 font-sans text-xs font-semibold leading-none cursor-pointer disabled:cursor-not-allowed disabled:opacity-50 focus-visible:outline focus-visible:outline-1 focus-visible:outline-offset-1 focus-visible:outline-blue-700 dark:focus-visible:outline-blue-500 w-8 !px-0 bg-transparent text-slate-600 dark:text-slate-400 border border-slate-300 dark:border-slate-750 hover:bg-slate-150 hover:text-slate-900 dark:hover:bg-slate-750 dark:hover:text-slate-200 shrink-0 mr-2.5"
          aria-label="Open sessions"
          title="Open sessions"
          onClick={onOpenSidebar}
        >
          <Menu size={17} />
        </button>
      )}
      <span className="min-w-0 max-w-[60%] overflow-hidden text-ellipsis whitespace-nowrap text-slate-900 dark:text-slate-200 font-sans text-[15px] font-medium">
        {title}
      </span>
    </div>
  )
}

/** Listen for __magnitude:interrupt-all custom event → Interrupt RPC with target: all */
function useInterruptAllListener(): void {
  const client = useAgentClient()
  const selectedSessionId = useSelectedSessionId()
  const interruptMutation = useAtomSet(client.rpc.mutation("Interrupt"))
  const interruptAtom = useMemo(
    () =>
      Atom.make(
        Effect.gen(function* () {
          const handler = () => {
            if (!selectedSessionId) return
            interruptMutation({
              payload: {
                sessionId: selectedSessionId,
                target: {
                  _tag: "all",
                },
              },
            })
          }
          window.addEventListener("__magnitude:interrupt-all", handler)
          yield* Effect.addFinalizer(() =>
            Effect.sync(() =>
              window.removeEventListener("__magnitude:interrupt-all", handler)
            )
          )
        })
      ),
    [selectedSessionId, interruptMutation]
  )
  useAtomMount(interruptAtom)
}

/** Inner app — has display view + AgentClient context */
function AppInner(): ReactNode {
  // Detect responsive mode (≤640px) — no useEffect, uses matchMedia store
  const isNarrow = useSyncExternalStore(subscribeResponsive, getIsNarrow)
  useMenuActions()
  useInterruptAllListener()
  const onboarding = useOnboardingModelSetup()
  const onboardingState = Option.getOrNull(Result.value(onboarding.view))
  if (Result.isFailure(onboarding.view)) {
    return (
      <div className="min-h-screen flex flex-col items-center justify-center gap-2.5 p-8 text-center bg-slate-50 dark:bg-slate-925 text-slate-900 dark:text-slate-200 [&_h1]:mt-1 [&_h1]:text-[22px] [&_p]:mb-2 [&_p]:text-slate-600 dark:[&_p]:text-slate-400">
        <AlertTriangleIcon />
        <h1>Couldn’t load local setup</h1>
        <p>The daemon did not return onboarding state.</p>
        <button
          className="appearance-none min-h-8 rounded-[7px] px-3 inline-flex items-center justify-center gap-1.5 font-sans text-xs font-semibold leading-none cursor-pointer disabled:cursor-not-allowed disabled:opacity-50 focus-visible:outline focus-visible:outline-1 focus-visible:outline-offset-1 focus-visible:outline-blue-700 dark:focus-visible:outline-blue-500 bg-blue-700 text-slate-50 hover:bg-blue-800 dark:bg-blue-500 dark:text-slate-925 dark:hover:bg-blue-400"
          type="button"
          onClick={onboarding.retry}
        >
          Retry
        </button>
      </div>
    )
  }
  if (onboardingState === null) {
    return (
      <div className="min-h-screen flex flex-col items-center justify-center gap-2.5 p-8 text-center bg-slate-50 dark:bg-slate-925 text-slate-900 dark:text-slate-200 [&_h1]:mt-1 [&_h1]:text-[22px] [&_p]:mb-2 [&_p]:text-slate-600 dark:[&_p]:text-slate-400">
        <div className="size-[26px] rounded-full border-2 border-slate-300 border-t-blue-700 animate-spin dark:border-slate-750 dark:border-t-blue-500" />
        <h1>Connecting to local inference</h1>
        <p>Reading the current daemon state…</p>
      </div>
    )
  }
  if (onboardingState._tag === "Open") {
    return <LocalModelOnboarding setup={onboarding} />
  }
  return <AuthenticatedAppContent isNarrow={isNarrow} />
}
function AlertTriangleIcon(): ReactNode {
  return (
    <span className="size-[30px] rounded-full grid place-items-center text-red-700 bg-red-200 font-extrabold dark:text-red-300 dark:bg-red-800">
      !
    </span>
  )
}
function AuthenticatedAppContent({
  isNarrow,
}: {
  isNarrow: boolean
}): ReactNode {
  useSessionPreload()
  useActiveSessionStatusesSubscription()
  const connectionError = useDisplayConnectionError()
  const platform = usePlatform()
  const isDesktop = platform.id === "desktop"
  const sidebarVisible = useAtomValue(sidebarVisibleAtom)
  const setSidebarVisible = useAtomSet(sidebarVisibleAtom)
  const { profiles: slotProfiles } = useSlotProfiles()
  const showOverlaySidebar = isNarrow && sidebarVisible
  const settingsTab = useAtomValue(settingsTabAtom)
  const controller = useDisplayViewController()
  const forkStack = controller.expandedForkStack
  const panelOpen = settingsTab !== null
  const workerDetailOpen = !panelOpen && forkStack.length > 0
  return (
    <div
      className={`${
        isDesktop ? "[background:transparent]" : "bg-slate-50 dark:bg-slate-925"
      }  app flex [height:100vh] overflow-hidden`}
    >
      {/* Docked sidebar — hidden by CSS when narrow */}
      {!isNarrow && <SessionsSidebarContainer />}
      {/* Overlay sidebar — shown when narrow + visible */}
      {showOverlaySidebar && (
        <SessionsSidebarContainer
          overlay
          onCloseOverlay={() => setSidebarVisible(false)}
        />
      )}
      <div className="chat-column [flex:1] min-w-0 flex flex-col relative bg-slate-50 dark:bg-slate-925">
        {/* Main chat column — always mounted, always in the layout. When a
            panel or worker detail is open, it's covered by an absolute
            overlay. Keeping it in the layout (not display:none) preserves
            scroll metrics so the scroll controller can capture and restore
            the correct position across overlay navigation. */}
        <div className="flex flex-col [flex:1] min-h-0">
          <ChatTitleBar
            onOpenSidebar={isNarrow ? () => setSidebarVisible(true) : undefined}
          />
          <ChatTimeline isVisible={!panelOpen && !workerDetailOpen} />
          <BottomDockContainer slotProfiles={slotProfiles} />
        </div>
        {(panelOpen || workerDetailOpen) && (
          <div className="absolute [inset:0px] flex flex-col bg-slate-50 dark:bg-slate-925 z-[1]">
            {panelOpen && (
              <>
                {isNarrow && (
                  <button
                    type="button"
                    className="appearance-none min-h-8 rounded-[7px] px-3 inline-flex items-center justify-center gap-1.5 font-sans text-xs font-semibold leading-none cursor-pointer disabled:cursor-not-allowed disabled:opacity-50 focus-visible:outline focus-visible:outline-1 focus-visible:outline-offset-1 focus-visible:outline-blue-700 dark:focus-visible:outline-blue-500 w-8 !px-0 bg-transparent text-slate-600 dark:text-slate-400 border border-slate-300 dark:border-slate-750 hover:bg-slate-150 hover:text-slate-900 dark:hover:bg-slate-750 dark:hover:text-slate-200 absolute top-3 left-3 z-[4] bg-slate-50 dark:bg-slate-925"
                    aria-label="Open settings navigation"
                    title="Open settings navigation"
                    onClick={() => setSidebarVisible(true)}
                  >
                    <Menu size={17} />
                  </button>
                )}
                <SettingsCenter tab={settingsTab} />
              </>
            )}
            {workerDetailOpen && (
              <WorkerDetailPageContainer slotProfiles={slotProfiles} />
            )}
          </div>
        )}
        <ToastContainer />
      </div>
      {!panelOpen && <FileViewerPanelContainer />}
      {connectionError && (
        <DaemonConnectionError
          message={connectionError.message}
          reconnecting={connectionError.reconnecting}
          invariantViolation={connectionError.invariantViolation}
          onRetry={() => {
            const retried = controller.retry()
            if (!retried) {
              controller.clearSession()
            }
          }}
          onQuit={() => {
            // If the platform supports quit (desktop), quit the app
            if (platform.quit) {
              platform.quit()
            } else {
              controller.clearSession()
            }
          }}
        />
      )}
    </div>
  )
}
export function App(): ReactNode {
  return (
    <DisplayViewControllerProvider>
      <AppInner />
    </DisplayViewControllerProvider>
  )
}
