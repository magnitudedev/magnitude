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
import { NotePencil, SidebarSimple } from "@phosphor-icons/react"
import { Cause, Option, Effect } from "effect"
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
  useAcnLifecycle,
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
  useProjects,
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
  selectedProjectIdAtom,
  selectedFilePathAtom,
  bashModeAtom,
  nextEscWillKillAllAtom,
} from "@magnitudedev/client-common"
import {
  sidebarSearchAtom,
  sidebarCollapsedAtom,
  sidebarWidthAtom,
  sidebarVisibleAtom,
  settingsTabAtom,
} from "./state/web-atoms"
import { useMenuActions } from "./hooks/use-menu-actions"
import { DaemonConnectionError } from "./components/daemon-connection-error"
import { AcnBootstrapScreen } from "./components/acn-bootstrap-screen"
import { MagnitudeMark } from "./components/magnitude-mark"
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
  AcnLifecycleState,
  DisplayActor,
  ReadFileResult,
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
  titlebarIntegrated?: boolean
}): ReactNode {
  const client = useAgentClient()
  const { startNewSession, resumeSession } = useSessionActions()
  const selectedSessionId = useSelectedSessionId()
  const searchQuery = useAtomValue(sidebarSearchAtom)
  const activeSessionStatuses = useAtomValue(activeSessionStatusesAtom)
  const settingsTab = useAtomValue(settingsTabAtom)
  const setSettingsTab = useAtomSet(settingsTabAtom)
  const trimmedSearchQuery = searchQuery.trim()
  const projects = useProjects()
  const selectedProjectId = useAtomValue(selectedProjectIdAtom)
  const setSelectedCwd = useAtomSet(selectedCwdAtom)
  const setSelectedProjectId = useAtomSet(selectedProjectIdAtom)
  const sessionPage = usePaginatedSessions({
    includeClosed: trimmedSearchQuery.length > 0,
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
  const closeSessionAtom = useMemo(() => client.rpc.mutation("CloseSession"), [client])
  const reopenSessionAtom = useMemo(() => client.rpc.mutation("ReopenSession"), [client])
  const revealProjectAtom = useMemo(() => client.rpc.mutation("RevealProjectSource"), [client])
  const closeSession = useAtomSet(closeSessionAtom, {
    mode: "promise",
  })
  const reopenSession = useAtomSet(reopenSessionAtom, { mode: "promise" })
  const revealProject = useAtomSet(revealProjectAtom, { mode: "promise" })
  const handleCompose = () => {
    setSettingsTab(null)
    startNewSession()
    if (props?.overlay && props.onCloseOverlay) props.onCloseOverlay()
  }
  return (
    <SessionsSidebar
      projects={projects.projects}
      revealKind={projects.revealKind}
      loading={projects.loading || sessionPage.loading}
      sessions={sessionPage.sessions.map((s) => {
        const liveStatus = activeSessionStatuses[s.id]
        const statusFields = liveStatus
          ? {
              updatedAt: liveStatus.lastMessageAt,
              workStatus: liveStatus.workStatus,
            }
          : {
              updatedAt: s.timestamp,
              workStatus: "idle" as const,
            }
        return {
          sessionId: s.id,
          projectId: s.projectId,
          title: s.title,
          cwd: s.workingDirectory,
          sidebarOpen: s.sidebarOpen,
          ...statusFields,
        }
      })}
      loadingMore={sessionPage.loadingMore}
      hasMore={sessionPage.hasMore}
      onLoadMore={sessionPage.loadMore}
      onSelectSession={(session) => {
        setSettingsTab(null)
        const select = () => {
          setSelectedProjectId(session.projectId)
          setSelectedCwd(session.cwd)
          resumeSession(session.sessionId)
        }
        if (session.sidebarOpen) {
          select()
          return
        }
        void reopenSession({
          payload: { sessionId: session.sessionId },
          reactivityKeys: ["sessions", "projects"],
        }).then(select).catch(() => showToast("error", "Could not reopen this session."))
      }}
      onCloseSession={(sessionId) => {
        void closeSession({
          payload: { sessionId },
          reactivityKeys: ["sessions", "projects"],
        }).then(() => {
          if (sessionId !== selectedSessionId) return
          const closed = sessionPage.sessions.find((session) => session.id === sessionId)
          startNewSession(closed
            ? { cwd: closed.workingDirectory, projectId: closed.projectId }
            : { cwd: null, projectId: null })
        }).catch(() => showToast("error", "Could not close this session."))
      }}
      onCompose={handleCompose}
      onRevealProject={(projectId) => {
        void revealProject({
          payload: { projectId },
          reactivityKeys: [],
        }).catch(() => showToast("error", "Could not reveal this project folder."))
      }}
      onCreateProject={(project) => {
        setSettingsTab(null)
        startNewSession({ cwd: project.sourceDirectory, projectId: project.projectId })
        props?.onCloseOverlay?.()
      }}
      onEditProject={(project) => {
        if (selectedProjectId !== project.projectId) return
        setSelectedCwd(project.sourceDirectory)
      }}
      onRemoveProject={(project) => {
        if (selectedProjectId !== project.projectId) return
        const next = projects.projects.find(
          (summary) => summary.project.projectId !== project.projectId,
        )?.project
        startNewSession(next
          ? { cwd: next.sourceDirectory, projectId: next.projectId }
          : { cwd: null, projectId: null })
      }}
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
      titlebarIntegrated={props?.titlebarIntegrated}
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
  footer,
}: {
  docked?: boolean
  footer?: ReactNode
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
      footer={footer}
    />
  )
}

/** FooterBar container */
function FooterBarContainer({
  slotProfiles,
}: {
  slotProfiles: SlotProfiles | null
}): ReactNode {
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
  const memoryLabel =
    currentModel._tag === "Running" && residentBytes !== null
      ? `${formatBytes(residentBytes)} mem`
      : null
  return (
    <FooterBar
      context={context}
      tokenCap={tokenCap}
      model={modelLabel}
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
    <div className="mx-auto my-[14px] flex w-[calc(100%-24px)] max-w-[800px] shrink-0 flex-col gap-2">
      <WorkStatusBarContainer slotProfiles={slotProfiles} />
      <ComposerContainer
        docked
        footer={<FooterBarContainer slotProfiles={slotProfiles} />}
      />
    </div>
  )
}
function ChatTitleBar({
  onOpenSidebar,
  desktop = false,
  onCompose,
  showTitle = true,
}: {
  onOpenSidebar?: () => void
  desktop?: boolean
  onCompose?: () => void
  showTitle?: boolean
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
  const sidebarCollapsed = useAtomValue(sidebarCollapsedAtom)
  const sidebarWidth = useAtomValue(sidebarWidthAtom)
  const setSidebarCollapsed = useAtomSet(sidebarCollapsedAtom)
  if (desktop) {
    const titlebarActions = (
      <>
        <button
          type="button"
          onClick={() => setSidebarCollapsed(!sidebarCollapsed)}
          className="flex size-8 shrink-0 items-center justify-center rounded-md border-0 bg-transparent text-slate-600 hover:bg-white dark:text-slate-400 dark:hover:bg-slate-800 [-webkit-app-region:no-drag]"
          aria-label={sidebarCollapsed ? "Expand sidebar" : "Collapse sidebar"}
          title={sidebarCollapsed ? "Expand sidebar" : "Collapse sidebar"}
        >
          <SidebarSimple size={18} />
        </button>
        <button
          type="button"
          onClick={onCompose}
          className="flex size-8 shrink-0 items-center justify-center rounded-md border-0 bg-transparent text-slate-600 hover:bg-white dark:text-slate-400 dark:hover:bg-slate-800 [-webkit-app-region:no-drag]"
          aria-label="New chat"
          title="New chat"
        >
          <NotePencil size={18} />
        </button>
      </>
    )

    return (
      <div
        className="relative h-11 shrink-0 bg-slate-50 dark:bg-slate-875 select-none [-webkit-app-region:drag]"
        title={title}
      >
        {!sidebarCollapsed ? (
          <div
            className="absolute inset-y-0 left-0 flex items-center justify-end gap-1 border-r border-slate-200 bg-slate-100 px-3 dark:border-slate-800 dark:bg-slate-875"
            style={{ width: sidebarWidth }}
          >
            {titlebarActions}
          </div>
        ) : (
          <div className="ml-[env(titlebar-area-x,_0px)] flex h-full w-[env(titlebar-area-width,_100%)] items-center gap-1 px-3 mac:pl-[84px]">
            {titlebarActions}
            {showTitle ? (
              <span className="ml-3 min-w-0 max-w-[60%] overflow-hidden text-ellipsis whitespace-nowrap font-sans text-[15px] font-medium text-slate-900 dark:text-slate-200">
                {title}
              </span>
            ) : null}
          </div>
        )}
        {showTitle && !sidebarCollapsed ? (
          <span
            className="absolute top-0 flex h-11 min-w-0 max-w-[60%] items-center overflow-hidden text-ellipsis whitespace-nowrap font-sans text-[15px] font-medium text-slate-900 dark:text-slate-200"
            style={{ left: sidebarWidth + 16 }}
          >
            {title}
          </span>
        ) : null}
      </div>
    )
  }
  return (
    <div
      className="h-11 shrink-0 flex items-center px-4 bg-slate-50 dark:bg-slate-875 select-none"
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
function AppInner({
  initialAcnLifecycle,
}: {
  readonly initialAcnLifecycle: AcnLifecycleState
}): ReactNode {
  // Detect responsive mode (≤640px) — no useEffect, uses matchMedia store
  const isNarrow = useSyncExternalStore(subscribeResponsive, getIsNarrow)
  useMenuActions()
  useInterruptAllListener()
  const platform = usePlatform()
  const acnLifecycle = useAcnLifecycle(initialAcnLifecycle)
  const onboarding = useOnboardingModelSetup()
  if (acnLifecycle.state._tag !== "Ready") {
    return (
      <AcnBootstrapScreen
        state={acnLifecycle.state}
        onRetry={acnLifecycle.retry}
        {...(platform.quit === undefined ? {} : { onQuit: platform.quit })}
      />
    )
  }
  const onboardingState = Option.getOrNull(Result.value(onboarding.view))
  if (Result.isFailure(onboarding.view)) {
    const failureDescription = Option.match(
      Cause.failureOption(onboarding.view.cause),
      {
        onNone: () => "Local model settings are temporarily unavailable.",
        onSome: (failure) => {
          switch (failure.source) {
            case "onboarding":
              return "Magnitude couldn’t read onboarding status from the daemon."
            case "local-models":
              return "Magnitude couldn’t load the local model catalog."
            case "model-slots":
              return "Magnitude couldn’t read the configured local model."
          }
        },
      }
    )
    return (
      <div className="min-h-screen flex flex-col items-center justify-center gap-2.5 p-8 text-center bg-slate-50 dark:bg-slate-875 text-slate-900 dark:text-slate-200 [&_h1]:mt-1 [&_h1]:text-[22px] [&_p]:mb-2 [&_p]:text-slate-600 dark:[&_p]:text-slate-400">
        <AlertTriangleIcon />
        <h1>Couldn’t load local setup</h1>
        <p>{failureDescription}</p>
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
      <div className="flex min-h-screen flex-col items-center justify-center bg-slate-50 p-8 text-center text-slate-900 dark:bg-slate-875 dark:text-slate-200">
        <MagnitudeMark className="mb-6 h-auto w-[82px]" />
        <h1 className="text-[30px] font-semibold leading-tight tracking-[-0.025em]">
          Opening Magnitude
        </h1>
        <div className="mt-5 flex items-center gap-2.5 text-[16px] leading-7 text-slate-600 dark:text-slate-300">
          <div className="size-[17px] rounded-full border-2 border-slate-300 border-t-blue-700 animate-spin motion-reduce:animate-none dark:border-slate-750 dark:border-t-blue-500" />
          <p>Loading local model settings…</p>
        </div>
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
  const setSettingsTab = useAtomSet(settingsTabAtom)
  const { startNewSession } = useSessionActions()
  const controller = useDisplayViewController()
  const forkStack = controller.expandedForkStack
  const panelOpen = settingsTab !== null
  const workerDetailOpen = !panelOpen && forkStack.length > 0
  return (
    <div
      className={`${
        isDesktop ? "[background:transparent]" : "bg-slate-50 dark:bg-slate-875"
      } app flex h-screen flex-col overflow-hidden`}
    >
      {isDesktop ? (
        <ChatTitleBar
          desktop
          showTitle={!panelOpen}
          onCompose={() => {
            setSettingsTab(null)
            startNewSession()
          }}
        />
      ) : null}
      <div className="flex min-h-0 flex-1 overflow-hidden">
        {/* Docked sidebar — hidden by CSS when narrow */}
        {!isNarrow && <SessionsSidebarContainer titlebarIntegrated={isDesktop} />}
        {/* Overlay sidebar — shown when narrow + visible */}
        {showOverlaySidebar && (
          <SessionsSidebarContainer
            overlay
            onCloseOverlay={() => setSidebarVisible(false)}
          />
        )}
        <div className="chat-column [flex:1] min-w-0 flex flex-col relative bg-slate-50 dark:bg-slate-875">
        {/* Main chat column — always mounted, always in the layout. When a
            panel or worker detail is open, it's covered by an absolute
            overlay. Keeping it in the layout (not display:none) preserves
            scroll metrics so the scroll controller can capture and restore
            the correct position across overlay navigation. */}
        <div className="flex flex-col [flex:1] min-h-0">
          {!isDesktop ? (
            <ChatTitleBar
              onOpenSidebar={isNarrow ? () => setSidebarVisible(true) : undefined}
            />
          ) : null}
          <ChatTimeline isVisible={!panelOpen && !workerDetailOpen} />
          <BottomDockContainer slotProfiles={slotProfiles} />
        </div>
        {(panelOpen || workerDetailOpen) && (
          <div className="absolute [inset:0px] flex flex-col bg-slate-50 dark:bg-slate-875 z-[1]">
            {panelOpen && (
              <>
                {isNarrow && (
                  <button
                    type="button"
                    className="appearance-none min-h-8 rounded-[7px] px-3 inline-flex items-center justify-center gap-1.5 font-sans text-xs font-semibold leading-none cursor-pointer disabled:cursor-not-allowed disabled:opacity-50 focus-visible:outline focus-visible:outline-1 focus-visible:outline-offset-1 focus-visible:outline-blue-700 dark:focus-visible:outline-blue-500 w-8 !px-0 bg-transparent text-slate-600 dark:text-slate-400 border border-slate-300 dark:border-slate-750 hover:bg-slate-150 hover:text-slate-900 dark:hover:bg-slate-750 dark:hover:text-slate-200 absolute top-3 left-3 z-[4] bg-slate-50 dark:bg-slate-875"
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
    </div>
  )
}
export function App({
  initialAcnLifecycle,
}: {
  readonly initialAcnLifecycle: AcnLifecycleState
}): ReactNode {
  return (
    <DisplayViewControllerProvider>
      <AppInner initialAcnLifecycle={initialAcnLifecycle} />
    </DisplayViewControllerProvider>
  )
}
