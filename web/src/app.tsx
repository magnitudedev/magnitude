/**
 * App root component — spec §9.2
 *
 * Wraps the component tree in DisplayViewControllerProvider.
 * Wires the display view controller, session list, composer, and panels.
 *
 * Boundary operations are members of useAgentClient() (client.Sessions.GetSession(input),
 * client.Agent.Interrupt); StreamDisplayView is consumed by the display view controller.
 * Local UI state uses plain atoms (spec §6.3).
 */
import {
  useCallback,
  useMemo,
  useRef,
  useSyncExternalStore,
  type ReactNode,
} from "react"
import { Menu, PanelRight } from "lucide-react"
import { Gear, NotePencil, SidebarSimple } from "@phosphor-icons/react"
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
  useOnboardingModelSetup,
  useAcnLifecycle,
  useLocalModels,
  useModelSlots,
  useModelConfig,
  useProviderModelCatalog,
  deriveCurrentLocalModel,
  deriveLocalModelLoadActivity,
  installedLocalModels,
  localModelProviderModelId,
  formatLocalModelDisplayName,
  selectedSlotModel,
  reasoningEffortControl,
  formatReasoningEffort,
  useActiveSessionStatuses,
} from "@magnitudedev/client-common"
import { SessionsSidebar } from "./components/sessions-sidebar"
import { ChatTimeline } from "./components/chat-timeline"
import { Composer } from "./components/composer"
import {
  FooterBar,
  type FooterModelOptionsState,
} from "./components/footer-bar"
import { WorkspacePanel } from "./components/workspace-panel"
import { WorkerDetailPanel } from "./components/worker-detail-panel"
import { ContextUsageIndicator } from "./components/context-usage-indicator"
import { SettingsCenter } from "./components/settings-center"
import { LocalModelOnboarding } from "./components/local-model-onboarding"
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
  workspacePanelEnteringAtom,
  workspacePanelOpenAtom,
  workspacePresentationAtom,
} from "./state/web-atoms"
import { addBrowserTab, addEmptyFileTab, changeWorkspaceProject, makeWorkspaceTabId } from "@/lib/workspace-tabs"
import { useMenuActions } from "./hooks/use-menu-actions"
import { DaemonConnectionError } from "./components/daemon-connection-error"
import { AcnBootstrapScreen } from "./components/acn-bootstrap-screen"
import { MagnitudeMark } from "./components/magnitude-mark"
import { Button } from "@/components/ui/button"
import { Spinner } from "@/components/ui/spinner"
import { Toaster } from "@/components/ui/toast"
import { ActionTooltip, TooltipProvider } from "@/components/ui/tooltip"
import { notify } from "@/lib/notifications"
import { subscribeResponsive, getIsNarrow } from "./stores/responsive-store"
import { useInitializeConversationPreferences } from "./stores/conversation-preferences"
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
  ProviderModelCatalogLifecycle,
  ReasoningEffortSchema,
  ROLE_TO_SLOT,
  SECONDARY_SLOT_ID,
} from "@magnitudedev/sdk"
import type {
  AcnLifecycleState,
  DisplayActor,
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

/** Sessions sidebar container — session/project actions around the sidebar's own queries */
function SessionsSidebarContainer(props?: {
  overlay?: boolean
  onCloseOverlay?: () => void
  titlebarIntegrated?: boolean
}): ReactNode {
  const client = useAgentClient()
  const { startNewSession, resumeSession } = useSessionActions()
  const selectedSessionId = useSelectedSessionId()
  const activeSessionStatuses = useActiveSessionStatuses()
  const settingsTab = useAtomValue(settingsTabAtom)
  const setSettingsTab = useAtomSet(settingsTabAtom)
  const selectedProjectId = useAtomValue(selectedProjectIdAtom)
  const setSelectedCwd = useAtomSet(selectedCwdAtom)
  const setSelectedProjectId = useAtomSet(selectedProjectIdAtom)

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
  const archiveSession = useAtomSet(client.Sessions.ArchiveSession, {
    mode: "promise",
  })
  const setSessionPinned = useAtomSet(client.Sessions.SetSessionPinned, { mode: "promise" })
  const revealProject = useAtomSet(client.Projects.RevealProjectSource, { mode: "promise" })
  const handleCompose = () => {
    setSettingsTab(null)
    startNewSession()
    if (props?.overlay && props.onCloseOverlay) props.onCloseOverlay()
  }
  return (
    <SessionsSidebar
      liveStatuses={activeSessionStatuses}
      onSelectSession={(session, project) => {
        setSettingsTab(null)
        setSelectedProjectId(project?.projectId ?? null)
        setSelectedCwd(session.cwd)
        resumeSession(session.sessionId)
      }}
      onArchiveSession={(session) => {
        void archiveSession({ sessionId: session.sessionId }).then(() => {
          if (session.sessionId !== selectedSessionId) return
          startNewSession({ cwd: session.cwd, projectId: null })
        }).catch(() => notify("error", "Could not archive this session."))
      }}
      onSetSessionPinned={(sessionId, pinned) => {
        void setSessionPinned({ sessionId, pinned })
          .catch(() => notify("error", `Could not ${pinned ? "pin" : "unpin"} this session.`))
      }}
      onCompose={handleCompose}
      onRevealProject={(projectId) => {
        void revealProject({ projectId })
          .catch(() => notify("error", "Could not reveal this project folder."))
      }}
      onCreateProject={(project) => {
        setSettingsTab(null)
        startNewSession({ cwd: project.cwd, projectId: project.projectId })
        props?.onCloseOverlay?.()
      }}
      onEditProject={(project) => {
        if (selectedProjectId !== project.projectId) return
        setSelectedCwd(project.cwd)
      }}
      onRemoveProject={(project, next) => {
        if (selectedProjectId !== project.projectId) return
        startNewSession(next
          ? { cwd: next.cwd, projectId: next.projectId }
          : { cwd: null, projectId: null })
      }}
      onOpenSettings={() => {
        setSettingsTab("general")
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
  const currentModel = slots === null
    ? null
    : deriveCurrentLocalModel(Option.some(slots.slots.primary))
  const disabledReason = currentModel?._tag === "NoSelection"
    ? "Choose a model before sending"
    : null
  const commandContext: CommandContext = useMemo(
    () => ({
      resetConversation: () => startNewSession(),
      showSystemMessage: (message: string) => notify("info", message),
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
        notify(
          "info",
          "Project initialization is not available in the web app yet."
        )
      },
      openSettings: () => setSettingsTab("general"),
      openSetup: onboarding.open,
      openModelMenu: (menu) => {
        if (menu === "models" || menu === "catalog" || menu === "hardware") {
          setSettingsTab(menu)
        }
      },
      toggleAutopilot: () => {
        notify("info", "Autopilot mode is not yet available in the web app.")
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
      key={`${composer.sessionId ?? "draft"}:${composer.cwd ?? ""}`}
      role={composer.roleLabel}
      isStreaming={composer.isStreaming}
      bashMode={composer.bashMode}
      onSend={(text, mentions, uploads) => {
        void composer.handleSend(text, {
          mentions,
          uploads,
        })
      }}
      onAttachmentError={(message) => notify("error", message)}
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
  const hasMessages = useDisplayState(
    (state) => (state.timelines.root?.messages.order.length ?? 0) > 0
  )
  const context = useDisplayState(
    (state) => state.actors["root"]?.context ?? null
  )
  const { profile } = useRootSlotProfile(slotProfiles)
  const tokenCap = profile?.contextWindow ?? null
  const bashMode = useAtomValue(bashModeAtom)
  const nextEscWillKillAll = useAtomValue(nextEscWillKillAllAtom)
  const localModelsResult = useLocalModels()
  const slotsResult = useModelSlots()
  const catalogResult = useProviderModelCatalog()
  const modelConfig = useModelConfig()
  const slots = Option.getOrNull(Result.value(slotsResult))
  const currentModel = deriveCurrentLocalModel(
    Option.fromNullable(slots?.slots.primary)
  )
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
  const thinkingLevel = thinkingOptions.length > 0 && profile?.reasoningEffort
    ? formatReasoningEffort(profile.reasoningEffort)
    : null
  const localModels = Option.getOrNull(Result.value(localModelsResult))
  const providerCatalog = Option.match(Result.value(catalogResult), {
    onNone: () => ({ _tag: "Loading" as const, models: [] }),
    onSome: (state) => ProviderModelCatalogLifecycle.match(state, {
      Loading: () => ({ _tag: "Loading" as const, models: [] }),
      Ready: ({ models }) => ({ _tag: "Ready" as const, models }),
      Refreshing: ({ models }) => ({ _tag: "Loading" as const, models }),
      Degraded: ({ models }) => ({ _tag: "Degraded" as const, models }),
      Unavailable: () => ({ _tag: "Failed" as const, models: [] }),
    }),
  })
  const modelOptions = Option.match(Result.value(localModelsResult), {
    onNone: () => [],
    onSome: (state) =>
      installedLocalModels(state)
        .flatMap((model) => {
          const providerModelId = Option.getOrUndefined(localModelProviderModelId(model))
          if (providerModelId === undefined) return []
          const providerModel = providerCatalog.models.find((candidate) =>
            candidate.providerId === "local"
            && candidate.providerModelId === providerModelId)
          // The compound picker commits model and reasoning atomically. A model
          // is not selectable until its authoritative reasoning capabilities
          // have arrived from the provider catalog.
          if (providerModel === undefined) return []
          const thinkingControl = reasoningEffortControl(providerModel)
          return [
            {
              value: providerModelId,
              label: formatLocalModelDisplayName(model),
              thinkingOptions: thinkingControl._tag === "Available" ? thinkingControl.options : [],
              defaultThinkingEffort: Option.getOrElse(
                providerModel.capabilities.reasoning.defaultEffort,
                () => ReasoningEffortSchema.make("none"),
              ),
            },
          ]
        })
        .sort((left, right) => left.label.localeCompare(right.label)),
  })
  const modelOptionsState: FooterModelOptionsState =
    Result.isFailure(localModelsResult) ||
    Result.isFailure(catalogResult) ||
    providerCatalog._tag === "Failed"
      ? { _tag: "Failed", options: modelOptions }
      : providerCatalog._tag === "Degraded"
      ? { _tag: "Degraded", options: modelOptions }
      : localModels === null ||
        !localModels.preparation.discovery.complete ||
        providerCatalog._tag === "Loading"
      ? { _tag: "Loading", options: modelOptions }
      : { _tag: "Ready", options: modelOptions }
  const primarySlot = slots?.slots.primary
  const selectedModelId =
    primarySlot && primarySlot._tag !== "Unassigned"
      ? primarySlot.selection.providerModelId
      : null
  const modelLabel =
    currentModel._tag === "NoSelection"
      ? "Choose model"
      : currentModel.displayName
  return (
    <FooterBar
      context={context}
      showContext={hasMessages}
      tokenCap={tokenCap}
      model={modelLabel}
      thinkingLevel={thinkingLevel}
      thinkingEffort={profile?.reasoningEffort ?? null}
      thinkingOptions={thinkingOptions}
      modelOptionsState={modelOptionsState}
      selectedModelId={selectedModelId}
      onSelectionCommit={(providerModelId, reasoningEffort) => {
        modelConfig.updateSlotSelection(PRIMARY_SLOT_ID, {
          providerId: ProviderIdSchema.make("local"),
          providerModelId,
          reasoningEffort,
        })
      }}
      onThinkingSelect={(effort) => {
        modelConfig.updateSlotReasoning(PRIMARY_SLOT_ID, effort)
      }}
      bashMode={bashMode}
      nextEscWillKillAll={nextEscWillKillAll}
    />
  )
}
function BottomDockContainer({
  slotProfiles,
}: {
  slotProfiles: SlotProfiles | null
}): ReactNode {
  return (
    <div className="mx-auto my-[14px] flex w-[calc(100%-24px)] max-w-[800px] shrink-0 flex-col">
      <ComposerContainer
        docked
        footer={<FooterBarContainer slotProfiles={slotProfiles} />}
      />
    </div>
  )
}
function ChatTitleBar({
  onOpenSidebar,
  onOpenWorkspacePanel,
  workspacePanelExpanded = false,
  workspacePanelAvailable,
  desktop = false,
  onCompose,
  showTitle = true,
}: {
  onOpenSidebar?: () => void
  onOpenWorkspacePanel: () => void
  workspacePanelExpanded?: boolean
  workspacePanelAvailable: boolean
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
        ? Atom.make((get) => get(client.Sessions.GetSession({ sessionId: selectedSessionId })).result)
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
  const settingsTab = useAtomValue(settingsTabAtom)
  const setSettingsTab = useAtomSet(settingsTabAtom)
  const workspacePanelButton = (
    <ActionTooltip
      label="Expand sidebar"
      side="bottom"
      trigger={
        <span
          className="inline-flex [-webkit-app-region:no-drag]"
          tabIndex={!workspacePanelAvailable ? 0 : undefined}
          aria-label={!workspacePanelAvailable ? "Expand sidebar" : undefined}
        >
          <Button variant="unstyled" size="unstyled" type="button"
            onClick={onOpenWorkspacePanel}
            disabled={!workspacePanelAvailable}
            className="flex size-8 shrink-0 items-center justify-center rounded-md border-0 bg-transparent text-slate-600 hover:bg-slate-150 dark:text-slate-400 dark:hover:bg-slate-800"
            aria-label="Expand sidebar"
          ><PanelRight size={18} /></Button>
        </span>
      }
    />
  )
  if (desktop) {
    const titlebarActions = (
      <>
        <ActionTooltip
          label="Settings"
          side="bottom"
          trigger={
            <Button variant="unstyled" size="unstyled"
              type="button"
              onClick={() => {
                if (settingsTab !== null) {
                  setSettingsTab(null)
                  return
                }
                setSidebarCollapsed(false)
                setSettingsTab("general")
              }}
              className="flex size-8 shrink-0 items-center justify-center rounded-md border-0 bg-transparent text-slate-600 hover:bg-white aria-[current=page]:bg-slate-200 aria-[current=page]:text-blue-700 dark:text-slate-400 dark:hover:bg-slate-800 dark:aria-[current=page]:bg-slate-750 dark:aria-[current=page]:text-blue-400 [-webkit-app-region:no-drag]"
              aria-label="Settings"
              aria-current={settingsTab !== null ? "page" : undefined}
            >
              <Gear size={17} />
            </Button>
          }
        />
        <ActionTooltip
          label={sidebarCollapsed ? "Expand sidebar" : "Collapse sidebar"}
          side="bottom"
          trigger={
            <Button variant="unstyled" size="unstyled"
              type="button"
              onClick={() => setSidebarCollapsed(!sidebarCollapsed)}
              className="flex size-8 shrink-0 items-center justify-center rounded-md border-0 bg-transparent text-slate-600 hover:bg-white dark:text-slate-400 dark:hover:bg-slate-800 [-webkit-app-region:no-drag]"
              aria-label={sidebarCollapsed ? "Expand sidebar" : "Collapse sidebar"}
            >
              <SidebarSimple size={18} />
            </Button>
          }
        />
        <ActionTooltip
          label="New chat"
          side="bottom"
          trigger={
            <Button variant="unstyled" size="unstyled"
              type="button"
              onClick={onCompose}
              className="flex size-8 shrink-0 items-center justify-center rounded-md border-0 bg-transparent text-slate-600 hover:bg-white dark:text-slate-400 dark:hover:bg-slate-800 [-webkit-app-region:no-drag]"
              aria-label="New chat"
            >
              <NotePencil size={18} />
            </Button>
          }
        />
      </>
    )

    return (
      <div
        className="relative h-11 shrink-0 bg-slate-50 dark:bg-slate-900 select-none [-webkit-app-region:drag]"
        title={title}
      >
        {!sidebarCollapsed ? (
          <div
            className="absolute inset-y-0 left-0 flex items-center justify-end gap-1 border-r border-slate-200 bg-slate-100 px-3 dark:border-slate-800 dark:bg-slate-850"
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
        {!workspacePanelExpanded ? (
          <div className="absolute inset-y-0 right-2 flex items-center">{workspacePanelButton}</div>
        ) : null}
      </div>
    )
  }
  return (
    <div
      className="h-11 shrink-0 flex items-center px-4 bg-slate-50 dark:bg-slate-900 select-none"
      title={title}
    >
      {onOpenSidebar && (
        <ActionTooltip
          label="Open sessions"
          side="bottom"
          trigger={
            <Button variant="unstyled" size="unstyled"
              type="button"
              className="appearance-none min-h-8 rounded-[7px] px-3 inline-flex items-center justify-center gap-1.5 font-sans text-xs font-semibold leading-none cursor-pointer disabled:cursor-not-allowed disabled:opacity-50 focus-visible:outline focus-visible:outline-1 focus-visible:outline-offset-1 focus-visible:outline-blue-700 dark:focus-visible:outline-blue-500 w-8 !px-0 bg-transparent text-slate-600 dark:text-slate-400 border border-slate-300 dark:border-slate-750 hover:bg-slate-150 hover:text-slate-900 dark:hover:bg-slate-750 dark:hover:text-slate-200 shrink-0 mr-2.5"
              aria-label="Open sessions"
              onClick={onOpenSidebar}
            >
              <Menu size={17} />
            </Button>
          }
        />
      )}
      <span className="min-w-0 max-w-[60%] overflow-hidden text-ellipsis whitespace-nowrap text-slate-900 dark:text-slate-200 font-sans text-[15px] font-medium">
        {title}
      </span>
      {!workspacePanelExpanded ? <div className="ml-auto">{workspacePanelButton}</div> : null}
    </div>
  )
}

/** Listen for __magnitude:interrupt-all custom event → Interrupt RPC with target: all */
function useInterruptAllListener(): void {
  const client = useAgentClient()
  const selectedSessionId = useSelectedSessionId()
  const interruptMutation = useAtomSet(client.Agent.Interrupt)
  const interruptAtom = useMemo(
    () =>
      Atom.make(
        Effect.gen(function* () {
          const handler = () => {
            if (!selectedSessionId) return
            interruptMutation({
              sessionId: selectedSessionId,
              target: {
                _tag: "all",
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
          if (failure._tag !== "OnboardingModelSetupObservationFailed") {
            return "Local model settings are temporarily unavailable."
          }
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
      <div className="min-h-screen flex flex-col items-center justify-center gap-2.5 p-8 text-center bg-slate-50 dark:bg-slate-900 text-slate-900 dark:text-slate-200 [&_h1]:mt-1 [&_h1]:text-[22px] [&_p]:mb-2 [&_p]:text-slate-600 dark:[&_p]:text-slate-400">
        <AlertTriangleIcon />
        <h1>Couldn’t load local setup</h1>
        <p>{failureDescription}</p>
        <Button variant="unstyled" size="unstyled"
          className="appearance-none min-h-8 rounded-[7px] px-3 inline-flex items-center justify-center gap-1.5 font-sans text-xs font-semibold leading-none cursor-pointer disabled:cursor-not-allowed disabled:opacity-50 focus-visible:outline focus-visible:outline-1 focus-visible:outline-offset-1 focus-visible:outline-blue-700 dark:focus-visible:outline-blue-500 bg-blue-700 text-slate-50 hover:bg-blue-800 dark:bg-blue-500 dark:text-slate-925 dark:hover:bg-blue-400"
          type="button"
          onClick={onboarding.retry}
        >
          Retry
        </Button>
      </div>
    )
  }
  if (onboardingState === null) {
    return (
      <div className="flex min-h-screen flex-col items-center justify-center bg-slate-50 p-8 text-center text-slate-900 dark:bg-slate-900 dark:text-slate-200">
        <MagnitudeMark className="mb-6 h-auto w-[82px]" />
        <h1 className="text-[30px] font-semibold leading-tight tracking-[-0.025em]">
          Opening Magnitude
        </h1>
        <div className="mt-5 flex items-center gap-2.5 text-[16px] leading-7 text-slate-600 dark:text-slate-300">
          <Spinner className="size-[17px] text-blue-700 motion-reduce:animate-none dark:text-blue-500" />
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
  useInitializeConversationPreferences()
  const connectionError = useDisplayConnectionError()
  const platform = usePlatform()
  const isDesktop = platform.id === "desktop"
  const sidebarVisible = useAtomValue(sidebarVisibleAtom)
  const setSidebarVisible = useAtomSet(sidebarVisibleAtom)
  const {
    profiles: slotProfiles,
    slots: slotsResult,
    rootSlotId,
    rootProfile,
  } = useSlotProfiles()
  const modelSlots = Option.getOrNull(Result.value(slotsResult))
  const modelLoadActivity = modelSlots === null
    ? null
    : deriveLocalModelLoadActivity(modelSlots, rootSlotId)
  const rootActor = useDisplayState((state) => state.actors["root"] ?? null)
  const rootStatus = rootActor?.kind === "root" ? rootActor.status : null
  const showOverlaySidebar = isNarrow && sidebarVisible
  const settingsTab = useAtomValue(settingsTabAtom)
  const workspacePanelOpen = useAtomValue(workspacePanelOpenAtom)
  const setWorkspacePanelOpen = useAtomSet(workspacePanelOpenAtom)
  const setWorkspacePanelEntering = useAtomSet(workspacePanelEnteringAtom)
  const workspacePresentation = useAtomValue(workspacePresentationAtom)
  const setWorkspacePresentation = useAtomSet(workspacePresentationAtom)
  const selectedProjectId = useAtomValue(selectedProjectIdAtom)
  const setSettingsTab = useAtomSet(settingsTabAtom)
  const { startNewSession } = useSessionActions()
  const controller = useDisplayViewController()
  const forkStack = controller.expandedForkStack
  const panelOpen = settingsTab !== null
  const workerDetailOpen = !panelOpen && forkStack.length > 0
  const browser = platform.embeddedBrowser
  const filesAvailable = selectedProjectId !== null
  const browserAvailable = browser !== undefined
  const workspacePanelAvailable = filesAvailable || browserAvailable
  const workspacePanelExpanded = !panelOpen
    && !workerDetailOpen
    && workspacePanelOpen
    && workspacePanelAvailable
  const openWorkspacePanel = () => {
    setSettingsTab(null)
    const compatibleTabs = changeWorkspaceProject(workspacePresentation, selectedProjectId).tabs
    if (compatibleTabs.length === 0) {
      if (selectedProjectId !== null) {
        setWorkspacePresentation((current) => {
          const scoped = changeWorkspaceProject(current, selectedProjectId)
          return scoped.tabs.length === 0
            ? addEmptyFileTab(scoped, makeWorkspaceTabId(), selectedProjectId)
            : scoped
        })
      } else if (browser !== undefined) {
        void browser.createTab().then((browserTabId) => {
          setWorkspacePresentation((current) => addBrowserTab(current, makeWorkspaceTabId(), browserTabId))
        }).catch((cause: unknown) => {
          console.error("[workspace] Could not create a browser tab.", cause)
          notify("error", "Could not create a browser tab.")
        })
      }
    }
    if (!workspacePanelOpen) {
      setWorkspacePanelEntering(true)
      setWorkspacePanelOpen(true)
    }
  }
  return (
    <div
      className={`${
        isDesktop ? "[background:transparent]" : "bg-slate-50 dark:bg-slate-900"
      } app relative flex h-screen overflow-hidden`}
    >
      <div className="flex min-w-0 flex-1 flex-col overflow-hidden">
        {isDesktop ? (
          <ChatTitleBar
            desktop
            showTitle={!panelOpen}
            workspacePanelExpanded={workspacePanelExpanded}
            workspacePanelAvailable={workspacePanelAvailable}
            onOpenWorkspacePanel={openWorkspacePanel}
            onCompose={() => {
              setSettingsTab(null)
              startNewSession()
            }}
          />
        ) : null}
        <div className="relative flex min-h-0 flex-1 overflow-hidden">
        {/* Docked sidebar — hidden by CSS when narrow */}
        {!isNarrow && <SessionsSidebarContainer titlebarIntegrated={isDesktop} />}
        {/* Overlay sidebar — shown when narrow + visible */}
        {showOverlaySidebar && (
          <SessionsSidebarContainer
            overlay
            onCloseOverlay={() => setSidebarVisible(false)}
          />
        )}
        <div className="chat-column [flex:1] min-w-0 flex flex-col relative bg-slate-50 dark:bg-slate-900">
        {/* Main chat column — always mounted, always in the layout. When a
            panel or worker detail is open, it's covered by an absolute
            overlay. Keeping it in the layout (not display:none) preserves
            scroll metrics so the scroll controller can capture and restore
            the correct position across overlay navigation. */}
        <div className="flex flex-col [flex:1] min-h-0">
          {!isDesktop ? (
            <ChatTitleBar
              onOpenSidebar={isNarrow ? () => setSidebarVisible(true) : undefined}
              workspacePanelExpanded={workspacePanelExpanded}
              workspacePanelAvailable={workspacePanelAvailable}
              onOpenWorkspacePanel={openWorkspacePanel}
            />
          ) : null}
          <ChatTimeline
            isVisible={!panelOpen && !workerDetailOpen}
            rootStatus={rootStatus}
            modelLoadActivity={modelLoadActivity}
            modelName={rootProfile?.modelDisplayName ?? null}
          />
          <BottomDockContainer
            slotProfiles={slotProfiles}
          />
        </div>
        {(panelOpen || workerDetailOpen) && (
          <div className="absolute [inset:0px] flex flex-col bg-slate-50 dark:bg-slate-900 z-[1]">
            {panelOpen && (
              <>
                {isNarrow && (
                  <ActionTooltip
                    label="Open settings navigation"
                    side="right"
                    trigger={
                      <Button variant="unstyled" size="unstyled"
                        type="button"
                        className="appearance-none min-h-8 rounded-[7px] px-3 inline-flex items-center justify-center gap-1.5 font-sans text-xs font-semibold leading-none cursor-pointer disabled:cursor-not-allowed disabled:opacity-50 focus-visible:outline focus-visible:outline-1 focus-visible:outline-offset-1 focus-visible:outline-blue-700 dark:focus-visible:outline-blue-500 w-8 !px-0 bg-transparent text-slate-600 dark:text-slate-400 border border-slate-300 dark:border-slate-750 hover:bg-slate-150 hover:text-slate-900 dark:hover:bg-slate-750 dark:hover:text-slate-200 absolute top-3 left-3 z-[4] bg-slate-50 dark:bg-slate-900"
                        aria-label="Open settings navigation"
                        onClick={() => setSidebarVisible(true)}
                      >
                        <Menu size={17} />
                      </Button>
                    }
                  />
                )}
                <SettingsCenter tab={settingsTab} />
              </>
            )}
            {workerDetailOpen && (
              <WorkerDetailPageContainer slotProfiles={slotProfiles} />
            )}
          </div>
        )}
        <Toaster />
        </div>
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
      {workspacePanelExpanded
        ? <WorkspacePanel projectId={selectedProjectId} browser={browser} />
        : null}
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
      <TooltipProvider>
        <AppInner initialAcnLifecycle={initialAcnLifecycle} />
      </TooltipProvider>
    </DisplayViewControllerProvider>
  )
}
