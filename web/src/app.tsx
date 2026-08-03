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
import { useCallback, useEffect, useMemo, useRef, useState, useSyncExternalStore, type ReactNode } from "react"
import { Option, Effect, Runtime, Cause } from "effect"
import { useAtomValue, useAtomSet, useAtomMount, Atom, Result } from "@effect-atom/atom-react"
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
import { SettingsPanel, type ApiKeyState } from "./components/settings-panel"
import { ChatColumnPage } from "./components/chat-column-page"
import {
  selectedCwdAtom,
  selectedFilePathAtom,
  settingsOpenAtom,
  usageOpenAtom,
  bashModeAtom,
  nextEscWillKillAllAtom,
} from "@magnitudedev/client-common"
import {
  sidebarSearchAtom,
  sidebarCwdFilterAtom,
  sidebarVisibleAtom,
  apiKeyVerifiedAtom,
} from "./state/web-atoms"
import { useMenuActions } from "./hooks/use-menu-actions"
import { DaemonConnectionError } from "./components/daemon-connection-error"
import { ToastContainer } from "./components/toast"
import { showToast } from "./stores/toast-store"
import { LoginScreen } from "./components/login-screen"
import { subscribeResponsive, getIsNarrow } from "./stores/responsive-store"
import { Loader2 } from "lucide-react"
import {
  useSlotProfiles,
  useModelConfig,
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
  SLOT_DISPLAY_NAMES,
  SLOT_DESCRIPTIONS,
} from "@magnitudedev/sdk"
import type {
  CloudUsageResponse,
  DisplayActor,
  ListSessionsResult,
  ReadFileResult,
  SessionCwdSummary,
  SessionMetadata,
  UsagePeriod,
} from "@magnitudedev/sdk"
import type { SlotId, ProviderAuth } from "@magnitudedev/sdk"

const SESSION_PAGE_SIZE = 50
const MAGNITUDE_PROVIDER_ID = ProviderIdSchema.make("magnitude")

type SessionPageState = {
  sessions: SessionMetadata[]
  nextCursor: string | null
  hasMore: boolean
  loadingMore: boolean
}

function appendUniqueSessions(
  existing: readonly SessionMetadata[],
  incoming: readonly SessionMetadata[],
): SessionMetadata[] {
  const seen = new Set(existing.map((session) => session.sessionId))
  const next = [...existing]
  for (const session of incoming) {
    if (seen.has(session.sessionId)) continue
    seen.add(session.sessionId)
    next.push(session)
  }
  return next
}

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
  role: string | null | undefined,
): SlotProfile | null {
  if (!profiles || !role || !isRoleId(role)) return null
  const slotId = ROLE_TO_SLOT[role] === "primary" ? PRIMARY_SLOT_ID : SECONDARY_SLOT_ID
  return Option.getOrNull(findSlotProfile(profiles, slotId))
}

function useRootSlotProfile(
  slotProfiles: SlotProfiles | null,
): { roleId: string; roleLabel: string; profile: SlotProfile | null } {
  const rootRole = useDisplayState((state) => state.actors["root"]?.role ?? null)
  const roleId = rootRole ?? "leader"
  return {
    roleId,
    roleLabel: formatRoleLabel(roleId),
    profile: findSlotProfileForRole(slotProfiles, roleId),
  }
}

/** Sessions sidebar container — ListSessions query + shared session actions */
function SessionsSidebarContainer(props?: { overlay?: boolean; onCloseOverlay?: () => void }): ReactNode {
  const client = useAgentClient()
  const { startNewSession, resumeSession } = useSessionActions()
  const cwdFilter = useAtomValue(sidebarCwdFilterAtom)
  const setCwdFilter = useAtomSet(sidebarCwdFilterAtom)
  const searchQuery = useAtomValue(sidebarSearchAtom)
  const activeSessionStatuses = useAtomValue(activeSessionStatusesAtom)
  const setSettingsOpen = useAtomSet(settingsOpenAtom)
  const setUsageOpen = useAtomSet(usageOpenAtom)
  const sessionPageGenerationRef = useRef(0)
  const [sessionPage, setSessionPage] = useState<SessionPageState>({
    sessions: [],
    nextCursor: null,
    hasMore: false,
    loadingMore: false,
  })

  const trimmedSearchQuery = searchQuery.trim()
  const firstPageAtom = useMemo(
    () => client.query("ListSessions", {
      cwd: cwdFilter ? Option.some(cwdFilter) : Option.none(),
      query: trimmedSearchQuery ? Option.some(trimmedSearchQuery) : Option.none(),
      cursor: Option.none(),
      limit: SESSION_PAGE_SIZE,
    }, { reactivityKeys: ["sessions"] }),
    [client, cwdFilter, trimmedSearchQuery],
  )
  const firstPageResult = useAtomValue(firstPageAtom)
  const listSessionsMutation = useAtomSet(client.mutation("ListSessions"), { mode: "promise" })

  useEffect(() => {
    sessionPageGenerationRef.current += 1
    if (Result.isSuccess(firstPageResult)) {
      const page = firstPageResult.value as ListSessionsResult
      setSessionPage({
        sessions: [...page.items],
        nextCursor: page.nextCursor._tag === "Some" ? page.nextCursor.value : null,
        hasMore: page.hasMore,
        loadingMore: false,
      })
      return
    }
    if (Result.isInitial(firstPageResult)) {
      setSessionPage((prev) => ({
        ...prev,
        loadingMore: false,
      }))
      return
    }
    setSessionPage({
      sessions: [],
      nextCursor: null,
      hasMore: false,
      loadingMore: false,
    })
  }, [firstPageResult])

  const loadMoreSessions = useCallback(async () => {
    if (sessionPage.loadingMore || !sessionPage.hasMore || !sessionPage.nextCursor) return
    const generation = sessionPageGenerationRef.current
    setSessionPage((prev) => ({ ...prev, loadingMore: true }))
    try {
      const page = await listSessionsMutation({
        payload: {
          cwd: cwdFilter ? Option.some(cwdFilter) : Option.none(),
          query: trimmedSearchQuery ? Option.some(trimmedSearchQuery) : Option.none(),
          cursor: Option.some(sessionPage.nextCursor as string),
          limit: SESSION_PAGE_SIZE,
        },
        reactivityKeys: ["sessions"],
      })
      if (generation !== sessionPageGenerationRef.current) return
      setSessionPage((prev) => ({
        sessions: appendUniqueSessions(prev.sessions, page.items),
        nextCursor: page.nextCursor._tag === "Some" ? page.nextCursor.value : null,
        hasMore: page.hasMore,
        loadingMore: false,
      }))
    } catch (err) {
      console.error("[SessionsSidebar] Failed to load more sessions:", err)
      if (generation !== sessionPageGenerationRef.current) return
      setSessionPage((prev) => ({ ...prev, loadingMore: false }))
    }
  }, [
    listSessionsMutation,
    cwdFilter,
    sessionPage.hasMore,
    sessionPage.loadingMore,
    sessionPage.nextCursor,
    trimmedSearchQuery,
  ])

  const sessionsLoading = Result.isInitial(firstPageResult) && sessionPage.sessions.length === 0
  const sessions = sessionPage.sessions
  const apiKeyResult = useAtomValue(
    client.query("GetProviderAuth", { providerId: MAGNITUDE_PROVIDER_ID }, { reactivityKeys: ["apiKey"] }),
  )
  const accountState = Result.match(apiKeyResult, {
    onInitial: () => ({ label: "Account", subLabel: null as string | null }),
    onFailure: () => ({ label: "Account", subLabel: null as string | null }),
    onSuccess: (success) => {
      const value = success.value as { auth: Option.Option<ProviderAuth> }
      if (value.auth._tag !== "Some" || value.auth.value.type !== "api" || value.auth.value.key.length === 0) {
        return { label: "Account", subLabel: null as string | null }
      }
      const key = value.auth.value.key
      const masked = key.length > 8 ? `${key.slice(0, 4)}...${key.slice(-4)}` : "configured"
      return { label: "Signed in", subLabel: masked }
    },
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
            Effect.sync(() => window.removeEventListener("__magnitude:focus-search", handler)),
          )
        }),
      ),
    [],
  )
  useAtomMount(focusSearchAtom)

  const cwdOptionsResult = useAtomValue(
    client.query("ListSessionCwds", {}, { reactivityKeys: ["sessions"] }),
  )
  const cwdOptions = Result.match(cwdOptionsResult, {
    onInitial: () => [] as string[],
    onFailure: () => [] as string[],
    onSuccess: (result) => (result.value as SessionCwdSummary[]).map((summary) => summary.cwd),
  })

  const handleNewSession = () => {
    startNewSession({ cwd: cwdFilter })
    if (props?.overlay && props.onCloseOverlay) props.onCloseOverlay()
  }

  return (
    <SessionsSidebar
      loading={sessionsLoading}
      sessions={sessions.map((s) => {
        const liveStatus = activeSessionStatuses[s.sessionId]
        const statusFields = liveStatus
          ? {
              updatedAt: liveStatus.lastMessageAt,
              workStatus: liveStatus.workStatus,
              activeWorkerCount: liveStatus.activeWorkerCount,
            }
          : {
              updatedAt: s.updatedAt,
              workStatus: "idle" as const,
              activeWorkerCount: 0,
            }
        return {
          sessionId: s.sessionId,
          title: s.title,
          cwd: s.cwd,
          messageCount: s.messageCount,
          ...statusFields,
        }
      })}
      cwdFilter={cwdFilter}
      cwdOptions={cwdOptions}
      loadingMore={sessionPage.loadingMore}
      hasMore={sessionPage.hasMore}
      accountLabel={accountState.label}
      accountSubLabel={accountState.subLabel}
      onCwdFilterChange={setCwdFilter}
      onLoadMore={loadMoreSessions}
      onSelectSession={(id) => {
        resumeSession(id)
      }}
      onNewSession={handleNewSession}
      onOpenSettings={() => setSettingsOpen(true)}
      onOpenUsage={() => setUsageOpen(true)}
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
        filePath.split(".").pop() ?? "",
      )
    : false

  // Only query when we have a real path + cwd.
  // When no file is selected, use a static idle atom so the hook count stays stable.
  // P1: reactivityKeys: ["files"] so the query refreshes when files change.
  const readFileAtom = useMemo(
    () => filePath && selectedCwd
      ? client.query("ReadFile", {
          cwd: selectedCwd,
          path: filePath,
          format: isImageFile ? "base64" : "text",
        }, { reactivityKeys: ["files"] })
      : Atom.make(() => null),
    [client, selectedCwd, filePath, isImageFile],
  )
  const result = useAtomValue(readFileAtom)

  // P2: Handle loading and error states explicitly
  // result is null when no file is selected (idle atom), so guard for that.
  const loading = !!filePath && !!selectedCwd && result !== null && Result.isInitial(result)
  const errorMsg = filePath && selectedCwd && result !== null && Result.isFailure(result)
    ? "Failed to read file. The file may not exist or is not accessible."
    : null
  const content = filePath && result !== null && Result.isSuccess(result)
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

function WorkerDetailPageContainer({ slotProfiles }: { slotProfiles: SlotProfiles | null }): ReactNode {
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
      actions={actor ? (
        <ContextUsageIndicator
          context={actor.context}
          tokenCap={profile?.contextWindow ?? null}
          size={20}
          strokeWidth={2}
          showTokenLabel
          tooltip="native"
        />
      ) : null}
    >
      <WorkerDetailPanelContainer slotProfiles={slotProfiles} />
    </ChatColumnPage>
  )
}

function deriveWorkerInfo(
  forkId: string,
  actors: Record<string, DisplayActor>,
): { forkId: string; role: string; name: string } | null {
  const actor = actors[forkId]
  if (!actor || actor.kind !== "worker") return null
  return { forkId, role: actor.role, name: actor.name }
}

/** Unified settings+usage panel container.
 *  Receives `slotProfiles` from parent to avoid a duplicate subscription. */
function SettingsPanelContainer({
  slotProfiles,
  initialTab,
}: {
  slotProfiles: SlotProfiles | null
  initialTab: "settings" | "usage"
}): ReactNode {
  const client = useAgentClient()
  const setApiKeyVerified = useAtomSet(apiKeyVerifiedAtom)
  const [period, setPeriod] = useState<UsagePeriod>("24h")
  const modelConfig = useModelConfig()

  // ── API key state ──
  const apiKeyResult = useAtomValue(
    client.query("GetProviderAuth", { providerId: MAGNITUDE_PROVIDER_ID }, { reactivityKeys: ["apiKey"] }),
  )
  const apiKeyState: ApiKeyState = Result.match(apiKeyResult, {
    onInitial: () => ({ status: "none" } as ApiKeyState),
    onFailure: () => ({ status: "none" } as ApiKeyState),
    onSuccess: (s) => {
      const value = s.value as { auth: Option.Option<ProviderAuth> }
      if (value.auth._tag === "Some" && value.auth.value.type === "api") {
        const key = value.auth.value.key
        const masked = key.length > 8 ? `${key.slice(0, 4)}...${key.slice(-4)}` : "****"
        return { status: "config" as const, maskedKey: masked }
      }
      return { status: "none" as const }
    },
  })

  const updateProviderAuth = useAtomSet(
    client.mutation("UpdateProviderAuth"),
    { mode: "promise" },
  )
  const handleSave = async (key: string) => {
    await updateProviderAuth({ payload: { providerId: MAGNITUDE_PROVIDER_ID, auth: { type: "api", key } }, reactivityKeys: ["apiKey"] })
  }
  const handleDisconnect = async () => {
    await updateProviderAuth({ payload: { providerId: MAGNITUDE_PROVIDER_ID, auth: { type: "api", key: "" } }, reactivityKeys: ["apiKey"] })
    setApiKeyVerified(false)
  }

  // ── Slots ──
  const slots = useMemo(() => {
    return ([
      PRIMARY_SLOT_ID,
      // "secondary", // Secondary model settings are temporarily hidden.
    ] as const).map((slotId) => ({
      slotId,
      label: SLOT_DISPLAY_NAMES.primary,
      description: SLOT_DESCRIPTIONS.primary,
      modelDisplayName: slotProfiles?.primary?.modelDisplayName ?? "—",
      contextWindow: slotProfiles?.primary?.contextWindow ?? null,
    }))
  }, [slotProfiles])

  // ── Cloud subscription and usage limits ──
  const tz = useMemo(() => {
    try { return Intl.DateTimeFormat().resolvedOptions().timeZone } catch { return "UTC" }
  }, [])
  const cloudUsageAtom = useMemo(
    () => client.query("GetCloudUsage", { period, days: 30, tz }),
    [client, period, tz],
  )
  const cloudUsageResult = useAtomValue(cloudUsageAtom)
  const usageLoading = Result.isInitial(cloudUsageResult)
  const usageError = Result.isFailure(cloudUsageResult)
    ? (Cause.pretty(cloudUsageResult.cause) || "Failed to load usage data.")
    : null
  const usageData = Result.isSuccess(cloudUsageResult)
    ? (cloudUsageResult.value as CloudUsageResponse)
    : null

  return (
    <SettingsPanel
      apiKey={apiKeyState}
      onSaveApiKey={handleSave}
      onDisconnectApiKey={handleDisconnect}
      slots={slots}
      modelConfig={modelConfig}
      usageLoading={usageLoading}
      usageError={usageError}
      usageData={usageData}
      usagePeriod={period}
      onUsagePeriodChange={setPeriod}
      initialTab={initialTab}
    />
  )
}

/** Work status container — timer + active task table above composer */
function WorkStatusBarContainer({ slotProfiles }: { slotProfiles: SlotProfiles | null }): ReactNode {
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

function ComposerContainer({ docked = false }: { docked?: boolean }): ReactNode {
  const platform = usePlatform()
  const setBashMode = useAtomSet(bashModeAtom)
  const setSettingsOpen = useAtomSet(settingsOpenAtom)
  const setUsageOpen = useAtomSet(usageOpenAtom)
  const setFilePath = useAtomSet(selectedFilePathAtom)
  const sidebarVisible = useAtomValue(sidebarVisibleAtom)
  const setSidebarVisible = useAtomSet(sidebarVisibleAtom)
  const { startNewSession } = useSessionActions()
  const sendRef = useRef<(text: string) => void>(() => {})

  const commandContext: CommandContext = useMemo(() => ({
    resetConversation: () => startNewSession(),
    showSystemMessage: (message: string) => showToast("info", message),
    exitApp: () => { if (platform.quit) platform.quit() },
    openRecentChats: () => {
      if (sidebarVisible === false) {
        setSidebarVisible(true)
      }
      window.dispatchEvent(new CustomEvent("__magnitude:focus-search"))
    },
    enterBashMode: () => setBashMode(true),
    activateSkill: (skillName: string, _skillPath: string | undefined, args: string) => {
      const content = args.trim() ? `/${skillName} ${args.trim()}` : `/${skillName}`
      sendRef.current(content)
    },
    initProject: () => {
      showToast("info", "Project initialization is not available in the web app yet.")
    },
    openSettings: () => setSettingsOpen(true),
    openUsage: () => setUsageOpen(true),
    toggleAutopilot: () => {
      showToast("info", "Autopilot mode is not yet available in the web app.")
    },
  }), [
    startNewSession,
    platform,
    sidebarVisible,
    setSidebarVisible,
    setBashMode,
    setSettingsOpen,
    setUsageOpen,
  ])

  const composer = useComposerState(commandContext)
  sendRef.current = (text: string) => composer.handleSend(text)

  const handleMentionConfirm = useCallback((item: { path: string }) => {
    setFilePath(item.path)
  }, [setFilePath])

  return (
    <Composer
      role={composer.roleLabel}
      isStreaming={composer.isStreaming}
      bashMode={composer.bashMode}
      onSend={(text, mentions) => { void composer.handleSend(text, { mentions }) }}
      onInterrupt={composer.handleInterrupt}
      onRunBash={composer.handleRunBash}
      onSlashCommand={composer.handleSlashCommand}
      onToggleBashMode={() => composer.setBashMode((prev: boolean) => !prev)}
      onMentionConfirm={handleMentionConfirm}
      mentionClient={composer.mentionClient}
      cwd={composer.cwd}
      docked={docked}
    />
  )
}

/** FooterBar container */
function FooterBarContainer({ slotProfiles }: { slotProfiles: SlotProfiles | null }): ReactNode {
  const client = useAgentClient()
  const selectedSessionId = useSelectedSessionId()
  const selectedCwd = useAtomValue(selectedCwdAtom)
  const context = useDisplayState((state) => state.actors["root"]?.context ?? null)
  const { profile } = useRootSlotProfile(slotProfiles)
  const tokenCap = profile?.contextWindow ?? null
  const bashMode = useAtomValue(bashModeAtom)
  const nextEscWillKillAll = useAtomValue(nextEscWillKillAllAtom)
  const { displayMode } = useDisplayViewController()
  const setSettingsOpen = useAtomSet(settingsOpenAtom)
  const selectedSessionAtom = useMemo(
    () => selectedSessionId
      ? client.query("GetSession", { sessionId: selectedSessionId }, { reactivityKeys: ["sessions"] })
      : Atom.make(() => null),
    [client, selectedSessionId],
  )
  const selectedSessionResult = useAtomValue(selectedSessionAtom)
  const sessionCwd = selectedSessionResult !== null && Result.isSuccess(selectedSessionResult)
    ? (selectedSessionResult.value as SessionMetadata).cwd
    : null
  const cwd = sessionCwd ?? selectedCwd

  const thinkingLevel = profile?.reasoningEffort
    ? profile.reasoningEffort.charAt(0).toUpperCase() + profile.reasoningEffort.slice(1)
    : null
  const openSettings = useCallback(() => setSettingsOpen(true), [setSettingsOpen])

  return (
    <FooterBar
      context={context}
      tokenCap={tokenCap}
      cwd={cwd}
      model={profile?.modelDisplayName ?? null}
      thinkingLevel={thinkingLevel}
      onModelClick={openSettings}
      onThinkingClick={openSettings}
      bashMode={bashMode}
      nextEscWillKillAll={nextEscWillKillAll}
      transcriptMode={displayMode === "transcript"}
    />
  )
}

function BottomDockContainer({ slotProfiles }: { slotProfiles: SlotProfiles | null }): ReactNode {
  return (
    <div
      style={{
        margin: "14px 12px 14px",
        display: "flex",
        flexDirection: "column",
        gap: 8,
        flexShrink: 0,
      }}
    >
      <WorkStatusBarContainer slotProfiles={slotProfiles} />
      <ComposerContainer docked />
      <FooterBarContainer slotProfiles={slotProfiles} />
    </div>
  )
}

function ChatTitleBar(): ReactNode {
  const client = useAgentClient()
  const selectedSessionId = useSelectedSessionId()
  const displaySession = useDisplayState((state) => state.session)
  const selectedSessionAtom = useMemo(
    () => selectedSessionId
      ? client.query("GetSession", { sessionId: selectedSessionId }, { reactivityKeys: ["sessions"] })
      : Atom.make(() => null),
    [client, selectedSessionId],
  )
  const selectedSessionResult = useAtomValue(selectedSessionAtom)
  const metadataTitle = selectedSessionResult !== null && Result.isSuccess(selectedSessionResult)
    ? (selectedSessionResult.value as SessionMetadata).title
    : null
  const streamedTitle = displaySession.sessionId === selectedSessionId
    ? displaySession.title
    : null
  const title = selectedSessionId
    ? (streamedTitle ?? metadataTitle)?.trim() || "Untitled session"
    : "New session"

  return (
    <div className="chat-title-bar" title={title}>
      <span className="chat-title-bar-title">{title}</span>
    </div>
  )
}

/** Listen for __magnitude:interrupt-all custom event → Interrupt RPC with target: all */
function useInterruptAllListener(): void {
  const client = useAgentClient()
  const selectedSessionId = useSelectedSessionId()
  const interruptMutation = useAtomSet(client.mutation("Interrupt"))

  const interruptAtom = useMemo(
    () =>
      Atom.make(
        Effect.gen(function* () {
          const handler = () => {
            if (!selectedSessionId) return
            interruptMutation({
              payload: {
                sessionId: selectedSessionId,
                target: { _tag: "all" },
              },
            })
          }
          window.addEventListener("__magnitude:interrupt-all", handler)
          yield* Effect.addFinalizer(() =>
            Effect.sync(() => window.removeEventListener("__magnitude:interrupt-all", handler)),
          )
        }),
      ),
    [selectedSessionId, interruptMutation],
  )

  useAtomMount(interruptAtom)
}

/** Inner app — has display view + AgentClient context */
function AppInner(): ReactNode {
  const apiKeyVerified = useAtomValue(apiKeyVerifiedAtom)
  const platform = usePlatform()
  const sidebarVisible = useAtomValue(sidebarVisibleAtom)
  const setSidebarVisible = useAtomSet(sidebarVisibleAtom)

  // Detect responsive mode (≤640px) — no useEffect, uses matchMedia store
  const isNarrow = useSyncExternalStore(subscribeResponsive, getIsNarrow)

  // Sync sidebarVisibleAtom with responsive state:
  // - Not narrow → null (sidebar always docked, not overlay)
  // - Narrow → false (overlay hidden by default)
  // This runs during render — when isNarrow changes, the atom is updated.
  // The atom write is idempotent (only fires if value actually changes).
  const expectedSidebar = isNarrow ? false : null
  if (sidebarVisible !== expectedSidebar) {
    // Only reset to false when entering narrow mode; don't clobber user's open state
    if (!isNarrow && sidebarVisible !== null) {
      setSidebarVisible(null)
    } else if (isNarrow && sidebarVisible === null) {
      setSidebarVisible(false)
    }
  }

  useMenuActions()
  useInterruptAllListener()

  // Check if API key is already set — gate the login screen from the parent
  // to avoid setState-during-render in LoginScreen.
  // The apiKeyVerified atom is only set by LoginScreen in an event handler
  // (after the user submits a key). Here we gate purely on the query result.
  const client = useAgentClient()
  const apiKeyResult = useAtomValue(
    client.query("GetProviderAuth", { providerId: MAGNITUDE_PROVIDER_ID }, { reactivityKeys: ["apiKey"] }),
  )

  // Distinguish loading from failure:
  // - Initial (loading) → spinner
  // - Failure → show login screen (user can enter a key)
  // - Success → check if key is set
  const apiKeyLoading = Result.isInitial(apiKeyResult)
  const keyAlreadySet = Result.match(apiKeyResult, {
    onInitial: () => false,
    onFailure: () => false,
    onSuccess: (s) => {
      const value = s.value as { auth: Option.Option<ProviderAuth> }
      return value.auth._tag === "Some" && value.auth.value.type === "api" && value.auth.value.key.length > 0
    },
  })

  // Key exists → show app. Loading → spinner. Missing/failed → login screen.
  if (apiKeyLoading && !apiKeyVerified) {
    return (
      <div style={{ position: "fixed", inset: 0, background: "var(--bg-base)", display: "flex", alignItems: "center", justifyContent: "center", zIndex: 200 }}>
        <Loader2 size={24} style={{ color: "var(--accent-primary)", animation: "spin 1s linear infinite" }} />
      </div>
    )
  }

  if (Result.isFailure(apiKeyResult) && !apiKeyVerified) {
    return (
      <DaemonConnectionError
        message="Failed to connect to the Magnitude daemon."
        reconnecting={false}
        invariantViolation={false}
        onRetry={() => window.location.reload()}
        onQuit={() => {
          if (platform.quit) {
            platform.quit()
          }
        }}
      />
    )
  }

  if (!keyAlreadySet && !apiKeyVerified) {
    return <LoginScreen />
  }

  return <AuthenticatedAppContent isNarrow={isNarrow} />
}

function AuthenticatedAppContent({ isNarrow }: { isNarrow: boolean }): ReactNode {
  useSessionPreload()
  useActiveSessionStatusesSubscription()

  const connectionError = useDisplayConnectionError()
  const platform = usePlatform()
  const isDesktop = platform.id === "desktop"
  const sidebarVisible = useAtomValue(sidebarVisibleAtom)
  const setSidebarVisible = useAtomSet(sidebarVisibleAtom)
  const { profiles: slotProfiles } = useSlotProfiles()
  const showOverlaySidebar = isNarrow && sidebarVisible === true
  const settingsOpen = useAtomValue(settingsOpenAtom)
  const usageOpen = useAtomValue(usageOpenAtom)
  const setSettingsOpen = useAtomSet(settingsOpenAtom)
  const setUsageOpen = useAtomSet(usageOpenAtom)
  const controller = useDisplayViewController()
  const forkStack = controller.expandedForkStack

  const panelOpen = settingsOpen || usageOpen
  const panelTab: "settings" | "usage" = usageOpen && !settingsOpen ? "usage" : "settings"
  const workerDetailOpen = !panelOpen && forkStack.length > 0

  return (
    <div
      className="app"
      style={{
        display: "flex",
        height: "100vh",
        overflow: "hidden",
        background: isDesktop ? "transparent" : "var(--bg-base)",
      }}
    >
      {/* Docked sidebar — hidden by CSS when narrow */}
      {!isNarrow && <SessionsSidebarContainer />}
      {/* Overlay sidebar — shown when narrow + visible */}
      {showOverlaySidebar && (
        <SessionsSidebarContainer overlay onCloseOverlay={() => setSidebarVisible(false)} />
      )}
      <div className="chat-column" style={{ flex: 1, minWidth: 0, display: "flex", flexDirection: "column", position: "relative", background: "var(--bg-base)" }}>
        {/* Main chat column — always mounted, always in the layout. When a
            panel or worker detail is open, it's covered by an absolute
            overlay. Keeping it in the layout (not display:none) preserves
            scroll metrics so the scroll controller can capture and restore
            the correct position across overlay navigation. */}
        <div
          style={{
            flexDirection: "column",
            flex: 1,
            minHeight: 0,
          }}
        >
          <ChatTitleBar />
          <ChatTimeline isVisible={!panelOpen && !workerDetailOpen} />
          <BottomDockContainer slotProfiles={slotProfiles} />
        </div>
        {(panelOpen || workerDetailOpen) && (
          <div
            style={{
              position: "absolute",
              inset: 0,
              display: "flex",
              flexDirection: "column",
              background: "var(--bg-base)",
              zIndex: 1,
            }}
          >
            {panelOpen && (
              <ChatColumnPage
                title={panelTab === "usage" ? "Usage" : "Settings"}
                backLabel="Back to session"
                onBack={() => { setSettingsOpen(false); setUsageOpen(false) }}
              >
                <SettingsPanelContainer slotProfiles={slotProfiles} initialTab={panelTab} />
              </ChatColumnPage>
            )}
            {workerDetailOpen && (
              <WorkerDetailPageContainer slotProfiles={slotProfiles} />
            )}
          </div>
        )}
        <ToastContainer />
      </div>
      <FileViewerPanelContainer />
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
