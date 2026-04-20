import { useState, useEffect, useRef, useCallback, useMemo } from 'react'
import { useKeyboard, useRenderer } from '@opentui/react'
import { Effect, Layer, Cause } from 'effect'

import { createCodingAgentClient, ChatPersistence, getSessionTitleFromTaskGraph, type DisplayState, type AgentStatusState, type DebugSnapshot, type AppEvent, type UnexpectedErrorMessage, PROVIDERS, getProvider, type ProviderDefinition, type AuthMethodDef, type ModelSelection, type ProviderAuthMethodStatus, type ForkMemoryState, type CompactionState, type ToolStateProjectionState } from '@magnitudedev/agent'
import { loadSkills } from '@magnitudedev/skills'
import { textParts } from '@magnitudedev/agent'
import { JsonChatPersistence, loadSessionSummary } from './persistence'

import { MessageView } from './components/message-view'
import { ErrorBoundary } from './components/error-boundary'
import { StickyWorkingHeader } from './components/think-block'
import { PendingCommunicationsPanel } from './components/pending-communications-panel'
import { LoadPreviousButton } from './components/chat-controls'


import { usePaginatedTimeline } from './hooks/use-paginated-timeline'
import { useCollapsedBlocks } from './hooks/use-collapsed-blocks'

import { useTheme } from './hooks/use-theme'
import { SelectedFileProvider } from './hooks/use-file-viewer'

import { BOX_CHARS } from './utils/ui-constants'
import { hasConversationActivity } from './utils/start-state'
import {
  localProviderAddedModelToast,
  localProviderSavedApiKeyToast,
  localProviderSavedEmptyApiKeyToast,
  localProviderSavedEndpointToast,
} from './utils/local-provider-toast-messages'
import { AnimatedLogo } from './components/animated-logo'
import { RecentChatsWidget } from './components/recent-chats-widget'
import { SessionLoadingView } from './components/session-loading-view'
import { routeSlashCommand, type CommandContext } from './commands/command-router'
import { INIT_PROMPT } from './commands/init-prompt'
import { registerSkillCommands, type SlashCommandDefinition } from './commands/slash-commands'
import { useSelectionAutoCopy } from './utils/clipboard'
import { useRecentChatsNavigation } from './hooks/use-recent-chats-navigation'
import { useModelSelectNavigation } from './hooks/use-model-select-navigation'
import { useProviderSelectNavigation } from './hooks/use-provider-select-navigation'
import type { SettingsTab } from './hooks/use-settings-navigation'

import { useAuthFlow } from './hooks/use-auth-flow'
import { type WizardStep } from './components/setup-wizard-overlay'
import { AppOverlays } from './components/app-overlays'


import { buildModelPickerItems, filterModelPickerItems, resolveSlotDefaultSelection } from './utils/model-picker'
import { resolveLocalWizardSlotDefaults } from './utils/wizard-flow'
import { getRecentChats, type RecentChat } from './data/recent-chats'
import { logger, initLogger, subscribeToLogs, clearSessionLog, getSessionLogPath, type LogEntry } from '@magnitudedev/logger'


import path from 'path'
import { executeBashCommand, type BashResult } from './utils/bash-executor'


import { BashOutput } from './components/bash-output'


import { FileViewerPanel } from './components/file-viewer-panel'
import type { Attachment } from '@magnitudedev/agent'
import { DebugPanel } from './components/debug-panel'
import { ChatController } from './components/chat/chat-controller'
import { useTasks } from './hooks/use-tasks'
import { useLocalWidth } from './hooks/use-local-width'

import { initTelemetry, shutdownTelemetry, trackSessionStart, trackSessionEnd, SessionTracker } from '@magnitudedev/telemetry'

import { setSessionTracker } from './utils/telemetry-state'
import { TextAttributes, type KeyEvent } from '@opentui/core'

import { createId } from '@magnitudedev/generate-id'

import { useProviderRuntime } from './providers/provider-runtime'
import { useStorage } from './providers/storage-provider'
import { useProviderUiState } from './hooks/use-provider-ui-state'
import { useFilePanel } from './hooks/use-file-panel'
import { useLazyClient } from './hooks/use-lazy-client'
import { MAGNITUDE_SLOTS, type MagnitudeSlot } from '@magnitudedev/agent'

import type { Preset, ProviderOptions } from '@magnitudedev/storage'

const SYSTEM_DEFAULTS_PRESET = '__system_defaults__'

export const getSelectedForkContentVersion = (
  selectedForkId: string | null,
  forkDisplay: Pick<DisplayState, 'messages' | 'pendingInboundCommunications'> | null
): string => {
  if (!selectedForkId) return 'main'
  return [
    selectedForkId,
    forkDisplay?.messages.length ?? 0,
    forkDisplay?.pendingInboundCommunications.length ?? 0,
  ].join(':')
}

type AgentClient = Awaited<ReturnType<typeof createCodingAgentClient>>

export type SessionStart =
  | { _tag: 'new' }
  | { _tag: 'latest' }
  | { _tag: 'resume'; sessionId: string }

export function App({ sessionStart, debug, onClientReady, onSessionId }: { sessionStart: SessionStart; debug: boolean; onClientReady?: (client: AgentClient | null) => void; onSessionId?: (id: string) => void }) {
  const [conversationKey, setConversationKey] = useState(0)
  const [sessionSelection, setSessionSelection] = useState<string | null | undefined>(
    sessionStart._tag === 'new' ? null : sessionStart._tag === 'latest' ? undefined : sessionStart.sessionId
  )
  const hasAnimatedRef = useRef(false)

  const handleReset = useCallback(() => {
    hasAnimatedRef.current = true
    setSessionSelection(null)
    setConversationKey(prev => prev + 1)
  }, [])

  const handleResumeSession = useCallback((sessionId: string) => {
    hasAnimatedRef.current = true
    setSessionSelection(sessionId)
    setConversationKey(prev => prev + 1)
  }, [])

  return (
    <AppInner
      debugMode={debug}
      key={conversationKey}
      skipAnimation={hasAnimatedRef.current}
      sessionSelection={sessionSelection}
      onReset={handleReset}
      onResumeSession={handleResumeSession}
      onClientReady={onClientReady}
      onSessionId={onSessionId}
    />
  )
}

function AppInner({
  debugMode,
  skipAnimation,
  sessionSelection,
  onReset,
  onResumeSession,
  onClientReady,
  onSessionId,
}: {
  debugMode: boolean
  skipAnimation: boolean
  sessionSelection: string | null | undefined
  onReset: () => void
  onResumeSession: (sessionId: string) => void
  onClientReady?: (client: AgentClient | null) => void
  onSessionId?: (id: string) => void
}) {
  const renderer = useRenderer()
  const providerRuntime = useProviderRuntime()
  const storage = useStorage()
  const { state: providerUiState, reload: reloadProviderState } = useProviderUiState()
  const { client, workspacePath, send: clientSend, ensureReady: ensureClientReady, setFactory: setClientFactory, setClient: setLazyClient } = useLazyClient()
  const [display, setDisplay] = useState<DisplayState | null>(null)
  const [toolState, setToolState] = useState<ToolStateProjectionState | null>(null)
  const [agentStatusState, setAgentStatusState] = useState<AgentStatusState | null>(null)
  const [expandedForkStack, setExpandedForkStack] = useState<string[]>([])
  const expandedForkId = expandedForkStack.length > 0 ? expandedForkStack[expandedForkStack.length - 1] : null
  const pushForkOverlay = (forkId: string) => setExpandedForkStack(s => [...s, forkId])
  const popForkOverlay = () => {
    setExpandedForkStack(s => s.slice(0, -1))
  }

  const [forkDisplay, setForkDisplay] = useState<DisplayState | null>(null)
  const [forkTokenEstimate, setForkTokenEstimate] = useState(0)
  const [forkLastActualInputTokens, setForkLastActualInputTokens] = useState<number | null>(null)
  const [forkHasCompletedTurn, setForkHasCompletedTurn] = useState(false)
  const [forkIsCompacting, setForkIsCompacting] = useState(false)

  const [systemMessages, setSystemMessages] = useState<Array<{ id: string; text: string; timestamp: number }>>([])
  const systemMessageTimeoutsRef = useRef<Map<string, ReturnType<typeof setTimeout>>>(new Map())
  const [showRecentChatsOverlay, setShowRecentChatsOverlay] = useState(false)
  const [settingsTab, setSettingsTab] = useState<SettingsTab | null>(null)
  const [selectingModelFor, setSelectingModelFor] = useState<MagnitudeSlot | null>(null)
  const emptySlotModels = (): Record<MagnitudeSlot, ModelSelection | null> => (
    Object.fromEntries(MAGNITUDE_SLOTS.map((slot) => [slot, null])) as Record<MagnitudeSlot, ModelSelection | null>
  )
  const [slotModels, setSlotModels] = useState<Record<MagnitudeSlot, ModelSelection | null>>(emptySlotModels)
  const [presets, setPresets] = useState<Preset[]>([])

  const [preferencesSelectedIndex, setPreferencesSelectedIndex] = useState(0)
  const [showAllProviders, setShowAllProviders] = useState(false)
  const [showRecommendedOnly, setShowRecommendedOnly] = useState(false)
  const [modelSearch, setModelSearch] = useState('')
  const [providerDetailId, setProviderDetailId] = useState<string | null>(null)
  const [providerDetailSelectedIndex, setProviderDetailSelectedIndex] = useState(0)
  const [providerRefreshKey, setProviderRefreshKey] = useState(0)
  const [providerDetailStatus, setProviderDetailStatus] = useState<ProviderAuthMethodStatus | null>(null)
  const [providerDetailOptions, setProviderDetailOptions] = useState<ProviderOptions | undefined>(undefined)
  const [contextHardCap, setContextHardCap] = useState<number | null>(null)
  const [agentProjectionMode, setAgentProjectionMode] = useState<string>('default')
  const [bashMode, setBashMode] = useState(false)
  const [bashOutputs, setBashOutputs] = useState<BashResult[]>([])
  const [debugPanelVisible, setDebugPanelVisible] = useState(false)
  const [debugSnapshot, setDebugSnapshot] = useState<DebugSnapshot | null>(null)
  const [debugEvents, setDebugEvents] = useState<AppEvent[]>([])
  const [debugLogs, setDebugLogs] = useState<LogEntry[]>([])
  const [composerHasContent, setComposerHasContent] = useState(false)
  const [restoredQueuedInputText, setRestoredQueuedInputText] = useState<string | null>(null)
  const [tokenEstimate, setTokenEstimate] = useState(0)
  const [lastActualInputTokens, setLastActualInputTokens] = useState<number | null>(null)
  const [hasCompletedTurn, setHasCompletedTurn] = useState(false)
  const [isCompacting, setIsCompacting] = useState(false)
  const returnToProviderDetailRef = useRef<string | null>(null)
  const turnStartTimeRef = useRef<number | null>(null)
  const hasAnimatedRef = useRef(skipAnimation)


  const resetModelPickerState = useCallback(() => {
    setModelSearch('')
    setShowAllProviders(false)
    setShowRecommendedOnly(false)
    setSelectingModelFor(null)
  }, [])

  const formatFooterTokens = (n: number) => {
    if (n >= 1000) {
      const v = (n / 1000).toFixed(1)
      return (v.endsWith('.0') ? v.slice(0, -2) : v) + 'k'
    }
    return `${n}`
  }
  const tokenUsage = lastActualInputTokens ?? (hasCompletedTurn ? tokenEstimate : null)
  const contextPercent = (tokenUsage != null && contextHardCap) ? Math.round((tokenUsage / contextHardCap) * 100) : null
  const contextDisplayText = tokenUsage == null
    ? '-'
    : (contextHardCap
      ? `${contextPercent}% ${formatFooterTokens(tokenUsage)}/${formatFooterTokens(contextHardCap)}`
      : `${formatFooterTokens(tokenUsage)}/Unknown`)
  const contextRenderedText = isCompacting ? `>>> ${contextDisplayText} <<<` : contextDisplayText

  // Always reserve width for the longest possible escape hint so that
  // attachments don't reflow when hints appear/disappear.
  const maxEscHintWidth = 'Press Esc again to interrupt all workers'.length

  const chatColumn = useLocalWidth()
  const chatColumnWidth = chatColumn.width ?? 80
  const footerRightGap = contextRenderedText ? 1 : 0
  const footerHorizontalPadding = 4
  const footerSafetyBuffer = 4
  const attachmentsMaxWidth = Math.max(
    0,
    chatColumnWidth
      - footerHorizontalPadding
      - maxEscHintWidth
      - contextRenderedText.length
      - footerRightGap
      - footerSafetyBuffer,
  )



  // Browser setup overlay state
  const [showBrowserSetup, setShowBrowserSetup] = useState(false)

  // Setup wizard state
  const [showSetupWizard, setShowSetupWizard] = useState(false)
  const [wizardStep, setWizardStep] = useState<WizardStep>('provider')
  const [wizardProviderSelectedIndex, setWizardProviderSelectedIndex] = useState(0)
  const [wizardModelSelectedIndex, setWizardModelSelectedIndex] = useState(0)
  const [recentChatsSelectedIndex, setRecentChatsSelectedIndex] = useState(0)
  const [authMethodSelectedIndex, setAuthMethodSelectedIndex] = useState(0)
  const [wizardSlotModels, setWizardSlotModels] = useState<Record<MagnitudeSlot, ModelSelection | null>>(emptySlotModels)
  const [wizardConnectedProvider, setWizardConnectedProvider] = useState<string | null>(null)
  const [wizardNeedsChromium, setWizardNeedsChromium] = useState<boolean | null>(null)

  const [wizardSelectedProviderId, setWizardSelectedProviderId] = useState<string | null>(null)
  const [wizardSelectedProviderDiscoveredModels, setWizardSelectedProviderDiscoveredModels] = useState<Array<{ id: string, name: string }>>([])
  const [wizardSelectedProviderRememberedModelIds, setWizardSelectedProviderRememberedModelIds] = useState<string[]>([])
  const localProviderSet = useMemo(() => new Set(['lmstudio', 'ollama', 'llama.cpp', 'openai-compatible-local']), [])
  const wizardHasProviderEndpointStep = !!wizardSelectedProviderId && localProviderSet.has(wizardSelectedProviderId)

  const wizardTotalSteps = useMemo(() => {
    const baseSteps = wizardHasProviderEndpointStep ? 3 : 2 // provider (+endpoint when local) + models
    if (wizardNeedsChromium === false) return baseSteps
    return baseSteps + 1
  }, [wizardNeedsChromium, wizardHasProviderEndpointStep])

  const [recentChats, setRecentChats] = useState<RecentChat[] | null>(null)

  const refreshRecentChats = useCallback(() => {
    getRecentChats(storage).then(setRecentChats)
  }, [storage])

  useEffect(() => {
    logger.info('App started')
    if (debugMode) logger.info('Debug mode enabled - press Ctrl+X to toggle debug panel')
    refreshRecentChats()
  }, [debugMode, refreshRecentChats])

  useEffect(() => {
    if (!providerUiState) return

    setSlotModels(providerUiState.slotModels)
    storage.config.getPresets().then((loaded: Preset[]) => {
      setPresets(loaded)
    }).catch((error: unknown) => {
      logger.warn({ error: error instanceof Error ? error.message : String(error) }, 'Failed to load presets')
    })

    initTelemetry({ telemetryEnabled: providerUiState.telemetryEnabled })

    if (!providerUiState.setupComplete) {
      setShowSetupWizard(true)
    }
  }, [providerUiState, storage])

  useEffect(() => {
    if (!providerUiState) return
    providerRuntime.state.contextLimits('lead').then((limits) => {
      setContextHardCap(limits.hardCap)
    }).catch((error) => {
      logger.warn({ error: error instanceof Error ? error.message : String(error) }, 'Failed to load provider context limits')
    })
  }, [providerRuntime, slotModels.lead, providerUiState])

  // Check Chromium installation when wizard opens
  useEffect(() => {
    if (!showSetupWizard) return
    import('@magnitudedev/browser-harness').then(({ isBrowserInstalled }) => {
      setWizardNeedsChromium(!isBrowserInstalled())
    }).catch(() => {
      setWizardNeedsChromium(true)
    })
  }, [showSetupWizard])

  // Subscribe to live logs for debug panel
  useEffect(() => {
    if (!debugMode) return
    return subscribeToLogs((entry) => {
      setDebugLogs(prev => [...prev, entry])
    })
  }, [debugMode])

  useEffect(() => {
    if (showRecentChatsOverlay) {
      refreshRecentChats()
    }
  }, [showRecentChatsOverlay, refreshRecentChats])

  useEffect(() => {
    let mounted = true
    let c: AgentClient | null = null

    // Register skills as slash commands (fire-and-forget, non-blocking)
    loadSkills(process.cwd()).then((skillsMap) => {
      const commands: SlashCommandDefinition[] = []

      for (const s of skillsMap.values()) {
        commands.push({
          id: s.name,
          label: s.name,
          description: s.description,
          source: 'skill' as const,
          skillPath: s.path,
        })
      }

      if (commands.length > 0) {
        registerSkillCommands(commands)
        logger.info({ count: commands.length, names: commands.map(c => c.id) }, 'Registered skill commands')
      }
    }).catch((err) => {
      logger.warn({ error: err.message }, 'Failed to load skills')
    })

    let resolvedWorkspacePath: string | null = null
    let resolvedSessionId: string | null = null

    const createClient = async () => {
      let sessionId: string | undefined
      if (sessionSelection === undefined) {
        sessionId = await storage.sessions.findLatest() ?? undefined
      } else if (sessionSelection === null) {
        sessionId = undefined
      } else {
        sessionId = sessionSelection
      }

      const persistence = new JsonChatPersistence({
        storage,
        workingDirectory: process.cwd(),
        sessionId,
      })
      const activeSessionId = persistence.getSessionId()
      resolvedSessionId = activeSessionId
      onSessionId?.(activeSessionId)
      resolvedWorkspacePath = storage.sessions.getWorkspacePath(activeSessionId) ?? null
      initLogger(persistence.getSessionId())
      clearSessionLog(persistence.getSessionId())
      logger.info({ logFile: getSessionLogPath(persistence.getSessionId()) }, 'Session logger initialized')
      const persistenceLayer = Layer.succeed(ChatPersistence, persistence)
      return createCodingAgentClient({
        persistence: persistenceLayer,
        storage,
        debug: debugMode,
        providerRuntime: providerRuntime.layer,
        sessionId: activeSessionId,
      })
    }

    const setupClient = (client: AgentClient) => {
      if (!mounted) {
        client.dispose()
        return
      }
      c = client
      setLazyClient(client, resolvedWorkspacePath)
      onClientReady?.(client)
      renderer.setTerminalTitle("Magnitude")

      // Telemetry tracking state
      const sessionTracker = new SessionTracker()
      setSessionTracker(sessionTracker)

      // Log all events to event log file + collect for debug panel
      client.onEvent((event) => {
        if (debugMode && mounted) {
          setDebugEvents(prev => [...prev, event])
        }

        // Telemetry event tracking
        if (event.type === 'session_initialized') {
          trackSessionStart({
            platform: event.context.platform,
            shell: event.context.shell,
            isResume: sessionSelection !== null && sessionSelection !== undefined,
          })
        }

        if (event.type === 'user_message') {
          sessionTracker.recordUserMessage()
        }

        if (event.type === 'turn_completed') {
          sessionTracker.recordTurn(event.providerId ?? null, event.modelId ?? null, event.inputTokens, event.outputTokens)
        }

        if (event.type === 'compaction_completed') {
          sessionTracker.recordCompaction()
        }
        if (event.type === 'compaction_failed') {
          sessionTracker.recordCompaction()
        }
      })

      // Framework errors bypass the event system entirely — render directly in the TUI
      client.onError((error) => {
        if (!mounted) return
        const errorMsg: UnexpectedErrorMessage = {
          id: createId(),
          type: 'unexpected_error',
          tag: error._tag,
          message: Cause.pretty(error.cause),
          timestamp: Date.now()
        }
        setDisplay(prev => prev ? { ...prev, messages: [...prev.messages, errorMsg] } : prev)
      })

      // Subscribe to agent state (global projection)
      client.state.agentStatus.subscribe((state) => {
        if (mounted) {
          setAgentStatusState(state)
        }
      })

      // Subscribe to restore queued messages signal (only for main/root)
      client.on.restoreQueuedMessages(({ forkId, messages }) => {
        // Only restore if this is for the main agent (not a fork)
        if (mounted && forkId === null && messages.length > 0) {
          const restored = messages.join('\n')
          logger.info({ restored, length: restored.length }, 'Restoring queued messages to input')
          setRestoredQueuedInputText(restored)
        }
      })


      client.state.taskGraph.subscribe((state) => {
        if (!mounted) return
        const title = getSessionTitleFromTaskGraph(state)
        if (!title) return
        logger.info({ title }, 'Session title derived from task graph')
        renderer.setTerminalTitle(title)
      })
    }

    if (sessionSelection === null) {
      // NEW SESSION: defer client creation, show empty UI immediately
      setDisplay({
        status: 'idle',
        messages: [],
        pendingInboundCommunications: [],
        currentTurnId: null,
        streamingMessageId: null,
        activeThinkBlockId: null,
        showButton: 'send',
      })
      setClientFactory(async () => {
        const client = await createClient()
        setupClient(client)
        return client
      })
    } else {
      // RESUMED SESSION: create client immediately (existing behavior)
      setClientFactory(null)
      createClient().catch((err) => {
        logger.error({ error: err.message, stack: err.stack }, 'Failed to create agent client');
        throw err;
      }).then(async (client) => {
        logger.info('Agent client created successfully');
        setupClient(client)

        if (!resolvedSessionId) return

        const summary = await loadSessionSummary(storage, resolvedSessionId)
        if (summary?.title) {
          renderer.setTerminalTitle(summary.title)
        }
      })
    }

    return () => {
      mounted = false
      setClientFactory(null)
      onClientReady?.(null)
      c?.dispose()
    }
  }, [debugMode, onClientReady, onSessionId, providerRuntime, renderer, sessionSelection, setClientFactory, setLazyClient, storage])

  // Subscribe to display state for selected fork
  useEffect(() => {
    if (!client) {
      logger.warn("handleInterrupt: no client, returning")
      return
    }

    const unsubscribe = client.state.display.subscribeFork(null, (state) => {
      if (state.status === 'streaming' && turnStartTimeRef.current === null) {
        turnStartTimeRef.current = Date.now()
      } else if (state.status === 'idle') {
        turnStartTimeRef.current = null
      }
      if (state.streamingMessageId) {
        lastStreamingMessageIdRef.current = state.streamingMessageId
      }
      if (state.status === 'streaming') {
        interruptedMessageIdRef.current = null
      } else if (state.status === 'idle' && state.messages.some(m => m.type === 'interrupted')) {
        interruptedMessageIdRef.current = lastStreamingMessageIdRef.current
      }
      setDisplay(state)
    })

    return unsubscribe
  }, [client])

  useEffect(() => {
    const onFocus = () => {
      client?.send({ type: 'window_focus_changed', forkId: null, focused: true })
    }
    const onBlur = () => {
      client?.send({ type: 'window_focus_changed', forkId: null, focused: false })
    }
    renderer.on('focus', onFocus)
    renderer.on('blur', onBlur)
    return () => {
      renderer.off('focus', onFocus)
      renderer.off('blur', onBlur)
    }
  }, [renderer, client])

  // Subscribe to compaction state for context usage bar
  useEffect(() => {
    if (!client) return

    const unsubscribe = client.state.compaction.subscribeFork(null, (state: CompactionState) => {
      setTokenEstimate(state.tokenEstimate)
      setLastActualInputTokens(state.lastActualInputTokens)
      setHasCompletedTurn(state.hasCompletedTurn)
      setIsCompacting(state._tag !== 'idle')
    })

    return unsubscribe
  }, [client])

  const tasks = useTasks({
    client,
  })

  const selectedForkId: string | null = null

  // Subscribe to selected fork's display
  useEffect(() => {
    if (!client || !selectedForkId) {
      setForkDisplay(null)
      return
    }
    const unsubscribe = client.state.display.subscribeFork(selectedForkId, (state) => {
      setForkDisplay(state)
    })
    return unsubscribe
  }, [client, selectedForkId])

  // Subscribe to selected fork's compaction state
  useEffect(() => {
    if (!client || !selectedForkId) {
      setForkTokenEstimate(0)
      setForkLastActualInputTokens(null)
      setForkHasCompletedTurn(false)
      setForkIsCompacting(false)
      return
    }
    const unsubscribe = client.state.compaction.subscribeFork(selectedForkId, (state: CompactionState) => {
      setForkTokenEstimate(state.tokenEstimate)
      setForkLastActualInputTokens(state.lastActualInputTokens)
      setForkHasCompletedTurn(state.hasCompletedTurn)
      setForkIsCompacting(state._tag !== 'idle')
    })
    return unsubscribe
  }, [client, selectedForkId])

  // Subscribe to debug stream when debug mode is enabled and panel is visible
  useEffect(() => {
    if (!client || !debugMode || !debugPanelVisible) return

    const unsubscribe = client.subscribeDebug(null, (snapshot) => {
      setDebugSnapshot(snapshot)
    })

    return unsubscribe
  }, [client, debugMode, debugPanelVisible])





  const activeDisplay = selectedForkId ? forkDisplay : display

  const activeModelSummary = useMemo(() => {
    const leadModel = slotModels.lead
    const rootModelSummary = leadModel ? {
      provider: getProvider(leadModel.providerId)?.name ?? leadModel.providerId,
      model: getProvider(leadModel.providerId)?.models.find(m => m.id === leadModel.modelId)?.name ?? leadModel.modelId,
    } : null
    if (!selectedForkId || !agentStatusState) return rootModelSummary
    const agentId = agentStatusState.agentByForkId.get(selectedForkId)
    const agent = agentId ? agentStatusState.agents.get(agentId) : undefined
    if (!agent) return rootModelSummary
    // agent.role is the slot name (lead or worker)
    const slot = (MAGNITUDE_SLOTS as readonly string[]).includes(agent.role)
      ? agent.role as MagnitudeSlot
      : 'lead' as MagnitudeSlot
    const selection = slotModels[slot]
    if (!selection) return null
    return {
      provider: getProvider(selection.providerId)?.name ?? selection.providerId,
      model: getProvider(selection.providerId)?.models.find(m => m.id === selection.modelId)?.name ?? selection.modelId,
    }
  }, [selectedForkId, agentStatusState, slotModels])

  const forkSlot = useMemo(() => {
    if (!expandedForkId || !agentStatusState) return null
    const agentId = agentStatusState.agentByForkId.get(expandedForkId)
    const agent = agentId ? agentStatusState.agents.get(agentId) : undefined
    if (!agent) return null
    return (MAGNITUDE_SLOTS as readonly string[]).includes(agent.role)
      ? agent.role as MagnitudeSlot
      : 'lead' as MagnitudeSlot
  }, [expandedForkId, agentStatusState])

  const forkModelSummary = useMemo(() => {
    if (!forkSlot) return null
    const selection = slotModels[forkSlot]
    if (!selection) return null
    return {
      provider: getProvider(selection.providerId)?.name ?? selection.providerId,
      model: getProvider(selection.providerId)?.models.find(m => m.id === selection.modelId)?.name ?? selection.modelId,
    }
  }, [forkSlot, slotModels])

  const [forkContextHardCap, setForkContextHardCap] = useState<number | null>(null)

  useEffect(() => {
    let cancelled = false

    if (!forkSlot) {
      setForkContextHardCap(contextHardCap)
      return () => {
        cancelled = true
      }
    }

    providerRuntime.state.contextLimits(forkSlot).then((limits) => {
      if (!cancelled) {
        setForkContextHardCap(limits.hardCap)
      }
    }).catch(() => {
      if (!cancelled) {
        setForkContextHardCap(contextHardCap)
      }
    })

    return () => {
      cancelled = true
    }
  }, [providerRuntime, forkSlot, contextHardCap])

  const mainTimelineMessages = useMemo(
    () => (activeDisplay?.messages ?? []).filter(m => {
      if (m.type === 'fork_activity') return false
      if (selectedForkId === null && m.type === 'agent_communication') return false
      return true
    }),
    [activeDisplay?.messages, selectedForkId]
  )

  const { visibleItems, hiddenCount, loadMore, hasMore } = usePaginatedTimeline(
    mainTimelineMessages,
    bashOutputs,
    systemMessages
  )

  const { isCollapsed, toggleCollapse, collapseBlock } = useCollapsedBlocks()

  // Auto-collapse think blocks when they complete
  const autoCollapsedRef = useRef<Set<string>>(new Set())
  useEffect(() => {
    const messages = display?.messages ?? []
    for (const msg of messages) {
      if (msg.type === 'think_block' && msg.status === 'completed' && !autoCollapsedRef.current.has(msg.id)) {
        autoCollapsedRef.current.add(msg.id)
        collapseBlock(msg.id)
      }
    }
  }, [display?.messages, collapseBlock])

  const theme = useTheme()
  const { showCopiedToast: clipboardToast } = useSelectionAutoCopy()

  // Ephemeral status bar message (auto-dismisses)
  const [ephemeralMessage, setEphemeralMessage] = useState<{ text: string; color: string } | null>(null)
  const ephemeralTimerRef = useRef<NodeJS.Timeout | null>(null)

  const showEphemeral = useCallback((message: string, color: string, durationMs = 5000) => {
    if (ephemeralTimerRef.current) clearTimeout(ephemeralTimerRef.current)
    setEphemeralMessage({ text: message, color })
    ephemeralTimerRef.current = setTimeout(() => setEphemeralMessage(null), durationMs)
  }, [])

  useEffect(() => {
    return () => { if (ephemeralTimerRef.current) clearTimeout(ephemeralTimerRef.current) }
  }, [])

  const hasRunningForks = agentStatusState
    ? Array.from(agentStatusState.agents.values()).some(a => a.status === 'working')
    : false

  // Find pending approval request from display messages (for keyboard intercept)
  const pendingApproval = useMemo(() => {
    if (!display) return null
    const msg = display.messages.find(
      m => m.type === 'approval_request' && m.status === 'pending'
    )
    return msg?.type === 'approval_request' ? msg : null
  }, [display?.messages])


  const handleApprove = useCallback(() => {
    if (!client || !pendingApproval) return
    logger.info({ toolCallId: pendingApproval.toolCallId }, 'Approving tool call')
    client.send({
      type: 'tool_approved',
      forkId: null,
      toolCallId: pendingApproval.toolCallId,
    })
  }, [client, pendingApproval])

  const handleReject = useCallback(() => {
    if (!client || !pendingApproval) return
    logger.info({ toolCallId: pendingApproval.toolCallId }, 'Rejecting tool call')
    client.send({
      type: 'tool_rejected',
      forkId: null,
      toolCallId: pendingApproval.toolCallId,
      reason: 'User rejected',
    })
  }, [client, pendingApproval])


  // Slash command context
  const resetConversation = useCallback(() => {
    if (client) {
      client.dispose()
    }
    onReset()
  }, [client, onReset])

  const showSystemMessage = useCallback((message: string, durationMs = 10000) => {
    const id = createId()
    const existingTimeout = systemMessageTimeoutsRef.current.get(id)
    if (existingTimeout) clearTimeout(existingTimeout)

    setSystemMessages(prev => [...prev, { id, text: message, timestamp: Date.now() }])

    const timeoutId = setTimeout(() => {
      systemMessageTimeoutsRef.current.delete(id)
      setSystemMessages(prev => prev.filter(m => m.id !== id))
    }, durationMs)

    systemMessageTimeoutsRef.current.set(id, timeoutId)
  }, [])

  useEffect(() => {
    return () => {
      for (const timeoutId of systemMessageTimeoutsRef.current.values()) {
        clearTimeout(timeoutId)
      }
      systemMessageTimeoutsRef.current.clear()
    }
  }, [])

  const exitApp = useCallback(() => {
    process.kill(process.pid, 'SIGINT')
  }, [])

  const openRecentChats = useCallback(() => {
    refreshRecentChats()
    setShowRecentChatsOverlay(true)
  }, [refreshRecentChats])

  const modeColor = theme.modeDefault
  const modeLabel = 'Default'

  const enterBashMode = useCallback(() => {
    setBashMode(true)
  }, [])

  const activateSkill = useCallback((skillName: string, skillPath: string | undefined, args: string) => {
    if (!skillPath) {
      showEphemeral(`Failed to activate /${skillName}: missing skill path`, theme.error, 8000)
      return
    }
    clientSend({
      type: 'skill_activated',
      forkId: null,
      skillName,
      skillPath,
      message: args.trim() || null,
      source: 'user',
    })
    logger.info({ skillName, skillPath, hasArgs: !!args.trim() }, 'Skill activated')
  }, [clientSend, showEphemeral, theme.error])

  const initProject = useCallback(() => {
    clientSend({
      type: 'user_message',
      messageId: createId(),
      timestamp: Date.now(),
      forkId: null,
      content: textParts(INIT_PROMPT),
      attachments: [],
      mode: 'text',
      synthetic: false,
      taskMode: false,
    })
    logger.info('Init project activated')
  }, [clientSend])

  const openSettings = useCallback((tab: SettingsTab = 'provider') => {
    setSettingsTab(tab)
  }, [])

  const openBrowserSetup = useCallback(() => {
    setShowBrowserSetup(true)
  }, [])

  const exitBashMode = useCallback(() => {
    setBashMode(false)
  }, [])

  const handleResumeChat = useCallback((chat: RecentChat) => {
    hasAnimatedRef.current = true
    setShowRecentChatsOverlay(false)
    onResumeSession(chat.id)
  }, [onResumeSession])

  const hasActivity = hasConversationActivity({
    displayMessageCount: (display?.messages ?? []).length,
    bashOutputCount: bashOutputs.length,
  })

  // Navigation for startup widget (active when no activity, overlay closed, and input empty)
  const widgetNavActive = !showRecentChatsOverlay && !hasActivity && !composerHasContent
  const widgetNavigation = useRecentChatsNavigation(
    recentChats ? recentChats.slice(0, 5) : [],
    handleResumeChat,
    widgetNavActive,
  )

  useEffect(() => {
    if (showRecentChatsOverlay) {
      setRecentChatsSelectedIndex(0)
    }
  }, [showRecentChatsOverlay])

  // Model selection overlay handlers
  const handleModelSelect = useCallback(async (providerId: string, modelId: string) => {
    if (!selectingModelFor || !providerUiState) return

    const selection: ModelSelection = { providerId, modelId }
    const auth = await providerRuntime.auth.getAuth(providerId)
    await providerRuntime.state.setSelection(selectingModelFor, providerId, modelId, auth ?? null)
    await storage.config.setModelSelection(selectingModelFor, selection)
    await storage.config.updateFull((current) => {
      const existing = current.providers?.[providerId] ?? {}
      const rememberedRaw = (existing as any).rememberedModelIds
      const remembered = Array.isArray(rememberedRaw) ? rememberedRaw.filter((id): id is string => typeof id === 'string') : []
      return {
        ...current,
        providers: {
          ...(current.providers ?? {}),
          [providerId]: {
            ...existing,
            modelId,
            rememberedModelIds: Array.from(new Set([...remembered, modelId])),
          },
        },
      }
    })
    setSlotModels(prev => ({ ...prev, [selectingModelFor]: selection }))

    await reloadProviderState()
    setSelectingModelFor(null)
  }, [selectingModelFor, providerUiState, providerRuntime, reloadProviderState, storage])

  const applySlotModelMap = useCallback(async (models: Preset['models']) => {
    for (const slot of MAGNITUDE_SLOTS) {
      const selection = models[slot]
      if (selection) {
        const auth = await providerRuntime.auth.getAuth(selection.providerId)
        await providerRuntime.state.setSelection(slot, selection.providerId, selection.modelId, auth ?? null)
        await storage.config.setModelSelection(slot, selection)
      } else {
        await providerRuntime.state.clear(slot)
        await storage.config.setModelSelection(slot, null)
      }
    }
    const slotMap = Object.fromEntries(
      MAGNITUDE_SLOTS.map((slot) => [slot, models[slot] ?? null])
    ) as Record<MagnitudeSlot, ModelSelection | null>
    setSlotModels(slotMap)
    await reloadProviderState()
  }, [providerRuntime, storage, reloadProviderState])

  const refreshPresets = useCallback(async () => {
    const loaded: Preset[] = await storage.config.getPresets()
    setPresets(loaded)
  }, [storage])

  const handleSavePreset = useCallback(async (name: string) => {
    await storage.config.savePreset(name, slotModels)
    await refreshPresets()
  }, [storage, slotModels, refreshPresets])

  const handleDeletePreset = useCallback(async (name: string) => {
    await storage.config.deletePreset(name)
    await refreshPresets()
  }, [storage, refreshPresets])

  const handleLoadPreset = useCallback(async (name: string, preferredProviderId?: string) => {
    if (!providerUiState) return

    if (name === SYSTEM_DEFAULTS_PRESET) {
      if (!preferredProviderId) return
      const connectedProviderIds = new Set(providerUiState.detectedProviders.map(d => d.provider.id))
      const detectedAuthTypeByProviderId = new Map<string, string | null>()
      for (const detected of providerUiState.detectedProviders) {
        detectedAuthTypeByProviderId.set(detected.provider.id, detected.auth?.type ?? null)
      }

      const defaultModels: Record<MagnitudeSlot, ModelSelection | null> = emptySlotModels()

      const preferredProvider = PROVIDERS.find((provider) => provider.id === preferredProviderId)
      const isLocalPreferredProvider = preferredProvider?.providerFamily === 'local'
      const firstLocalModelId = isLocalPreferredProvider ? preferredProvider?.models[0]?.id ?? null : null

      if (isLocalPreferredProvider && firstLocalModelId) {
        for (const slot of MAGNITUDE_SLOTS) {
          defaultModels[slot] = { providerId: preferredProviderId, modelId: firstLocalModelId }
        }
      } else {
        for (const slot of MAGNITUDE_SLOTS) {
          defaultModels[slot] = resolveSlotDefaultSelection({
            allProviders: PROVIDERS,
            connectedProviderIds,
            slot,
            preferredProviderId,
            detectedAuthTypeByProviderId,
          })
        }
      }

      await applySlotModelMap(defaultModels)
      return
    }

    const preset = presets.find((p) => p.name === name)
    if (!preset) return
    await applySlotModelMap(preset.models)
  }, [providerUiState, presets, applySlotModelMap])

  const detectedProviders = providerUiState?.detectedProviders ?? []

  useEffect(() => {
    if (!providerDetailId) {
      setProviderDetailStatus(null)
      setProviderDetailOptions(undefined)
      return
    }
    setProviderDetailStatus(null)
    setProviderDetailOptions(undefined)
    let stale = false

    providerRuntime.auth.detectProviderAuthMethods(providerDetailId).then((status) => {
      if (!stale) setProviderDetailStatus(status)
    }).catch((error) => {
      logger.warn({
        providerId: providerDetailId,
        error: error instanceof Error ? error.message : String(error),
      }, 'Failed to load provider auth methods')
      if (!stale) setProviderDetailStatus(null)
    })

    storage.config.getProviderOptions(providerDetailId).then((options) => {
      if (!stale) setProviderDetailOptions(options)
    }).catch((error) => {
      logger.warn({
        providerId: providerDetailId,
        error: error instanceof Error ? error.message : String(error),
      }, 'Failed to load provider options')
      if (!stale) setProviderDetailOptions(undefined)
    })

    return () => { stale = true }
  }, [providerDetailId, providerRefreshKey, providerRuntime, storage])

  const providerDetailActions = useMemo(() => {
    if (!providerDetailStatus) return []
    const actions: Array<{ type: 'connect' | 'disconnect' | 'update-key' | 'retry-discovery' | 'configure-endpoint'; methodIndex: number; label: string }> = []
    const providerId = providerDetailStatus.provider.id
    const isLocal = localProviderSet.has(providerId)

    for (const m of providerDetailStatus.methods) {
      if (m.connected) {
        if (m.method.type === 'api-key' && m.source === 'stored') {
          actions.push({ type: 'update-key', methodIndex: m.methodIndex, label: 'Update Key' })
          actions.push({ type: 'disconnect', methodIndex: m.methodIndex, label: 'Disconnect' })
        } else if (!isLocal && (m.source === 'stored' || m.source === 'none')) {
          actions.push({ type: 'disconnect', methodIndex: m.methodIndex, label: 'Disconnect' })
        }
      } else if (!isLocal) {
        actions.push({ type: 'connect', methodIndex: m.methodIndex, label: 'Connect' })
      }
    }
    return actions
  }, [providerDetailStatus, localProviderSet])

  const connectedProviderIds = useMemo(
    () => new Set(detectedProviders.map(d => d.provider.id)),
    [detectedProviders],
  )

  const connectedProviders = useMemo(
    () => PROVIDERS.filter(p => connectedProviderIds.has(p.id)),
    [connectedProviderIds],
  )

  const authStatusesByProviderId = useMemo(() => {
    const map = new Map<string, ProviderAuthMethodStatus | null>()
    if (providerDetailId) {
      map.set(providerDetailId, providerDetailStatus)
    }
    return map
  }, [providerDetailId, providerDetailStatus])

  const detectedAuthTypeByProviderId = useMemo(() => {
    const map = new Map<string, string | null>()
    for (const detected of detectedProviders) {
      map.set(detected.provider.id, detected.auth?.type ?? null)
    }
    return map
  }, [detectedProviders])

  const modelItems = useMemo(() => {
    if (!selectingModelFor) return []
    return buildModelPickerItems({
      allProviders: PROVIDERS,
      connectedProviderIds,
      selectingModelFor,

      authStatusesByProviderId,
      detectedAuthTypeByProviderId,
    })
  }, [selectingModelFor, connectedProviderIds, providerUiState, authStatusesByProviderId, detectedAuthTypeByProviderId])

  const filteredModelItems = useMemo(() => {
    if (!selectingModelFor) return []
    return filterModelPickerItems({
      items: modelItems,
      selectingModelFor,
      showAllProviders,
      showRecommendedOnly,
      search: modelSearch,
    })
  }, [modelItems, selectingModelFor, showAllProviders, showRecommendedOnly, modelSearch])

  const modelNavigation = useModelSelectNavigation(
    filteredModelItems,
    handleModelSelect,
    settingsTab === 'model',
  )

  // Track wizard state in a ref so authFlow callbacks don't trigger re-creation
  const showSetupWizardRef = useRef(false)
  showSetupWizardRef.current = showSetupWizard

  // Auth flow (shared between settings overlay and setup wizard)
  const authFlow = useAuthFlow({
    onAuthSuccess: (providerId, providerName) => {
      setProviderRefreshKey(prev => prev + 1)

      if (showSetupWizardRef.current) {
        // Wizard mode: advance to models step with defaults for this provider
        const provider = getProvider(providerId)
        setWizardSelectedProviderId(providerId)

        const newWizardSlotModels: Record<MagnitudeSlot, ModelSelection | null> = emptySlotModels()
        const freshConnectedProviderIds = new Set([...connectedProviderIds, providerId])
        for (const slot of MAGNITUDE_SLOTS) {
          const selection = resolveSlotDefaultSelection({
            allProviders: PROVIDERS,
            connectedProviderIds: freshConnectedProviderIds,
            slot,
            preferredProviderId: providerId,
            detectedAuthTypeByProviderId,
          })
          if (selection) newWizardSlotModels[slot] = selection
        }
        setWizardSlotModels(newWizardSlotModels)
        setWizardConnectedProvider(providerName)
        setWizardStep('models')
      } else {
        // Settings mode: return to provider list
        setSettingsTab('provider')
        if (returnToProviderDetailRef.current) {
          setProviderDetailId(returnToProviderDetailRef.current)
          setProviderDetailSelectedIndex(0)
          returnToProviderDetailRef.current = null
        }
      }
    },
    onAuthCancel: () => {
      if (showSetupWizardRef.current) {
        // Wizard mode: go back to provider selection
        setWizardStep('provider')
      } else {
        // Settings mode
        setSettingsTab('provider')
        if (returnToProviderDetailRef.current) {
          setProviderDetailId(returnToProviderDetailRef.current)
          setProviderDetailSelectedIndex(0)
          returnToProviderDetailRef.current = null
        }
      }
    },
    onMessage: (msg) => showSystemMessage(msg),
    showEphemeral,
    theme,
    reloadProviderState,
  })

  // --- Setup wizard handlers ---

  const handleWizardProviderSelected = useCallback(async (providerId: string) => {
    const provider = getProvider(providerId)
    if (!provider) return
    setWizardSelectedProviderId(providerId)

    const isLocalProvider = localProviderSet.has(providerId)
    if (isLocalProvider) {
      // Local wizard path is models-first. Seed default endpoint in background if missing.
      const existing = await storage.config.getProviderOptions(providerId)
      const hasBaseUrl = typeof existing?.baseUrl === 'string' && existing.baseUrl.trim().length > 0
      if (!hasBaseUrl && provider.defaultBaseUrl) {
        await storage.config.updateFull((current) => {
          const providerOptions = current.providers?.[providerId] ?? {}
          return {
            ...current,
            providers: {
              ...(current.providers ?? {}),
              [providerId]: {
                ...providerOptions,
                baseUrl: provider.defaultBaseUrl,
              },
            },
          }
        })
      }

      await providerRuntime.catalog.refresh().catch(() => undefined)
      await reloadProviderState().catch(() => undefined)

      const latestOptions = await storage.config.getProviderOptions(providerId).catch(() => null)
      const discoveredModels = Array.isArray((latestOptions as any)?.discoveredModels)
        ? (latestOptions as any).discoveredModels
          .filter((m: { id?: unknown }) => typeof m?.id === 'string')
          .map((m: { id: string; name?: unknown }) => ({ id: String(m.id), name: typeof m.name === 'string' ? m.name : String(m.id) }))
        : []
      const rememberedModelIds = Array.isArray((latestOptions as any)?.rememberedModelIds)
        ? (latestOptions as any).rememberedModelIds.filter((id: any) => typeof id === 'string')
        : []
      setWizardSelectedProviderDiscoveredModels(discoveredModels)
      setWizardSelectedProviderRememberedModelIds(rememberedModelIds)

      const refreshedConnectedProviderIds = new Set([
        ...connectedProviderIds,
        providerId,
      ])

      const newWizardSlotModels = resolveLocalWizardSlotDefaults({
        slots: MAGNITUDE_SLOTS,
        providerId,
        existingSlotModels: slotModels,
        discoveredModelIds: discoveredModels.map((m: { id: string }) => m.id),
        rememberedModelIds,
        applyWizardDefaults: true,
      }) as Record<MagnitudeSlot, ModelSelection | null>

      setWizardSlotModels(newWizardSlotModels)
      setWizardConnectedProvider(provider.name)
      setWizardStep('local-provider')
      return
    }

    const detected = detectedProviders
    const match = detected.find(d => d.provider.id === providerId)

    if (match) {
      // Already authenticated and ready — compute model defaults and go to models step
      const newWizardSlotModels: Record<MagnitudeSlot, ModelSelection | null> = emptySlotModels()
      for (const slot of MAGNITUDE_SLOTS) {
        const selection = resolveSlotDefaultSelection({
          allProviders: PROVIDERS,
          connectedProviderIds,
          slot,
          preferredProviderId: providerId,
        })
        if (selection) newWizardSlotModels[slot] = selection
      }
      setWizardSlotModels(newWizardSlotModels)
      setWizardConnectedProvider(provider.name)
      setWizardSelectedProviderDiscoveredModels([])
      setWizardSelectedProviderRememberedModelIds([])
      setWizardStep('models')
    } else if (provider.authMethods.length === 1) {
      // Single auth method — start it directly
      authFlow.startAuthForProvider(provider, 0)
    } else {
      // Multiple auth methods — show picker
      authFlow.openAuthMethodPicker(provider)
    }
  }, [authFlow.startAuthForProvider, authFlow.openAuthMethodPicker, detectedProviders, connectedProviderIds, localProviderSet, storage, providerRuntime, reloadProviderState, slotModels])

  const finishWizard = useCallback(async () => {
    setShowSetupWizard(false)
    setWizardStep('provider')
    setWizardSlotModels(emptySlotModels())
    setWizardConnectedProvider(null)
    setWizardSelectedProviderId(null)
    setWizardSelectedProviderDiscoveredModels([])
    setWizardSelectedProviderRememberedModelIds([])
    setProviderRefreshKey(prev => prev + 1)

    // Initialize global skills on first run
    const { initGlobalSkills } = await import('@magnitudedev/skills')
    await initGlobalSkills().catch((err) => {
      logger.warn({ error: err.message }, 'Failed to initialize global skills')
    })
  }, [])

  const handleWizardComplete = useCallback(async (result: Record<MagnitudeSlot, ModelSelection | null>) => {
    if (!providerUiState) return
    authFlow.cancelAll()

    for (const [slot, selection] of Object.entries(result) as [MagnitudeSlot, ModelSelection | null][]) {
      if (!selection) continue
      const auth = await providerRuntime.auth.getAuth(selection.providerId)
      await providerRuntime.state.setSelection(slot, selection.providerId, selection.modelId, auth ?? null)
      await storage.config.setModelSelection(slot, selection)
    }
    setSlotModels(result)

    if (wizardNeedsChromium !== false) {
      await reloadProviderState()
      setWizardStep('browser')
    } else {
      await storage.config.setSetupComplete(true)
      await reloadProviderState()
      finishWizard()
    }
  }, [authFlow.cancelAll, wizardNeedsChromium, finishWizard, providerUiState, providerRuntime, reloadProviderState, storage])

  const handleWizardBrowserComplete = useCallback(async () => {
    if (!providerUiState) return
    await storage.config.setSetupComplete(true)
    await reloadProviderState()
    finishWizard()
  }, [finishWizard, providerUiState, reloadProviderState, storage])

  const handleWizardSkip = useCallback(async () => {
    if (!providerUiState) return
    authFlow.cancelAll()
    await storage.config.setSetupComplete(true)
    await reloadProviderState()
    finishWizard()
  }, [authFlow.cancelAll, finishWizard, providerUiState, reloadProviderState, storage])

  const handleWizardContinueFromLocalProvider = useCallback(() => {
    setWizardStep('models')
  }, [])

  const handleWizardBack = useCallback(() => {
    if (wizardStep === 'browser') {
      setWizardStep('models')
      return
    }
    if (wizardStep === 'models' && wizardHasProviderEndpointStep) {
      setWizardStep('local-provider')
      return
    }
    if (wizardStep === 'local-provider') {
      setWizardStep('provider')
      return
    }
    setWizardStep('provider')
    setWizardSlotModels(emptySlotModels())
    setWizardConnectedProvider(null)
  }, [wizardStep, wizardHasProviderEndpointStep])

  // Providers shown in the wizard (exclude cloud providers that require manual credential setup)
  const WIZARD_PROVIDERS = useMemo(() =>
    PROVIDERS,
  [])

  useEffect(() => {
    if (showSetupWizard && wizardStep === 'provider') {
      setWizardProviderSelectedIndex(0)
    }
    if (showSetupWizard && wizardStep === 'models') {
      setWizardModelSelectedIndex(0)
    }
  }, [showSetupWizard, wizardStep])

  const handleProviderSelect = useCallback((providerId: string) => {
    setProviderDetailId(providerId)
    setProviderDetailSelectedIndex(0)
  }, [])


  const handleProviderDisconnect = useCallback(async (providerId: string) => {
    if (!providerUiState) return
    await storage.auth.remove(providerId)

    const affectedSlots = MAGNITUDE_SLOTS.filter(slot => slotModels[slot]?.providerId === providerId)
    if (affectedSlots.length > 0) {
      for (const slot of affectedSlots) {
        await providerRuntime.state.clear(slot)
        await storage.config.setModelSelection(slot, null)
      }
      setSlotModels(prev => {
        const next = { ...prev }
        for (const slot of affectedSlots) next[slot] = null
        return next
      })
    }


    await storage.config.updateFull((current) => {
      const providers = { ...(current.providers ?? {}) }
      delete providers[providerId]
      return { ...current, providers }
    })

    await reloadProviderState()
    setProviderDetailSelectedIndex(0)
    setProviderRefreshKey(prev => prev + 1)
    showEphemeral(`Disconnected ${getProvider(providerId)?.name ?? providerId}`, theme.warning)
  }, [slotModels, showEphemeral, theme.warning, providerUiState, providerRuntime, reloadProviderState, storage])

  const handleProviderDetailBack = useCallback(() => {
    setProviderDetailId(null)
  }, [])

  const handleProviderUpdateKey = useCallback((providerId: string) => {
    const provider = getProvider(providerId)
    if (!provider) return
    const detected = detectedProviders.find(d => d.provider.id === providerId)
    const existingKey = detected?.auth?.type === 'api' ? (detected.auth as { type: 'api'; key: string }).key : undefined
    const method = provider.authMethods.find(m => m.type === 'api-key')
    authFlow.openApiKeyOverlay(provider, method?.envKeys?.[0] ?? '', existingKey)
    setProviderDetailId(null)
    setSettingsTab(null)
  }, [detectedProviders, authFlow.openApiKeyOverlay])

  const handleProviderDetailAction = useCallback((actionIndex: number) => {
    if (!providerDetailId) return
    const action = providerDetailActions[actionIndex]
    if (!action) return

    if (action.type === 'update-key') {
      handleProviderUpdateKey(providerDetailId)
    } else if (action.type === 'disconnect') {
      handleProviderDisconnect(providerDetailId)
    } else if (action.type === 'retry-discovery') {
      void providerRuntime.catalog.refresh()
        .then(() => reloadProviderState())
        .then(() => {
          setProviderRefreshKey(prev => prev + 1)
          showEphemeral(`Discovery refreshed for ${getProvider(providerDetailId)?.name ?? providerDetailId}`, theme.success)
        })
        .catch(() => {
          showEphemeral(`Couldn't refresh models right now`, theme.warning, 5000)
        })
    } else if (action.type === 'connect') {
      const provider = getProvider(providerDetailId)
      if (provider) {
        returnToProviderDetailRef.current = providerDetailId
        setProviderDetailId(null)
        setSettingsTab(null)
        authFlow.startAuthForProvider(provider, action.methodIndex)
      }
    }
  }, [providerDetailId, providerDetailActions, handleProviderUpdateKey, handleProviderDisconnect, authFlow.startAuthForProvider, providerRuntime, reloadProviderState, showEphemeral, theme.success, theme.warning])


  const handleLocalProviderSaveEndpoint = useCallback(async (providerId: string, url: string) => {
    const trimmed = url.trim()
    if (!trimmed) return
    const normalized = /^https?:\/\//.test(trimmed) ? trimmed : `http://${trimmed}`
    await storage.config.updateFull((current) => {
      const existing = current.providers?.[providerId] ?? {}
      return {
        ...current,
        providers: {
          ...(current.providers ?? {}),
          [providerId]: {
            ...existing,
            baseUrl: normalized,
          },
        },
      }
    })
    await providerRuntime.catalog.refresh().catch(() => undefined)
    await reloadProviderState().catch(() => undefined)
    const latestOptions = await storage.config.getProviderOptions(providerId).catch(() => null)
    if (wizardSelectedProviderId === providerId) {
      const discoveredModels = Array.isArray((latestOptions as any)?.discoveredModels)
        ? (latestOptions as any).discoveredModels
          .filter((m: { id?: unknown }) => typeof m?.id === 'string')
          .map((m: { id: string; name?: unknown }) => ({ id: String(m.id), name: typeof m.name === 'string' ? m.name : String(m.id) }))
        : []
      const rememberedModelIds = Array.isArray((latestOptions as any)?.rememberedModelIds)
        ? (latestOptions as any).rememberedModelIds.filter((id: any) => typeof id === 'string')
        : []
      setWizardSelectedProviderDiscoveredModels(discoveredModels)
      setWizardSelectedProviderRememberedModelIds(rememberedModelIds)
    }
    setProviderRefreshKey(prev => prev + 1)
    showEphemeral(localProviderSavedEndpointToast(getProvider(providerId)?.name ?? providerId), theme.success)
  }, [storage, providerRuntime, reloadProviderState, showEphemeral, theme.success, wizardSelectedProviderId])

  const handleLocalProviderRefreshModels = useCallback(async (providerId: string) => {
    await providerRuntime.catalog.refresh().catch(() => undefined)
    await reloadProviderState().catch(() => undefined)
    const latestOptions = await storage.config.getProviderOptions(providerId).catch(() => null)
    if (wizardSelectedProviderId === providerId) {
      const discoveredModels = Array.isArray((latestOptions as any)?.discoveredModels)
        ? (latestOptions as any).discoveredModels
          .filter((m: { id?: unknown }) => typeof m?.id === 'string')
          .map((m: { id: string; name?: unknown }) => ({ id: String(m.id), name: typeof m.name === 'string' ? m.name : String(m.id) }))
        : []
      const rememberedModelIds = Array.isArray((latestOptions as any)?.rememberedModelIds)
        ? (latestOptions as any).rememberedModelIds.filter((id: any) => typeof id === 'string')
        : []
      setWizardSelectedProviderDiscoveredModels(discoveredModels)
      setWizardSelectedProviderRememberedModelIds(rememberedModelIds)
    }
    setProviderRefreshKey(prev => prev + 1)
  }, [providerRuntime, reloadProviderState, storage, wizardSelectedProviderId])

  const handleLocalProviderAddManualModel = useCallback(async (providerId: string, modelId: string) => {
    const trimmed = modelId.trim()
    if (!trimmed) return
    await storage.config.updateFull((current) => {
      const existing = current.providers?.[providerId] ?? {}
      const rememberedRaw = (existing as any).rememberedModelIds
      const remembered = Array.isArray(rememberedRaw) ? rememberedRaw.filter((id): id is string => typeof id === 'string') : []
      return {
        ...current,
        providers: {
          ...(current.providers ?? {}),
          [providerId]: {
            ...existing,
            rememberedModelIds: Array.from(new Set([...remembered, trimmed])),
          },
        },
      }
    })
    await providerRuntime.catalog.refresh().catch(() => undefined)
    await reloadProviderState().catch(() => undefined)
    const latestOptions = await storage.config.getProviderOptions(providerId).catch(() => null)
    if (wizardSelectedProviderId === providerId) {
      const discoveredModels = Array.isArray((latestOptions as any)?.discoveredModels)
        ? (latestOptions as any).discoveredModels
          .filter((m: { id?: unknown }) => typeof m?.id === 'string')
          .map((m: { id: string; name?: unknown }) => ({ id: String(m.id), name: typeof m.name === 'string' ? m.name : String(m.id) }))
        : []
      const rememberedModelIds = Array.isArray((latestOptions as any)?.rememberedModelIds)
        ? (latestOptions as any).rememberedModelIds.filter((id: any) => typeof id === 'string')
        : []
      setWizardSelectedProviderDiscoveredModels(discoveredModels)
      setWizardSelectedProviderRememberedModelIds(rememberedModelIds)
    }
    setProviderRefreshKey(prev => prev + 1)
    showEphemeral(localProviderAddedModelToast(trimmed), theme.success)
  }, [storage, providerRuntime, reloadProviderState, showEphemeral, theme.success, wizardSelectedProviderId])

  const handleLocalProviderRemoveManualModel = useCallback(async (providerId: string, modelId: string) => {
    await storage.config.updateFull((current) => {
      const existing = current.providers?.[providerId] ?? {}
      const rememberedRaw = (existing as any).rememberedModelIds
      const remembered = Array.isArray(rememberedRaw) ? rememberedRaw.filter((id): id is string => typeof id === 'string') : []
      return {
        ...current,
        providers: {
          ...(current.providers ?? {}),
          [providerId]: {
            ...existing,
            rememberedModelIds: remembered.filter((id) => id !== modelId),
          },
        },
      }
    })
    await providerRuntime.catalog.refresh().catch(() => undefined)
    await reloadProviderState().catch(() => undefined)
    const latestOptions = await storage.config.getProviderOptions(providerId).catch(() => null)
    if (wizardSelectedProviderId === providerId) {
      const discoveredModels = Array.isArray((latestOptions as any)?.discoveredModels)
        ? (latestOptions as any).discoveredModels
          .filter((m: { id?: unknown }) => typeof m?.id === 'string')
          .map((m: { id: string; name?: unknown }) => ({ id: String(m.id), name: typeof m.name === 'string' ? m.name : String(m.id) }))
        : []
      const rememberedModelIds = Array.isArray((latestOptions as any)?.rememberedModelIds)
        ? (latestOptions as any).rememberedModelIds.filter((id: any) => typeof id === 'string')
        : []
      setWizardSelectedProviderDiscoveredModels(discoveredModels)
      setWizardSelectedProviderRememberedModelIds(rememberedModelIds)
    }
    setProviderRefreshKey(prev => prev + 1)
  }, [storage, providerRuntime, reloadProviderState, wizardSelectedProviderId])

  const handleLocalProviderSaveOptionalApiKey = useCallback(async (providerId: string, apiKey: string) => {
    const trimmed = apiKey.trim()
    if (trimmed.length === 0) {
      await storage.auth.remove(providerId)
      showEphemeral(localProviderSavedEmptyApiKeyToast(getProvider(providerId)?.name ?? providerId), theme.success)
    } else {
      await storage.auth.set(providerId, { type: 'api', key: trimmed })
      showEphemeral(localProviderSavedApiKeyToast(getProvider(providerId)?.name ?? providerId), theme.success)
    }
    await reloadProviderState().catch(() => undefined)
    setProviderRefreshKey(prev => prev + 1)
  }, [storage, reloadProviderState, showEphemeral, theme.success])

  const handleChangeSlot = useCallback((slot: MagnitudeSlot) => {
    resetModelPickerState()
    setSelectingModelFor(slot)
  }, [resetModelPickerState])

  const SLOT_UI_ORDER_KEYS: MagnitudeSlot[] = [...MAGNITUDE_SLOTS]

  // Combined model tab keyboard handler — switches between slot view and model picker
  const modelTabHandleKeyEvent = useCallback((key: KeyEvent): boolean => {
    if (settingsTab !== 'model') return false
    const plain = !key.ctrl && !key.meta && !key.option

    // When in model picker sub-view
    if (selectingModelFor) {
      // Esc goes back to slot view
      if (key.name === 'escape') {
        resetModelPickerState()
        return true
      }
      return modelNavigation.handleKeyEvent(key)
    }

    // Slot view navigation (7 items: 0-6)
    if (key.name === 'up' && plain) {
      setPreferencesSelectedIndex(prev => Math.max(0, prev - 1))
      return true
    }
    if (key.name === 'down' && plain) {
      setPreferencesSelectedIndex(prev => Math.min(6, prev + 1))
      return true
    }
    if (key.name === 'return' && plain) {
      handleChangeSlot(SLOT_UI_ORDER_KEYS[preferencesSelectedIndex])
      return true
    }
    return false
  }, [settingsTab, selectingModelFor, modelNavigation.handleKeyEvent, preferencesSelectedIndex, handleChangeSlot, resetModelPickerState])

  const providerNavigation = useProviderSelectNavigation(
    PROVIDERS,
    handleProviderSelect,
    settingsTab === 'provider',
  )

  useEffect(() => {
    if (authFlow.showAuthMethodOverlay) {
      setAuthMethodSelectedIndex(0)
    }
  }, [authFlow.showAuthMethodOverlay, authFlow.authMethodProvider?.id])



  const providerTabHandleKeyEvent = useCallback((key: KeyEvent): boolean => {
    if (providerDetailId) {
      const plain = !key.ctrl && !key.meta && !key.option
      if (key.name === 'escape' && !key.shift) {
        setProviderDetailId(null)
        return true
      }
      const actionCount = providerDetailActions.length
      if (actionCount > 0) {
        if (key.name === 'up' && plain) {
          setProviderDetailSelectedIndex(prev => Math.max(0, prev - 1))
          return true
        }
        if (key.name === 'down' && plain) {
          setProviderDetailSelectedIndex(prev => Math.min(actionCount - 1, prev + 1))
          return true
        }
        if ((key.name === 'return' || key.name === 'enter') && plain && !key.shift) {
          handleProviderDetailAction(providerDetailSelectedIndex)
          return true
        }
      }
      return false
    }
    return providerNavigation.handleKeyEvent(key)
  }, [providerDetailId, providerDetailActions, providerDetailSelectedIndex, providerNavigation, handleProviderDetailAction])

  const handleSettingsTabChange = useCallback((tab: SettingsTab) => {
    // If user navigates away from model tab while selecting, cancel selecting mode
    if (selectingModelFor && tab !== 'model') {
      resetModelPickerState()
    }
    setProviderDetailId(null)
    setSettingsTab(tab)
  }, [selectingModelFor, resetModelPickerState])

  const openSetup = useCallback(() => {
    setWizardStep('provider')
    setWizardProviderSelectedIndex(0)
    setWizardModelSelectedIndex(0)
    setWizardSlotModels(emptySlotModels())
    setWizardNeedsChromium(null)
    setShowSetupWizard(true)
  }, [])

  const commandContext: CommandContext = useMemo(() => ({
    resetConversation,
    showSystemMessage: (msg: string) => showEphemeral(msg, theme.error, 8000),
    exitApp,
    openRecentChats,
    enterBashMode,
    activateSkill,
    initProject,
    openSettings,
    openSetup,
    openBrowserSetup,
  }), [resetConversation, showEphemeral, theme.error, exitApp, openRecentChats, enterBashMode, activateSkill, initProject, openSettings, openSetup, openBrowserSetup])



  const handleInterruptFork = useCallback((forkId: string | null) => {
    if (!client) return
    logger.info({ forkId }, 'Sending interrupt event')
    client.send({ type: 'interrupt', forkId })
  }, [client])

  const handleInterrupt = useCallback(() => {
    handleInterruptFork(null)
  }, [handleInterruptFork])

  const handleInterruptAll = useCallback(() => {
    if (!client) return
    logger.info('Interrupt all: interrupting all workers')
    // Interrupt root with allKilled flag
    client.send({ type: 'interrupt', forkId: null, allKilled: true })
    // Interrupt every running fork
    if (agentStatusState) {
      for (const agent of agentStatusState.agents.values()) {
        if (agent.status === 'working') {
          client.send({ type: 'interrupt', forkId: agent.forkId })
        }
      }
    }
  }, [client, agentStatusState])

  const activeOverlayKind =
    (showSetupWizard && wizardStep === 'browser') ? 'setup-browser'
    : (showSetupWizard && wizardNeedsChromium !== null && !authFlow.oauthState && !authFlow.apiKeySetup && !authFlow.endpointSetup && !authFlow.showAuthMethodOverlay) ? 'setup-wizard'
    : showRecentChatsOverlay ? 'recent-chats'
    : (expandedForkId && client) ? 'fork-detail'
    : showBrowserSetup ? 'browser-setup'
    : settingsTab !== null ? 'settings'
    : (authFlow.showAuthMethodOverlay && authFlow.authMethodProvider) ? 'auth-method'
    : authFlow.endpointSetup ? 'provider-endpoint'
    : authFlow.apiKeySetup ? 'api-key'
    : authFlow.oauthState ? 'oauth'
    : 'none'

  const isOverlayActive = activeOverlayKind !== 'none'
  const isBlockingOverlayActive = isOverlayActive
  const canToggleRecentChatsWithCtrlR = activeOverlayKind === 'none' || activeOverlayKind === 'recent-chats'

  useKeyboard(
    useCallback(
      (key: KeyEvent) => {
        if (key.defaultPrevented) return

        const isCtrlC = key.ctrl && key.name === 'c' && !key.meta && !key.option
        const isCtrlX = key.ctrl && key.name === 'x' && !key.meta && !key.option
        const isCtrlR = key.ctrl && key.name === 'r' && !key.meta && !key.option

        if (isCtrlC) {
          if (composerHasContent) return
          if ((activeDisplay ?? display)?.status === 'streaming') return
          key.preventDefault()
          process.kill(process.pid, 'SIGINT')
          return
        }

        if (isCtrlX && debugMode) {
          key.preventDefault()
          setDebugPanelVisible(prev => !prev)
          return
        }

        if (isCtrlR) {
          if (!canToggleRecentChatsWithCtrlR) return
          key.preventDefault()
          hasAnimatedRef.current = true
          setShowRecentChatsOverlay(prev => !prev)
        }
      },
      [composerHasContent, debugMode, activeDisplay, display, canToggleRecentChatsWithCtrlR],
    ),
  )


  const {
    selectedFile,
    selectedFileContent,
    selectedFileStreaming,
    selectedFileResolvedPath,
    isOpen: isFilePanelOpen,
    canRenderPanel,
    openFile,
    closeFilePanel,
  } = useFilePanel({
    display: activeDisplay ?? display,
    toolState,
    workspacePath,
    projectRoot: process.cwd(),
  })

  // Find active expanded think block for sticky header
  const activeThinkBlock = useMemo(() => {
    const messages = display?.messages ?? []
    for (let i = messages.length - 1; i >= 0; i--) {
      const msg = messages[i]
      if (msg.type === 'think_block' && msg.status === 'active') {
        return msg
      }
    }
    return null
  }, [display?.messages])

  // Scroll-tracking for sticky header
  const scrollboxRef = useRef<any>(null)
  const thinkBlockRef = useRef<any>(null)
  const lastStreamingMessageIdRef = useRef<string | null>(null)
  const interruptedMessageIdRef = useRef<string | null>(null)
  const [headerScrolledOff, setHeaderScrolledOff] = useState(false)

  const snapChatToBottom = useCallback(() => {
    const scrollbox = scrollboxRef.current
    if (!scrollbox) return

    // OpenTUI ScrollBoxRenderable API: use scrollTo(...); scrollTop is a minimal fallback.
    if (typeof scrollbox.scrollTo === 'function') {
      scrollbox.scrollTo(Number.MAX_SAFE_INTEGER)
      return
    }

    if (typeof scrollbox.scrollTop === 'number') {
      scrollbox.scrollTop = Number.MAX_SAFE_INTEGER
    }
  }, [])

  useEffect(() => {
    const t = setTimeout(() => snapChatToBottom(), 0)
    return () => clearTimeout(t)
  }, [selectedForkId, snapChatToBottom])

  const selectedForkContentVersion = useMemo(
    () => getSelectedForkContentVersion(selectedForkId, forkDisplay),
    [selectedForkId, forkDisplay]
  )

  useEffect(() => {
    if (!selectedForkId) return

    const t1 = setTimeout(() => snapChatToBottom(), 0)
    const t2 = setTimeout(() => snapChatToBottom(), 50)

    return () => {
      clearTimeout(t1)
      clearTimeout(t2)
    }
  }, [selectedForkContentVersion, selectedForkId, snapChatToBottom])

  const showStickyHeader = activeThinkBlock != null && !isCollapsed(activeThinkBlock.id) && headerScrolledOff

  // Poll scroll position to detect when think block header scrolls off-screen
  useEffect(() => {
    if (!activeThinkBlock || isCollapsed(activeThinkBlock.id)) {
      setHeaderScrolledOff(false)
      return
    }

    const checkScroll = () => {
      const scrollbox = scrollboxRef.current
      const thinkBlockEl = thinkBlockRef.current
      if (!scrollbox || !thinkBlockEl) {
        setHeaderScrolledOff(false)
        return
      }

      // Compute absolute Y of think block within scrollbox content
      // by walking up the parent chain summing yoga computed tops
      let offsetY = 0
      let node: any = thinkBlockEl
      const contentNode = scrollbox.content
      while (node && node !== contentNode) {
        const yogaNode = node.yogaNode || node.getLayoutNode?.()
        if (yogaNode) {
          offsetY += yogaNode.getComputedTop()
        }
        node = node.parent
      }

      const scrollTop = scrollbox.scrollTop
      // Trigger 1 row before header fully scrolls off for seamless transition
      const isOff = scrollTop > offsetY - 1
      setHeaderScrolledOff(isOff)
    }

    const interval = setInterval(checkScroll, 50)
    checkScroll()

    return () => clearInterval(interval)
  }, [activeThinkBlock, isCollapsed])

  const onWizardCtrlCExit = useCallback(() => {
    if (providerUiState) {
      void storage.config.setSetupComplete(true).then(() => reloadProviderState())
    }
    authFlow.cancelAll()
    process.kill(process.pid, 'SIGINT')
  }, [providerUiState, storage, reloadProviderState, authFlow])

  const onSettingsClose = useCallback(() => {
    setSettingsTab(null)
    resetModelPickerState()
    setProviderDetailId(null)
  }, [resetModelPickerState])

  const handleBackFromModelPicker = useCallback(() => {
    resetModelPickerState()
  }, [resetModelPickerState])

  const handleSubmitViaClientBoundary = useCallback((payload: {
    forkId: string | null
    message: string
    attachments: Attachment[]
  }) => {
    clientSend({
      type: 'user_message',
      messageId: createId(),
      timestamp: Date.now(),
      forkId: payload.forkId,
      content: textParts(payload.message),
      attachments: payload.attachments,
      mode: 'text',
      synthetic: false,
      taskMode: false,
    })
  }, [clientSend])

  if (!display) {
    return (
      <SessionLoadingView
        sessionSelection={sessionSelection}
        recentChats={recentChats}
      />
    )
  }

  const overlayContent = (
    <AppOverlays
      showSetupWizard={showSetupWizard}
      wizardStep={wizardStep}
      wizardTotalSteps={wizardTotalSteps}
      wizardHasProviderEndpointStep={wizardHasProviderEndpointStep}
      wizardSlotModels={wizardSlotModels}
      wizardConnectedProvider={wizardConnectedProvider}
      wizardSelectedProviderId={wizardSelectedProviderId}
      wizardSelectedProviderDiscoveredModels={wizardSelectedProviderDiscoveredModels}
      wizardSelectedProviderRememberedModelIds={wizardSelectedProviderRememberedModelIds}
      wizardProviderSelectedIndex={wizardProviderSelectedIndex}
      wizardModelSelectedIndex={wizardModelSelectedIndex}
      showBrowserSetup={showBrowserSetup}
      setShowBrowserSetup={setShowBrowserSetup}
      handleWizardBrowserComplete={handleWizardBrowserComplete}
      handleWizardProviderSelected={handleWizardProviderSelected}
      handleWizardComplete={handleWizardComplete}
      handleWizardBack={handleWizardBack}
      handleWizardContinueFromLocalProvider={handleWizardContinueFromLocalProvider}
      handleWizardSkip={handleWizardSkip}
      onLocalProviderSaveEndpoint={handleLocalProviderSaveEndpoint}
      onLocalProviderRefreshModels={handleLocalProviderRefreshModels}
      onLocalProviderAddManualModel={handleLocalProviderAddManualModel}
      onLocalProviderRemoveManualModel={handleLocalProviderRemoveManualModel}
      onLocalProviderSaveOptionalApiKey={handleLocalProviderSaveOptionalApiKey}
      setWizardProviderSelectedIndex={setWizardProviderSelectedIndex}
      setWizardModelSelectedIndex={setWizardModelSelectedIndex}
      wizardProviders={WIZARD_PROVIDERS}
      onWizardCtrlCExit={onWizardCtrlCExit}
      authFlow={authFlow}
      authMethodSelectedIndex={authMethodSelectedIndex}
      setAuthMethodSelectedIndex={setAuthMethodSelectedIndex}
      detectedProviders={detectedProviders}
      connectedProviders={connectedProviders}
      slotModels={slotModels}
      selectingModelFor={selectingModelFor}
      setSelectingModelFor={setSelectingModelFor}
      preferencesSelectedIndex={preferencesSelectedIndex}
      setPreferencesSelectedIndex={setPreferencesSelectedIndex}
      providerDetailStatus={providerDetailStatus}
      providerDetailOptions={providerDetailOptions}
      providerDetailActions={providerDetailActions}
      providerDetailSelectedIndex={providerDetailSelectedIndex}
      setProviderDetailSelectedIndex={setProviderDetailSelectedIndex}
      settingsTab={settingsTab}
      handleSettingsTabChange={handleSettingsTabChange}
      handleModelSelect={handleModelSelect}
      modelSearch={modelSearch}
      onModelSearchChange={setModelSearch}
      showAllProviders={showAllProviders}
      onToggleShowAllProviders={() => setShowAllProviders(prev => !prev)}
      showRecommendedOnly={showRecommendedOnly}
      onToggleShowRecommendedOnly={() => setShowRecommendedOnly(prev => !prev)}
      handleProviderSelect={handleProviderSelect}
      handleProviderDetailAction={handleProviderDetailAction}
      handleProviderDetailBack={handleProviderDetailBack}
      handleChangeSlot={handleChangeSlot}
      modelTabHandleKeyEvent={modelTabHandleKeyEvent}
      providerTabHandleKeyEvent={providerTabHandleKeyEvent}
      modelNavigation={modelNavigation}
      providerNavigation={providerNavigation}
      onSettingsClose={onSettingsClose}
      onBackFromModelPicker={handleBackFromModelPicker}
      presets={presets}
      systemDefaultsPresetToken={SYSTEM_DEFAULTS_PRESET}
      onSavePreset={handleSavePreset}
      onLoadPreset={handleLoadPreset}
      onDeletePreset={handleDeletePreset}
      showRecentChatsOverlay={showRecentChatsOverlay}
      recentChats={recentChats}
      recentChatsSelectedIndex={recentChatsSelectedIndex}
      setRecentChatsSelectedIndex={setRecentChatsSelectedIndex}
      setShowRecentChatsOverlay={setShowRecentChatsOverlay}
      handleResumeChat={handleResumeChat}
      expandedForkId={expandedForkId}
      client={client}
      agentStatusState={agentStatusState}
      forkModelSummary={forkModelSummary}
      forkContextHardCap={forkContextHardCap}
      popForkOverlay={popForkOverlay}
      pushForkOverlay={pushForkOverlay}
      workspacePath={workspacePath}
      projectRoot={process.cwd()}
      showCopiedToast={clipboardToast}
    />
  )

  const chatScrollbox = (
    <scrollbox
      ref={scrollboxRef}
      focusable={false}
      stickyScroll
      stickyStart="bottom"
      scrollX={false}
      scrollbarOptions={{ visible: false }}
      verticalScrollbarOptions={{
        visible: display.status === 'idle',
        trackOptions: { width: 1 },
      }}
      style={{
        flexGrow: 1,
        rootOptions: {
          flexGrow: 1,
          backgroundColor: 'transparent',
        },
        wrapperOptions: {
          border: false,
          backgroundColor: 'transparent',
        },
        contentOptions: {
          paddingLeft: 1,
          paddingRight: 1,
          paddingTop: 1,
          justifyContent: 'flex-end',
        },
      }}
    >
      <box style={{ paddingLeft: 1, paddingBottom: 1 }}>
        <AnimatedLogo />
      </box>

      <box style={{ paddingLeft: 1, flexDirection: 'row' }}>
        <text style={{ fg: theme.muted }}>Current directory: </text>
        <text style={{ fg: theme.muted }} attributes={TextAttributes.BOLD}>{process.cwd().replace(process.env.HOME || '', '~')}</text>
      </box>

      <box style={{ paddingLeft: 1, paddingBottom: (hasActivity || (recentChats !== null && recentChats.length === 0)) ? 1 : 0, flexDirection: 'row' }}>
        <text style={{ fg: theme.foreground }} attributes={TextAttributes.BOLD}>Tip: </text>
        <text style={{ fg: theme.muted }}>Use </text>
        <text style={{ fg: theme.foreground }}>/settings</text>
        <text style={{ fg: theme.muted }}> to configure providers and models!</text>
      </box>

      {!hasActivity && (
        <box style={{ paddingLeft: 1 }}>
          <RecentChatsWidget
            chats={recentChats ? recentChats.slice(0, 5) : []}
            loading={recentChats === null}
            selectedIndex={widgetNavigation.selectedIndex}
            onSelect={handleResumeChat}
            onHoverIndex={widgetNavigation.setSelectedIndex}
            onOpenAll={openRecentChats}
            isNavigationActive={widgetNavActive}
          />
        </box>
      )}

      {hasMore && (
        <LoadPreviousButton hiddenCount={hiddenCount} onLoadMore={loadMore} />
      )}
      {(() => {
        type MergedItem = { kind: 'timeline'; item: (typeof visibleItems)[number] }

        const mergedItems: MergedItem[] = [
          ...visibleItems.map(item => ({ kind: 'timeline' as const, item })),
        ].sort((a, b) => a.item.timestamp - b.item.timestamp)

        return mergedItems.map((merged) => {
          const item = merged.item
          switch (item.kind) {
            case 'chat': {
              const msg = item.message
              const isStreamingMsg = display.status === 'streaming'
                && display.streamingMessageId === msg.id
              const isInterrupted = interruptedMessageIdRef.current === msg.id
              return (
                <ErrorBoundary key={msg.id} fallback={(err) => (
                  <box style={{ paddingLeft: 1 }}>
                    <text style={{ fg: theme.error }}>[Render error: {err.message}]</text>
                  </box>
                )}>
                  <MessageView
                    message={msg}
                    isStreaming={isStreamingMsg}
                    isInterrupted={isInterrupted}
                    isCollapsed={msg.type === 'think_block' ? isCollapsed(msg.id) : undefined}
                    onToggleCollapse={msg.type === 'think_block' ? () => toggleCollapse(msg.id) : undefined}
                    hideThinkBlockHeader={msg.type === 'think_block' && msg.status === 'active' && !isCollapsed(msg.id) && headerScrolledOff}
                    onThinkBlockHeaderRef={msg.type === 'think_block' && msg.status === 'active' ? (ref: any) => { thinkBlockRef.current = ref } : undefined}
                    pendingApproval={pendingApproval != null}
                    onApprove={handleApprove}
                    onReject={handleReject}

                    inputHasText={composerHasContent}
                    onFileClick={openFile}
                    onForkExpand={pushForkOverlay}
                  />
                </ErrorBoundary>
              )
            }
            case 'bash':
              return (
                <box key={item.id} style={{ paddingLeft: 1 }}>
                  <BashOutput result={item.result} />
                </box>
              )
            case 'system':
              return (
                <box key={item.id} style={{ paddingLeft: 1, paddingBottom: 1 }}>
                  <text style={{ fg: theme.muted }}>{item.text}</text>
                </box>
              )
          }
        })
      })()}
      {selectedForkId !== null && (
        <PendingCommunicationsPanel
          messages={activeDisplay?.pendingInboundCommunications ?? []}
          onFileClick={openFile}
        />
      )}
    </scrollbox>
  )

  const modelSummary = slotModels.lead ? {
    provider: getProvider(slotModels.lead.providerId)?.name ?? slotModels.lead.providerId,
    model: getProvider(slotModels.lead.providerId)?.models.find(m => m.id === slotModels.lead!.modelId)?.name ?? slotModels.lead.modelId,
  } : null

  const composerCanFocus = !showSetupWizard
    && !showBrowserSetup
    && !showRecentChatsOverlay
    && settingsTab === null
    && !authFlow.showAuthMethodOverlay
    && !authFlow.oauthState
    && !authFlow.endpointSetup
    && !authFlow.apiKeySetup
    && expandedForkId === null

  const debugVisible = debugMode && debugPanelVisible
  return (
    <SelectedFileProvider value={selectedFile}>
    {isOverlayActive && overlayContent}
    <box style={{ visible: !isOverlayActive, flexDirection: 'row', height: '100%', paddingBottom: 0, marginBottom: 0 }}>
      {/* Left column — debug panel (only when enabled and visible) */}
      {debugVisible && (
        <box style={{ width: '35%', flexShrink: 0, paddingLeft: 1, paddingBottom: 1 }}>
          <DebugPanel debugSnapshot={debugSnapshot} events={debugEvents} logs={debugLogs} onToggle={() => setDebugPanelVisible(false)} />
        </box>
      )}

      {/* Center column — chat, status bar, input, footer */}
      <box
        ref={chatColumn.ref}
        onSizeChange={chatColumn.onSizeChange}
        style={{ flexDirection: 'column', flexGrow: 1, minWidth: 0, position: 'relative', height: '100%', paddingBottom: 0, marginBottom: 0 }}
      >
        <box style={{ flexGrow: 1, minHeight: 0, flexDirection: 'column' }}>
          {showStickyHeader && activeThinkBlock && (
            <box style={{ flexShrink: 0, paddingLeft: 2 }}>
              <StickyWorkingHeader
                timerStartTime={activeThinkBlock.timestamp}
                onToggle={() => toggleCollapse(activeThinkBlock.id)}
                pendingApproval={pendingApproval != null}
              />
            </box>
          )}
          {chatScrollbox}
          <ChatController
            isBlockingOverlayActive={isBlockingOverlayActive}
            env={{
              status: (activeDisplay ?? display)?.status ?? 'idle',
              pendingApproval: pendingApproval != null,
              hasRunningForks,
              bashMode,
              modelsConfigured: !!slotModels.lead,
              modelSummary: activeModelSummary,
              tokenUsage: selectedForkId
                ? (forkLastActualInputTokens ?? (forkHasCompletedTurn ? forkTokenEstimate : null))
                : (lastActualInputTokens ?? (hasCompletedTurn ? tokenEstimate : null)),
              contextHardCap,
              isCompacting: selectedForkId ? forkIsCompacting : isCompacting,
              theme,
              modeColor,
              attachmentsMaxWidth,
              composerCanFocus,
              widgetNavActive,
              isSubagentView: selectedForkId !== null,
            }}
            services={{
              submitUserMessageToFork: ({ forkId, message, attachments }) => handleSubmitViaClientBoundary({ forkId, message, attachments }),
              runSlashCommand: (commandText: string) => routeSlashCommand(commandText, commandContext),
              executeBash: async (command: string) => {
                const { workspacePath: wp } = await ensureClientReady()
                return executeBashCommand(command, {
                  workspacePath: wp!,
                  projectRoot: process.cwd(),
                })
              },
              appendBashOutput: (result) => setBashOutputs(prev => [...prev, result]),
              recordBashCommand: (result) => {
                clientSend({
                  type: 'user_bash_command',
                  forkId: null,
                  timestamp: result.timestamp,
                  command: result.command,
                  cwd: result.cwd,
                  exitCode: result.exitCode,
                  stdout: result.stdout,
                  stderr: result.stderr,
                })
              },
              clearSystemBanners: () => {
                setSystemMessages([])
                for (const timeoutId of systemMessageTimeoutsRef.current.values()) {
                  clearTimeout(timeoutId)
                }
                systemMessageTimeoutsRef.current.clear()
                if (ephemeralTimerRef.current) clearTimeout(ephemeralTimerRef.current)
                setEphemeralMessage(null)
              },
              interruptFork: handleInterruptFork,
              interruptAll: handleInterruptAll,
              openSettings,

              handleWidgetKeyEvent: widgetNavigation.handleKeyEvent,
              enterBashMode: () => setBashMode(true),
              exitBashMode: exitBashMode,
              requestIdleSubagentClose: ({ forkId, agentId }) => {
                const agent = agentStatusState
                  ? Array.from(agentStatusState.agents.values()).find((a) => a.forkId === forkId && a.agentId === agentId)
                  : undefined
                const parentForkId = agent?.parentForkId ?? null
                client?.send({
                  type: 'subagent_idle_closed',
                  forkId,
                  parentForkId,
                  agentId,
                  source: 'idle_tab_close',
                })
              },
              requestActiveSubagentKill: ({ forkId, agentId }) => {
                const agent = agentStatusState
                  ? Array.from(agentStatusState.agents.values()).find((a) => a.forkId === forkId && a.agentId === agentId)
                  : undefined
                const parentForkId = agent?.parentForkId ?? null
                client?.send({
                  type: 'subagent_user_killed',
                  forkId,
                  parentForkId,
                  agentId,
                  source: 'tab_close_confirm',
                })
              },
            }}
            displayMessages={(activeDisplay ?? display).messages}
            tasks={tasks}
            selectedForkId={selectedForkId}
            pushForkOverlay={pushForkOverlay}
            selectedFileOpen={isFilePanelOpen}
            onCloseFilePanel={closeFilePanel}
            onApprove={handleApprove}
            onReject={handleReject}
            onInputHasTextChange={setComposerHasContent}
            restoredQueuedInputText={restoredQueuedInputText}
            onRestoredQueuedInputHandled={() => setRestoredQueuedInputText(null)}
          />
        </box>

        {/* Clipboard copy toast — bottom-right overlay */}
        {clipboardToast && (
          <box style={{ position: 'absolute', bottom: 1, right: 2 }}>
            <box style={{
              borderStyle: 'single',
              border: ['left'],
              borderColor: theme.success,
              customBorderChars: { ...BOX_CHARS, vertical: '┃' },
            }}>
              <box style={{
                backgroundColor: theme.surface,
                paddingTop: 1,
                paddingBottom: 1,
                paddingLeft: 2,
                paddingRight: 2,
              }}>
                <text style={{ fg: theme.success }}>Copied to clipboard</text>
              </box>
            </box>
          </box>
        )}


        {providerUiState && !providerUiState.slotModels.lead && (
          <box style={{
            paddingLeft: 1,
            paddingRight: 1,
            flexShrink: 0,
          }}>
            <box style={{
              borderStyle: 'single',
              borderColor: theme.error,
              customBorderChars: BOX_CHARS,
              paddingLeft: 1,
              paddingRight: 1,
            }}>
              <text style={{ fg: theme.error }}>
                No model configured. Run /models to set up your models.
              </text>
            </box>
          </box>
        )}
      </box>

      {canRenderPanel && selectedFile && (
        <box style={{ width: '45%', flexShrink: 0, paddingRight: 1, paddingBottom: 1 }}>
          <FileViewerPanel
            key={selectedFile.path}
            filePath={selectedFile.path}
            content={selectedFileContent}
            scrollToSection={selectedFile.section}
            onClose={closeFilePanel}
            onOpenFile={openFile}
            streaming={selectedFileStreaming}
          />
        </box>
      )}

    </box>
    </SelectedFileProvider>
  )
}
