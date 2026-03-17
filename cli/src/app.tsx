import { useState, useEffect, useRef, useCallback, useMemo } from 'react'
import { useKeyboard, useRenderer } from '@opentui/react'
import { Effect, Layer, Cause } from 'effect'

import { createCodingAgentClient, ChatPersistence, scanSkills, getActiveCoreSkills, type DisplayState, type AgentStatusState, type DebugSnapshot, type AppEvent, type UnexpectedErrorMessage, PROVIDERS, getProvider, type ProviderDefinition, type AuthMethodDef, type ModelSelection, type ProviderAuthMethodStatus, type ForkMemoryState, type ForkCompactionState, type ArtifactState, type AgentsViewState, getLatestInProgressArtifactStream } from '@magnitudedev/agent'
import { textParts } from '@magnitudedev/agent'
import { JsonChatPersistence } from './persistence'

import { MessageView } from './components/message-view'
import { ErrorBoundary } from './components/error-boundary'
import { StickyWorkingHeader } from './components/think-block'
import { LoadPreviousButton } from './components/chat-controls'
import { Button } from './components/button'


import { usePaginatedTimeline } from './hooks/use-paginated-timeline'
import { useCollapsedBlocks } from './hooks/use-collapsed-blocks'

import { useTheme } from './hooks/use-theme'
import { ArtifactProvider, SelectedArtifactProvider } from './hooks/use-artifacts'
import { BOX_CHARS } from './utils/ui-constants'
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
import { getRecentChats, type RecentChat } from './data/recent-chats'
import { logger, initLogger, subscribeToLogs, clearSessionLog, getSessionLogPath, type LogEntry } from '@magnitudedev/logger'
import { readFileSync } from 'fs'
import path from 'path'
import { executeBashCommand, type BashResult } from './utils/bash-executor'


import { BashOutput } from './components/bash-output'


import { ArtifactReaderPanel } from './components/artifact-reader-panel'
import type { Attachment } from '@magnitudedev/agent'
import { DebugPanel } from './components/debug-panel'
import { ChatController } from './components/chat/chat-controller'

import { AgentsView } from './components/agents-view/agents-view'
import { AgentSummaryBar } from './components/agent-summary-bar'

import { initTelemetry, shutdownTelemetry, trackSessionStart, trackSessionEnd, trackUserMessage, trackTurnCompleted, trackToolUsage, trackAgentSpawned, trackAgentCompleted, trackCompaction, SessionTracker } from '@magnitudedev/telemetry'

import { setSessionTracker } from './utils/telemetry-state'
import { TextAttributes, type KeyEvent } from '@opentui/core'

import { createId } from '@magnitudedev/generate-id'

import { useProviderRuntime } from './providers/provider-runtime'
import { useStorage } from './providers/storage-provider'
import { useProviderUiState } from './hooks/use-provider-ui-state'

type AgentClient = Awaited<ReturnType<typeof createCodingAgentClient>>

function ViewAllActivityButton({ onViewAll, theme }: { onViewAll: () => void; theme: ReturnType<typeof useTheme> }) {
  const [hovered, setHovered] = useState(false)
  return (
    <box style={{ flexDirection: 'row', justifyContent: 'flex-end', paddingRight: 2 }}>
      <Button
        onClick={onViewAll}
        onMouseOver={() => setHovered(true)}
        onMouseOut={() => setHovered(false)}
      >
        <text style={{ wrapMode: 'none' }}>
          <span fg={hovered ? theme.primary : theme.muted}>{'View all activity →'}</span>
        </text>
      </Button>
    </box>
  )
}

function ViewMainChatButton({ onViewMain, theme }: { onViewMain: () => void; theme: ReturnType<typeof useTheme> }) {
  const [hovered, setHovered] = useState(false)
  return (
    <box style={{ flexDirection: 'row', justifyContent: 'flex-end', paddingRight: 2 }}>
      <Button
        onClick={onViewMain}
        onMouseOver={() => setHovered(true)}
        onMouseOut={() => setHovered(false)}
      >
        <text style={{ wrapMode: 'none' }}>
          <span fg={hovered ? theme.primary : theme.muted}>{'← View main chat'}</span>
        </text>
      </Button>
    </box>
  )
}

export function App({ resume, debug, onClientReady }: { resume: boolean; debug: boolean; onClientReady?: (client: AgentClient | null) => void }) {
  const [conversationKey, setConversationKey] = useState(0)
  const [sessionSelection, setSessionSelection] = useState<string | null | undefined>(resume ? undefined : null)
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
}: {
  debugMode: boolean
  skipAnimation: boolean
  sessionSelection: string | null | undefined
  onReset: () => void
  onResumeSession: (sessionId: string) => void
  onClientReady?: (client: AgentClient | null) => void
}) {
  const renderer = useRenderer()
  const providerRuntime = useProviderRuntime()
  const storage = useStorage()
  const { state: providerUiState, reload: reloadProviderState } = useProviderUiState()
  const [client, setClient] = useState<AgentClient | null>(null)
  const [display, setDisplay] = useState<DisplayState | null>(null)
  const [agentStatusState, setAgentStatusState] = useState<AgentStatusState | null>(null)
  const [agentsViewState, setAgentsViewState] = useState<AgentsViewState | null>(null)
  const [activeTab, setActiveTab] = useState<'main' | 'agents'>('main')
  const [hasUnreadMain, setHasUnreadMain] = useState(false)
  const activeTabRef = useRef<'main' | 'agents'>('main')
  const prevAssistantCountRef = useRef(0)
  const [artifactState, setArtifactState] = useState<ArtifactState | null>(null)
  const [selectedArtifact, setSelectedArtifact] = useState<{ name: string; section?: string } | null>(null)
  const [expandedForkStack, setExpandedForkStack] = useState<string[]>([])
  const expandedForkId = expandedForkStack.length > 0 ? expandedForkStack[expandedForkStack.length - 1] : null
  const agentsScrollboxRef = useRef<any>(null)
  const agentsScrollPositionRef = useRef<number | null>(null)
  const pushForkOverlay = (forkId: string) => {
    const scrollbox = agentsScrollboxRef.current
    if (scrollbox) {
      agentsScrollPositionRef.current = scrollbox.scrollTop ?? null
    }
    setExpandedForkStack(s => [...s, forkId])
  }
  const popForkOverlay = () => {
    setExpandedForkStack(s => {
      const newStack = s.slice(0, -1)
      if (newStack.length === 0) {
        setTimeout(() => {
          const scrollbox = agentsScrollboxRef.current
          if (scrollbox && agentsScrollPositionRef.current != null) {
            scrollbox.scrollTo(agentsScrollPositionRef.current)
            agentsScrollPositionRef.current = null
          }
        }, 0)
      }
      return newStack
    })
  }

  const [nextCtrlCWillExit, setNextCtrlCWillExit] = useState(false)
  const exitTimeoutRef = useRef<NodeJS.Timeout | null>(null)

  const [systemMessages, setSystemMessages] = useState<Array<{ id: string; text: string; timestamp: number }>>([])
  const systemMessageTimeoutsRef = useRef<Map<string, ReturnType<typeof setTimeout>>>(new Map())
  const [showRecentChatsOverlay, setShowRecentChatsOverlay] = useState(false)
  const [settingsTab, setSettingsTab] = useState<SettingsTab | null>(null)
  const [selectingModelFor, setSelectingModelFor] = useState<'primary' | 'secondary' | 'browser' | null>(null)
  const [primaryModel, setPrimaryModelState] = useState<ModelSelection | null>(null)
  const [secondaryModel, setSecondaryModelState] = useState<ModelSelection | null>(null)
  const [browserModel, setBrowserModelState] = useState<ModelSelection | null>(null)

  const [preferencesSelectedIndex, setPreferencesSelectedIndex] = useState(0)
  const [showAllProviders, setShowAllProviders] = useState(false)
  const [showRecommendedOnly, setShowRecommendedOnly] = useState(false)
  const [modelSearch, setModelSearch] = useState('')
  const [providerDetailId, setProviderDetailId] = useState<string | null>(null)
  const [providerDetailSelectedIndex, setProviderDetailSelectedIndex] = useState(0)
  const [providerRefreshKey, setProviderRefreshKey] = useState(0)
  const [providerDetailStatus, setProviderDetailStatus] = useState<ProviderAuthMethodStatus | null>(null)
  const [contextHardCap, setContextHardCap] = useState<number | null>(null)
  const [agentProjectionMode, setAgentProjectionMode] = useState<string>('default')
  const [bashMode, setBashMode] = useState(false)
  const [bashOutputs, setBashOutputs] = useState<BashResult[]>([])
  const [debugPanelVisible, setDebugPanelVisible] = useState(false)
  const [debugSnapshot, setDebugSnapshot] = useState<DebugSnapshot | null>(null)
  const [debugEvents, setDebugEvents] = useState<AppEvent[]>([])
  const [debugLogs, setDebugLogs] = useState<LogEntry[]>([])
  const [inputHasText, setInputHasText] = useState(false)
  const [restoredQueuedInputText, setRestoredQueuedInputText] = useState<string | null>(null)
  const [tokenEstimate, setTokenEstimate] = useState(0)
  const [isCompacting, setIsCompacting] = useState(false)
  const returnToProviderDetailRef = useRef<string | null>(null)
  const turnStartTimeRef = useRef<number | null>(null)
  const hasAnimatedRef = useRef(skipAnimation)
  const initClientRef = useRef<(() => Promise<AgentClient>) | null>(null)

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
  const contextPercent = contextHardCap ? Math.round((tokenEstimate / contextHardCap) * 100) : 0
  const contextDisplayText = contextHardCap
    ? `${contextPercent}% ${formatFooterTokens(tokenEstimate)}/${formatFooterTokens(contextHardCap)}`
    : ''
  const contextRenderedText = tokenEstimate > 0 && contextHardCap
    ? (isCompacting ? `>>> ${contextDisplayText} <<<` : contextDisplayText)
    : ''

  // Always reserve width for the longest possible escape hint so that
  // attachments don't reflow when hints appear/disappear.
  const maxEscHintWidth = 'Press Esc again to interrupt all subagents'.length

  const terminalWidth = process.stdout.columns ?? 80
  const footerRightGap = contextRenderedText ? 1 : 0
  const footerHorizontalPadding = 4
  const footerSafetyBuffer = 4
  const attachmentsMaxWidth = Math.max(
    0,
    terminalWidth
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
  const [wizardPrimaryModel, setWizardPrimaryModel] = useState<ModelSelection | null>(null)
  const [wizardSecondaryModel, setWizardSecondaryModel] = useState<ModelSelection | null>(null)
  const [wizardBrowserModel, setWizardBrowserModel] = useState<ModelSelection | null>(null)
  const [wizardConnectedProvider, setWizardConnectedProvider] = useState<string | null>(null)
  const [wizardNeedsChromium, setWizardNeedsChromium] = useState<boolean | null>(null)

  const wizardTotalSteps = useMemo(() => {
    if (wizardNeedsChromium === false) return 2
    return 3
  }, [wizardNeedsChromium])

  const [recentChats, setRecentChats] = useState<RecentChat[] | null>(null)

  const refreshRecentChats = useCallback(() => {
    getRecentChats(storage).then(setRecentChats)
  }, [storage])

  useEffect(() => {
    logger.info('App started')
    if (debugMode) logger.info('Debug mode enabled - press Ctrl+D to toggle debug panel')
    refreshRecentChats()
  }, [debugMode, refreshRecentChats])

  useEffect(() => {
    if (!providerUiState) return

    setPrimaryModelState(providerUiState.primaryModel)
    setSecondaryModelState(providerUiState.secondaryModel)
    setBrowserModelState(providerUiState.browserModel)

    initTelemetry({ telemetryEnabled: providerUiState.telemetryEnabled })

    if (!providerUiState.setupComplete) {
      setShowSetupWizard(true)
    }
  }, [providerUiState])

  useEffect(() => {
    providerRuntime.state.contextLimits('primary').then((limits) => {
      setContextHardCap(limits.hardCap)
    }).catch((error) => {
      logger.warn({ error: error instanceof Error ? error.message : String(error) }, 'Failed to load provider context limits')
    })
  }, [providerRuntime, primaryModel])

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
    scanSkills(process.cwd()).then((skills) => {
      const commands: SlashCommandDefinition[] = []

      // Register core skills that aren't overridden by user skills
      const activeCoreSkills = getActiveCoreSkills(skills)
      for (const core of activeCoreSkills) {
        commands.push({
          id: core.name,
          label: core.name,
          description: core.description,
          source: 'skill' as const,
        })
      }

      // Register user skills
      for (const s of skills) {
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
      logger.warn({ error: err.message }, 'Failed to scan skills')
    })

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
      initLogger(persistence.getSessionId())
      clearSessionLog(persistence.getSessionId())
      logger.info({ logFile: getSessionLogPath(persistence.getSessionId()) }, 'Session logger initialized')
      const persistenceLayer = Layer.succeed(ChatPersistence, persistence)
      return createCodingAgentClient({
        persistence: persistenceLayer,
        storage,
        debug: debugMode
      })
    }

    const setupClient = (client: AgentClient) => {
      if (!mounted) {
        client.dispose()
        return
      }
      c = client
      setClient(client)
      onClientReady?.(client)
      renderer.setTerminalTitle("Magnitude")

      // Telemetry tracking state
      const sessionTracker = new SessionTracker()
      setSessionTracker(sessionTracker)
      const forkRoles = new Map<string, { role: string; startTime: number }>()
      const roleToSlot = (role: string): 'primary' | 'secondary' | 'browser' => {
        if (role === 'browser') return 'browser'
        if (role === 'orchestrator') return 'primary'
        return 'secondary'
      }

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
          trackUserMessage({
            mode: event.mode,
            synthetic: event.synthetic,
            taskMode: event.taskMode,
            hasAttachments: event.attachments.length > 0,
          })
          sessionTracker.recordUserMessage()
        }

        if (event.type === 'agent_created') {
          forkRoles.set(event.forkId, { role: event.role, startTime: Date.now() })
          trackAgentSpawned({
            agentType: event.role,
            mode: event.mode,
          })
          sessionTracker.recordAgentSpawned()
        }

        if (event.type === 'agent_dismissed') {
          const forkInfo = forkRoles.get(event.forkId)
          if (forkInfo) {
            trackAgentCompleted({
              agentType: forkInfo.role,
              durationSeconds: Math.round((Date.now() - forkInfo.startTime) / 1000),
            })
            forkRoles.delete(event.forkId)
          }
        }

        if (event.type === 'turn_completed') {
          const forkInfo = event.forkId ? forkRoles.get(event.forkId) : null
          const agentRole = forkInfo?.role ?? 'orchestrator'
          const slot = roleToSlot(agentRole)

          providerRuntime.state.peek(slot).then((resolved) => {
            trackTurnCompleted({
              providerId: resolved?.model.providerId ?? null,
              modelId: resolved?.model.id ?? null,
              modelSlot: slot,
              authType: resolved?.auth?.type ?? null,
              inputTokens: event.inputTokens,
              outputTokens: event.outputTokens,
              cacheReadTokens: event.cacheReadTokens,
              cacheWriteTokens: event.cacheWriteTokens,
              toolCount: event.toolCalls.length,
              success: event.result.success,
              forkId: event.forkId,
              agentRole,
            })
          }).catch(() => {
            trackTurnCompleted({
              providerId: null,
              modelId: null,
              modelSlot: slot,
              authType: null,
              inputTokens: event.inputTokens,
              outputTokens: event.outputTokens,
              cacheReadTokens: event.cacheReadTokens,
              cacheWriteTokens: event.cacheWriteTokens,
              toolCount: event.toolCalls.length,
              success: event.result.success,
              forkId: event.forkId,
              agentRole,
            })
          })

          // Track individual tool calls
          for (const tc of event.toolCalls) {
            let linesAdded: number | undefined
            let linesRemoved: number | undefined
            let linesWritten: number | undefined

            if (tc.result.status === 'success' && tc.result.display) {
              if (tc.result.display.type === 'edit_diff') {
                linesAdded = tc.result.display.diffs.reduce((sum, d) => sum + d.addedLines.length, 0)
                linesRemoved = tc.result.display.diffs.reduce((sum, d) => sum + d.removedLines.length, 0)
                sessionTracker.recordLinesAdded(linesAdded)
                sessionTracker.recordLinesRemoved(linesRemoved)
              } else if (tc.result.display.type === 'write_stats') {
                linesWritten = tc.result.display.linesWritten
                sessionTracker.recordLinesWritten(linesWritten)
              }
            }

            trackToolUsage({
              toolName: tc.toolName,
              group: tc.group,
              status: tc.result.status,
              linesAdded,
              linesRemoved,
              linesWritten,
              forkId: event.forkId,
              agentRole,
            })
          }

          sessionTracker.recordTurn(event.inputTokens, event.outputTokens, event.toolCalls.length)
        }

        if (event.type === 'compaction_completed') {
          trackCompaction({ tokensSaved: event.tokensSaved, success: true })
        }
        if (event.type === 'compaction_failed') {
          trackCompaction({ tokensSaved: 0, success: false })
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

      // Subscribe to artifact state (global projection)
      client.state.artifacts.subscribe((state) => {
        if (mounted) {
          setArtifactState(state)
        }
      })

      // Subscribe to agents view state (global projection)
      if (client.state.agentsView) {
        client.state.agentsView.subscribe((state: AgentsViewState) => {
          if (mounted) {
            setAgentsViewState(state)
          }
        })
      }

      // Subscribe to restore queued messages signal (only for main/root)
      client.on.restoreQueuedMessages(({ forkId, messages }) => {
        // Only restore if this is for the main agent (not a fork)
        if (mounted && forkId === null && messages.length > 0) {
          const restored = messages.join('\n')
          logger.info({ restored, length: restored.length }, 'Restoring queued messages to input')
          setRestoredQueuedInputText(restored)
        }
      })


      client.on.chatTitleGenerated(({ title }) => {
        if (mounted) {
          logger.info({ title }, 'Chat title generated')
          renderer.setTerminalTitle(title)
        }
      })
    }

    if (sessionSelection === null) {
      // NEW SESSION: defer client creation, show empty UI immediately
      setDisplay({
        status: 'idle',
        messages: [],
        currentTurnId: null,
        streamingMessageId: null,
        activeThinkBlockId: null,
        showButton: 'send',
        agentToolCounts: new Map(),
        agentStartedAt: new Map(),
      })
      // Store factory so handleSubmit can create client lazily
      initClientRef.current = async () => {
        const client = await createClient()
        setupClient(client)
        return client
      }
    } else {
      // RESUMED SESSION: create client immediately (existing behavior)
      initClientRef.current = null
      createClient().catch((err) => {
        logger.error({ error: err.message, stack: err.stack }, 'Failed to create agent client');
        throw err;
      }).then((client) => {
        logger.info('Agent client created successfully');
        setupClient(client)
      })
    }

    return () => {
      mounted = false
      initClientRef.current = null
      onClientReady?.(null)
      c?.dispose()
    }
  }, [debugMode, onClientReady, renderer, sessionSelection, storage])

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
      if (activeTabRef.current === 'agents') {
        const newCount = state.messages.filter(m => m.type === 'assistant_message').length
        if (newCount > prevAssistantCountRef.current) {
          setHasUnreadMain(true)
        }
        prevAssistantCountRef.current = newCount
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

    const unsubscribe = client.state.compaction.subscribeFork(null, (state: ForkCompactionState) => {
      setTokenEstimate(state.tokenEstimate)
      setIsCompacting(state.isCompacting)
    })

    return unsubscribe
  }, [client])

  // Subscribe to debug stream when debug mode is enabled and panel is visible
  useEffect(() => {
    if (!client || !debugMode || !debugPanelVisible) return

    const unsubscribe = client.subscribeDebug(null, (snapshot) => {
      setDebugSnapshot(snapshot)
    })

    return unsubscribe
  }, [client, debugMode, debugPanelVisible])


  const { visibleItems, hiddenCount, loadMore, hasMore } = usePaginatedTimeline(
    display?.messages ?? [],
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

  const toggleTaskPanel = useCallback(() => {
    // Task panel removed — work graph visualization TODO
  }, [])

  const modeColor = theme.modeDefault
  const modeLabel = 'Default'

  const enterBashMode = useCallback(() => {
    setBashMode(true)
  }, [])

  const activateSkill = useCallback((skillName: string, skillPath: string | undefined, args: string) => {
    if (!client) return
    try {
      if (skillPath) {
        // User skill — read from disk
        const raw = readFileSync(skillPath, 'utf-8')
        const body = raw.replace(/^---\r?\n[\s\S]*?\r?\n---\r?\n?/, '').trim()

        let message = `[User activated skill: ${skillName}]\n\n${body}`
        if (args.trim()) {
          message += `\n\n${args.trim()}`
        }

        client.send({ type: 'user_message', forkId: null, content: textParts(message), attachments: [], mode: 'text', synthetic: false, taskMode: false })
      } else {
        // Core skill — tell agent to activate via tool
        let message = `[User activated skill: ${skillName}]`
        if (args.trim()) {
          message += `\n\n${args.trim()}`
        }

        client.send({ type: 'user_message', forkId: null, content: textParts(message), attachments: [], mode: 'text', synthetic: false, taskMode: false })
      }
      logger.info({ skillName, skillPath, hasArgs: !!args.trim() }, 'Skill activated')
    } catch (err: any) {
      showEphemeral(`Failed to load skill: ${err.message}`, theme.error, 8000)
    }
  }, [client, showEphemeral, theme.error])

  const initProject = useCallback(() => {
    if (!client) return
    client.send({ type: 'user_message', forkId: null, content: textParts(INIT_PROMPT), attachments: [], mode: 'text', synthetic: false, taskMode: false })
    logger.info('Init project activated')
  }, [client])

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

  // Navigation for startup widget (active when no messages, overlay closed, and input empty)
  const widgetNavActive = !showRecentChatsOverlay && (display?.messages ?? []).length === 0 && !inputHasText
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

    if (selectingModelFor === 'primary') {
      setPrimaryModelState(selection)
      const auth = await providerRuntime.auth.getAuth(providerId)
      await providerRuntime.state.setSelection('primary', providerId, modelId, auth ?? null, { persist: false })
      await storage.config.setModelSelection('primary', selection)
    } else if (selectingModelFor === 'secondary') {
      setSecondaryModelState(selection)
      const auth = await providerRuntime.auth.getAuth(providerId)
      await providerRuntime.state.setSelection('secondary', providerId, modelId, auth ?? null)
      await storage.config.setModelSelection('secondary', selection)
    } else if (selectingModelFor === 'browser') {
      setBrowserModelState(selection)
      const auth = await providerRuntime.auth.getAuth(providerId)
      await providerRuntime.state.setSelection('browser', providerId, modelId, auth ?? null, { persist: false })
      await storage.config.setModelSelection('browser', selection)
    }

    await reloadProviderState()
    setSelectingModelFor(null)
  }, [selectingModelFor, providerUiState, providerRuntime, reloadProviderState, storage])

  const detectedProviders = providerUiState?.detectedProviders ?? []

  useEffect(() => {
    if (!providerDetailId) {
      setProviderDetailStatus(null)
      return
    }
    setProviderDetailStatus(null)
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
    return () => { stale = true }
  }, [providerDetailId, providerRefreshKey, providerRuntime])

  const providerDetailActions = useMemo(() => {
    if (!providerDetailStatus) return []
    const actions: Array<{ type: 'connect' | 'disconnect' | 'update-key'; methodIndex: number; label: string }> = []
    for (const m of providerDetailStatus.methods) {
      if (m.connected) {
        if (m.method.type === 'api-key' && m.source === 'stored') {
          actions.push({ type: 'update-key', methodIndex: m.methodIndex, label: 'Update Key' })
          actions.push({ type: 'disconnect', methodIndex: m.methodIndex, label: 'Disconnect' })
        } else if (m.source === 'stored' || m.source === 'none') {
          actions.push({ type: 'disconnect', methodIndex: m.methodIndex, label: 'Disconnect' })
        }
        // env-sourced auth: no actions (can't disconnect env vars)
      } else {
        actions.push({ type: 'connect', methodIndex: m.methodIndex, label: 'Connect' })
      }
    }
    return actions
  }, [providerDetailStatus])

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
      localProviderConfig: providerUiState?.localProviderConfig,
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

        const primarySelection = resolveSlotDefaultSelection({
          allProviders: PROVIDERS,
          connectedProviderIds,
          slot: 'primary',
          preferredProviderId: providerId,
          detectedAuthTypeByProviderId,
        })

        const secondarySelection = resolveSlotDefaultSelection({
          allProviders: PROVIDERS,
          connectedProviderIds,
          slot: 'secondary',
          preferredProviderId: providerId,
          detectedAuthTypeByProviderId,
        })

        const browserSelection = resolveSlotDefaultSelection({
          allProviders: PROVIDERS,
          connectedProviderIds,
          slot: 'browser',
          preferredProviderId: providerId,
          detectedAuthTypeByProviderId,
        })

        if (primarySelection && provider?.models.some(model => model.id === primarySelection.modelId)) setWizardPrimaryModel(primarySelection)
        if (secondarySelection) setWizardSecondaryModel(secondarySelection)
        setWizardBrowserModel(browserSelection)
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

  const handleWizardProviderSelected = useCallback((providerId: string) => {
    const provider = getProvider(providerId)
    if (!provider) return

    const detected = detectedProviders
    const match = detected.find(d => d.provider.id === providerId)

    if (match) {
      // Already authenticated — compute model defaults and go to models step
      const primarySelection = resolveSlotDefaultSelection({
        allProviders: PROVIDERS,
        connectedProviderIds,
        slot: 'primary',
        preferredProviderId: providerId,
      })

      const secondarySelection = resolveSlotDefaultSelection({
        allProviders: PROVIDERS,
        connectedProviderIds,
        slot: 'secondary',
        preferredProviderId: providerId,
      })

      const browserSelection = resolveSlotDefaultSelection({
        allProviders: PROVIDERS,
        connectedProviderIds,
        slot: 'browser',
        preferredProviderId: providerId,
      })

      if (primarySelection) setWizardPrimaryModel(primarySelection)
      if (secondarySelection) setWizardSecondaryModel(secondarySelection)
      setWizardBrowserModel(browserSelection)
      setWizardConnectedProvider(provider.name)
      setWizardStep('models')
    } else if (provider.authMethods.length === 1) {
      // Single auth method — start it directly
      authFlow.startAuthForProvider(provider, 0)
    } else {
      // Multiple auth methods — show picker
      authFlow.openAuthMethodPicker(provider)
    }
  }, [authFlow.startAuthForProvider, authFlow.openAuthMethodPicker, detectedProviders])

  const finishWizard = useCallback(() => {
    setShowSetupWizard(false)
    setWizardStep('provider')
    setWizardPrimaryModel(null)
    setWizardSecondaryModel(null)
    setWizardBrowserModel(null)
    setWizardConnectedProvider(null)
    setProviderRefreshKey(prev => prev + 1)
  }, [])

  const handleWizardComplete = useCallback(async (result: { primaryModel: ModelSelection; secondaryModel: ModelSelection; browserModel: ModelSelection | null }) => {
    if (!providerUiState) return
    authFlow.cancelAll()

    setPrimaryModelState(result.primaryModel)
    setSecondaryModelState(result.secondaryModel)
    if (result.browserModel) setBrowserModelState(result.browserModel)

    const primaryAuth = await providerRuntime.auth.getAuth(result.primaryModel.providerId)
    await providerRuntime.state.setSelection('primary', result.primaryModel.providerId, result.primaryModel.modelId, primaryAuth ?? null, { persist: false })
    await storage.config.setModelSelection('primary', result.primaryModel)

    const secondaryAuth = await providerRuntime.auth.getAuth(result.secondaryModel.providerId)
    await providerRuntime.state.setSelection('secondary', result.secondaryModel.providerId, result.secondaryModel.modelId, secondaryAuth ?? null)
    await storage.config.setModelSelection('secondary', result.secondaryModel)

    if (result.browserModel) {
      const browserAuth = await providerRuntime.auth.getAuth(result.browserModel.providerId)
      await providerRuntime.state.setSelection('browser', result.browserModel.providerId, result.browserModel.modelId, browserAuth ?? null, { persist: false })
      await storage.config.setModelSelection('browser', result.browserModel)
    }

    if (wizardNeedsChromium) {
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

  const handleWizardBack = useCallback(() => {
    if (wizardStep === 'browser') {
      setWizardStep('models')
      return
    }
    setWizardStep('provider')
    setWizardPrimaryModel(null)
    setWizardSecondaryModel(null)
    setWizardBrowserModel(null)
    setWizardConnectedProvider(null)
  }, [wizardStep])

  // Providers shown in the wizard (exclude cloud providers that require manual credential setup)
  const WIZARD_PROVIDERS = useMemo(() =>
    PROVIDERS.filter(p => !['google-vertex', 'google-vertex-anthropic', 'amazon-bedrock'].includes(p.id)),
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

    if (primaryModel?.providerId === providerId) {
      setPrimaryModelState(null)
      await providerRuntime.state.clear('primary')
      await storage.config.setModelSelection('primary', null)
    }
    if (secondaryModel?.providerId === providerId) {
      setSecondaryModelState(null)
      await providerRuntime.state.clear('secondary')
      await storage.config.setModelSelection('secondary', null)
    }
    if (browserModel?.providerId === providerId) {
      setBrowserModelState(null)
      await providerRuntime.state.clear('browser')
      await storage.config.setModelSelection('browser', null)
    }
    if (providerId === 'local') {
      await storage.config.setLocalProviderConfig(null)
    }

    await reloadProviderState()
    setProviderDetailSelectedIndex(0)
    setProviderRefreshKey(prev => prev + 1)
    showEphemeral(`Disconnected ${getProvider(providerId)?.name ?? providerId}`, theme.warning)
  }, [primaryModel, secondaryModel, browserModel, showEphemeral, theme.warning, providerUiState, providerRuntime, reloadProviderState, storage])

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
    } else if (action.type === 'connect') {
      const provider = getProvider(providerDetailId)
      if (provider) {
        returnToProviderDetailRef.current = providerDetailId
        setProviderDetailId(null)
        setSettingsTab(null)
        authFlow.startAuthForProvider(provider, action.methodIndex)
      }
    }
  }, [providerDetailId, providerDetailActions, handleProviderUpdateKey, handleProviderDisconnect, authFlow.startAuthForProvider])


  const handleChangePrimary = useCallback(() => {
    resetModelPickerState()
    setSelectingModelFor('primary')
  }, [resetModelPickerState])

  const handleChangeSecondary = useCallback(() => {
    resetModelPickerState()
    setSelectingModelFor('secondary')
  }, [resetModelPickerState])

  const handleChangeBrowser = useCallback(() => {
    resetModelPickerState()
    setSelectingModelFor('browser')
  }, [resetModelPickerState])

  // Combined model tab keyboard handler — switches between primary/secondary/browser view and model picker
  const modelTabHandleKeyEvent = useCallback((key: KeyEvent): boolean => {
    if (settingsTab !== 'model') return false
    const plain = !key.ctrl && !key.meta && !key.option

    // When in model picker sub-view
    if (selectingModelFor) {
      // Esc goes back to primary/secondary/browser view
      if (key.name === 'escape') {
        resetModelPickerState()
        return true
      }
      return modelNavigation.handleKeyEvent(key)
    }

    // Primary/secondary/browser view navigation (3 items: 0=primary, 1=secondary, 2=browser)
    if (key.name === 'up' && plain) {
      setPreferencesSelectedIndex(prev => Math.max(0, prev - 1))
      return true
    }
    if (key.name === 'down' && plain) {
      setPreferencesSelectedIndex(prev => Math.min(2, prev + 1))
      return true
    }
    if (key.name === 'return' && plain) {
      if (preferencesSelectedIndex === 0) handleChangePrimary()
      else if (preferencesSelectedIndex === 1) handleChangeSecondary()
      else handleChangeBrowser()
      return true
    }
    return false
  }, [settingsTab, selectingModelFor, modelNavigation.handleKeyEvent, preferencesSelectedIndex, handleChangePrimary, handleChangeSecondary, handleChangeBrowser, resetModelPickerState])

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
      if ((key.name === 'escape' || (key.name === 'b' && plain)) && !key.shift) {
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
    setWizardPrimaryModel(null)
    setWizardSecondaryModel(null)
    setWizardBrowserModel(null)
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



  const handleTabSwitch = useCallback((tab: 'main' | 'agents') => {
    setActiveTab(tab)
    activeTabRef.current = tab
    if (tab === 'main') setHasUnreadMain(false)
    if (tab === 'agents') {
      setTimeout(() => {
        const scrollbox = agentsScrollboxRef.current
        if (scrollbox) {
          const contentHeight = scrollbox.content?.yogaNode?.getComputedHeight?.() ?? 999999
          scrollbox.scrollTo(contentHeight)
        }
      }, 0)
    }
  }, [])

  const handleInterrupt = useCallback(() => {
    if (!client) return
    logger.info({ forkId: null }, 'Sending interrupt event')
    client.send({ type: 'interrupt', forkId: null })
  }, [client])

  const handleInterruptAll = useCallback(() => {
    if (!client) return
    logger.info('Interrupt all: interrupting all subagents')
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

  useKeyboard(
    useCallback(
      (key: KeyEvent) => {
        if (key.defaultPrevented) return

        const isCtrlC = key.ctrl && key.name === 'c' && !key.meta && !key.option
        const isCtrlD = key.ctrl && key.name === 'd' && !key.meta && !key.option
        const isCtrlR = key.ctrl && key.name === 'r' && !key.meta && !key.option

        if (isCtrlC) {
          key.preventDefault()
          if (nextCtrlCWillExit) {
            process.kill(process.pid, 'SIGINT')
          } else {
            setNextCtrlCWillExit(true)
            if (exitTimeoutRef.current) clearTimeout(exitTimeoutRef.current)
            exitTimeoutRef.current = setTimeout(() => setNextCtrlCWillExit(false), 2000)
          }
          return
        }

        if (isCtrlD && debugMode) {
          key.preventDefault()
          setDebugPanelVisible(prev => !prev)
          return
        }

        if (isCtrlR) {
          key.preventDefault()
          hasAnimatedRef.current = true
          setShowRecentChatsOverlay(prev => !prev)
        }
      },
      [nextCtrlCWillExit, debugMode],
    ),
  )



  const selectedArtifactContent = useMemo(() => {
    if (!selectedArtifact || !artifactState) return null
    const artifact = artifactState.artifacts.get(selectedArtifact.name)
    return artifact?.content ?? null
  }, [selectedArtifact, artifactState])

  // Freeze baseContent when a stream starts — don't let it update
  // as the artifact content changes during executing/saving phase.
  const frozenBaseContentRef = useRef<{ toolCallId: string; content: string | null } | null>(null)

  const selectedArtifactStreaming = useMemo(() => {
    if (!selectedArtifact || !display) return null
    const stream = getLatestInProgressArtifactStream(display, selectedArtifact.name, true)
    if (!stream) {
      frozenBaseContentRef.current = null
      return null
    }

    // Freeze base content on first sight of this stream
    if (!frozenBaseContentRef.current || frozenBaseContentRef.current.toolCallId !== stream.toolCallId) {
      frozenBaseContentRef.current = {
        toolCallId: stream.toolCallId,
        content: selectedArtifactContent,
      }
    }

    return {
      active: stream.phase === 'streaming',
      toolKey: stream.toolKey,
      phase: stream.phase,
      toolCallId: stream.toolCallId,
      ...(stream.preview.mode === 'write'
        ? {
            contentSoFar: stream.preview.contentSoFar,
          }
        : {
            oldStringSoFar: stream.preview.oldStringSoFar,
            newStringSoFar: stream.preview.newStringSoFar,
            replaceAll: stream.preview.replaceAll,
          }),
      baseContent: frozenBaseContentRef.current.content,
    }
  }, [selectedArtifact, display, selectedArtifactContent])

  const handleArtifactClick = useCallback((name: string, section?: string) => {
    setSelectedArtifact(prev => prev?.name === name && prev?.section === section ? null : { name, section })
  }, [])

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
    message: string
    attachments: Attachment[]
  }) => {
    const sendMessage = (c: AgentClient) => {
      c.send({ type: 'user_message', forkId: null, content: textParts(payload.message), attachments: payload.attachments, mode: 'text', synthetic: false, taskMode: false })
    }

    if (client) {
      sendMessage(client)
    } else if (initClientRef.current) {
      const initFn = initClientRef.current
      initClientRef.current = null
      initFn().then((newClient) => {
        sendMessage(newClient)
      }).catch((err) => {
        logger.error({ error: err.message, stack: err.stack }, 'Failed to create agent client')
      })
    }
  }, [client])

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
      wizardPrimaryModel={wizardPrimaryModel}
      wizardSecondaryModel={wizardSecondaryModel}
      wizardBrowserModel={wizardBrowserModel}
      wizardConnectedProvider={wizardConnectedProvider}
      wizardProviderSelectedIndex={wizardProviderSelectedIndex}
      wizardModelSelectedIndex={wizardModelSelectedIndex}
      showBrowserSetup={showBrowserSetup}
      setShowBrowserSetup={setShowBrowserSetup}
      handleWizardBrowserComplete={handleWizardBrowserComplete}
      handleWizardProviderSelected={handleWizardProviderSelected}
      handleWizardComplete={handleWizardComplete}
      handleWizardBack={handleWizardBack}
      handleWizardSkip={handleWizardSkip}
      setWizardProviderSelectedIndex={setWizardProviderSelectedIndex}
      setWizardModelSelectedIndex={setWizardModelSelectedIndex}
      wizardProviders={WIZARD_PROVIDERS}
      onWizardCtrlCExit={onWizardCtrlCExit}
      authFlow={authFlow}
      authMethodSelectedIndex={authMethodSelectedIndex}
      setAuthMethodSelectedIndex={setAuthMethodSelectedIndex}
      detectedProviders={detectedProviders}
      connectedProviders={connectedProviders}
      primaryModel={primaryModel}
      secondaryModel={secondaryModel}
      browserModel={browserModel}
      selectingModelFor={selectingModelFor}
      setSelectingModelFor={setSelectingModelFor}
      preferencesSelectedIndex={preferencesSelectedIndex}
      setPreferencesSelectedIndex={setPreferencesSelectedIndex}
      providerDetailStatus={providerDetailStatus}
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
      handleChangePrimary={handleChangePrimary}
      handleChangeSecondary={handleChangeSecondary}
      handleChangeBrowser={handleChangeBrowser}
      modelTabHandleKeyEvent={modelTabHandleKeyEvent}
      providerTabHandleKeyEvent={providerTabHandleKeyEvent}
      modelNavigation={modelNavigation}
      providerNavigation={providerNavigation}
      onSettingsClose={onSettingsClose}
      onBackFromModelPicker={handleBackFromModelPicker}
      showRecentChatsOverlay={showRecentChatsOverlay}
      recentChats={recentChats}
      recentChatsSelectedIndex={recentChatsSelectedIndex}
      setRecentChatsSelectedIndex={setRecentChatsSelectedIndex}
      setShowRecentChatsOverlay={setShowRecentChatsOverlay}
      handleResumeChat={handleResumeChat}
      expandedForkId={expandedForkId}
      client={client}
      agentStatusState={agentStatusState}
      popForkOverlay={popForkOverlay}
      pushForkOverlay={pushForkOverlay}
      localProviderConfig={providerUiState?.localProviderConfig
        ? {
            baseUrl: providerUiState.localProviderConfig.baseUrl ?? undefined,
            modelId: providerUiState.localProviderConfig.modelId ?? undefined,
          }
        : null}
    />
  )

  const isOverlayActive = (showSetupWizard && wizardStep === 'browser')
    || (showSetupWizard && !authFlow.oauthState && !authFlow.apiKeySetup && !authFlow.showLocalSetup && !authFlow.showAuthMethodOverlay)
    || showRecentChatsOverlay
    || (expandedForkId && client)
    || showBrowserSetup
    || settingsTab !== null
    || (authFlow.showAuthMethodOverlay && authFlow.authMethodProvider)
    || authFlow.showLocalSetup
    || authFlow.apiKeySetup
    || authFlow.oauthState

  if (isOverlayActive) return overlayContent

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

      <box style={{ paddingLeft: 1, paddingBottom: ((display?.messages ?? []).length > 0 || (recentChats !== null && recentChats.length === 0)) ? 1 : 0, flexDirection: 'row' }}>
        <text style={{ fg: theme.foreground }} attributes={TextAttributes.BOLD}>Tip: </text>
        <text style={{ fg: theme.muted }}>Use </text>
        <text style={{ fg: theme.foreground }}>/settings</text>
        <text style={{ fg: theme.muted }}> to configure providers and models!</text>
      </box>

      {(display?.messages ?? []).length === 0 && (
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

                    inputHasText={inputHasText}
                    onArtifactClick={handleArtifactClick}
                    onForkExpand={pushForkOverlay}
                    onViewAgents={() => handleTabSwitch('agents')}
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
    </scrollbox>
  )

  const modelSummary = primaryModel ? {
    provider: getProvider(primaryModel.providerId)?.name ?? primaryModel.providerId,
    model: getProvider(primaryModel.providerId)?.models.find(m => m.id === primaryModel.modelId)?.name ?? primaryModel.modelId,
  } : null

  const composerCanFocus = !showSetupWizard
    && !showBrowserSetup
    && !showRecentChatsOverlay
    && settingsTab === null
    && !authFlow.showAuthMethodOverlay
    && !authFlow.oauthState
    && !authFlow.showLocalSetup
    && !authFlow.apiKeySetup
    && expandedForkId === null

  const debugVisible = debugMode && debugPanelVisible

  return (
    <ArtifactProvider value={artifactState}>
    <SelectedArtifactProvider value={selectedArtifact?.name ?? null}>
    <box style={{ flexDirection: 'row', height: '100%', paddingBottom: 0, marginBottom: 0 }}>
      {/* Left column — debug panel (only when enabled and visible) */}
      {debugVisible && (
        <box style={{ width: '35%', flexShrink: 0, paddingLeft: 1, paddingBottom: 1 }}>
          <DebugPanel debugSnapshot={debugSnapshot} events={debugEvents} logs={debugLogs} onToggle={() => setDebugPanelVisible(false)} />
        </box>
      )}

      {/* Center column — chat, status bar, input, footer */}
      <box style={{ flexDirection: 'column', flexGrow: 1, minWidth: 0, position: 'relative', height: '100%', paddingBottom: 0, marginBottom: 0 }}>
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
          {activeTab === 'main' ? chatScrollbox : (
            <box style={{ flexGrow: 1, minWidth: 0 }}>
              <AgentsView
                items={agentsViewState?.items ?? []}
                activeActivityIds={agentsViewState?.activeActivityIds ?? new Map()}
                onForkExpand={pushForkOverlay}
                onArtifactClick={handleArtifactClick}
                scrollboxRef={agentsScrollboxRef}
                subscribeForkDisplay={(forkId, cb) => client!.state.display.subscribeFork(forkId, cb)}
              />
            </box>
          )}
          {agentsViewState && agentsViewState.items.length > 0 && activeTab === 'agents' && (
            <ViewMainChatButton onViewMain={() => handleTabSwitch('main')} theme={theme} />
          )}
          {agentsViewState && agentsViewState.items.length > 0 && activeTab === 'agents' && (
            <AgentSummaryBar
              agentsViewState={agentsViewState}
              onViewAll={() => handleTabSwitch('agents')}
              onArtifactClick={handleArtifactClick}
              activeTab={activeTab}
            />
          )}
          <ChatController
            env={{
              status: display.status,
              pendingApproval: pendingApproval != null,
              hasRunningForks,
              bashMode,
              modelsConfigured: !!primaryModel && !!secondaryModel && !!browserModel,
              modelSummary,
              tokenEstimate,
              contextHardCap,
              isCompacting,
              theme,
              modeColor,
              attachmentsMaxWidth,
              composerCanFocus,
              widgetNavActive,
              nextCtrlCWillExit,
            }}
            services={{
              submitUserMessage: ({ message, attachments }) => handleSubmitViaClientBoundary({ message, attachments }),
              runSlashCommand: (commandText: string) => routeSlashCommand(commandText, commandContext),
              executeBash: executeBashCommand,
              appendBashOutput: (result) => setBashOutputs(prev => [...prev, result]),
              clearSystemBanners: () => {
                setSystemMessages([])
                for (const timeoutId of systemMessageTimeoutsRef.current.values()) {
                  clearTimeout(timeoutId)
                }
                systemMessageTimeoutsRef.current.clear()
                if (ephemeralTimerRef.current) clearTimeout(ephemeralTimerRef.current)
                setEphemeralMessage(null)
              },
              interrupt: handleInterrupt,
              interruptAll: handleInterruptAll,
              openSettings,
              toggleTaskPanel,
              handleWidgetKeyEvent: widgetNavigation.handleKeyEvent,
              enterBashMode: () => setBashMode(true),
              exitBashMode: exitBashMode,
            }}
            displayMessages={display.messages}
            selectedArtifactOpen={selectedArtifact != null}
            onCloseArtifact={() => setSelectedArtifact(null)}
            onApprove={handleApprove}
            onReject={handleReject}
            onInputHasTextChange={setInputHasText}
            restoredQueuedInputText={restoredQueuedInputText}
            onRestoredQueuedInputHandled={() => setRestoredQueuedInputText(null)}
            activeTab={activeTab}
            hasActiveAgents={hasRunningForks}
            hasUnreadMain={hasUnreadMain}
            onTabSwitch={handleTabSwitch}
            aboveInputSlot={
              agentsViewState && agentsViewState.items.length > 0 && activeTab === 'main'
                ? (
                    <ViewAllActivityButton
                      onViewAll={() => handleTabSwitch('agents')}
                      theme={theme}
                    />
                  )
                : undefined
            }
            inputTopSlot={
              agentsViewState && agentsViewState.items.length > 0 && activeTab === 'main'
                ? (
                    <AgentSummaryBar
                      agentsViewState={agentsViewState}
                      onViewAll={() => handleTabSwitch('agents')}
                      onArtifactClick={handleArtifactClick}
                      activeTab={activeTab}
                      variant="main-content"
                    />
                  )
                : undefined
            }
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
              backgroundColor: theme.surface,
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


        {providerUiState && (!primaryModel || !secondaryModel || !browserModel) && (
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
                {(() => {
                  const missing = [
                    !primaryModel && 'primary',
                    !secondaryModel && 'secondary',
                    !browserModel && 'browser',
                  ].filter(Boolean) as string[]
                  const list = missing.length === 1
                    ? missing[0]
                    : missing.slice(0, -1).join(', ') + ', or ' + missing[missing.length - 1]
                  return `No ${list} model configured. Run /settings to set up your models.`
                })()}
              </text>
            </box>
          </box>
        )}
      </box>

      {selectedArtifact && (selectedArtifactContent !== null || selectedArtifactStreaming !== null) && (
        <box style={{ width: '45%', flexShrink: 0, paddingRight: 1, paddingBottom: 1 }}>
          <ArtifactReaderPanel
            key={selectedArtifact.name}
            artifactName={selectedArtifact.name}
            content={selectedArtifactContent}
            streaming={selectedArtifactStreaming ?? undefined}
            scrollToSection={selectedArtifact.section}
            onClose={() => setSelectedArtifact(null)}
            onOpenArtifact={handleArtifactClick}
          />
        </box>
      )}

    </box>
    </SelectedArtifactProvider>
    </ArtifactProvider>
  )
}
