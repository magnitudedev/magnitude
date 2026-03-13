import { useState, useEffect, useRef, useCallback, useMemo } from 'react'
import { useKeyboard, useRenderer } from '@opentui/react'
import { Effect, Layer, Cause } from 'effect'

import { createCodingAgentClient, ChatPersistence, scanSkills, getActiveCoreSkills, type DisplayState, type ForkState, type DebugSnapshot, type AppEvent, type UnexpectedErrorMessage, PROVIDERS, getProvider, detectBrowserModel, isBrowserCompatible, type ProviderDefinition, type AuthMethodDef, type ModelSelection, type ProviderAuthMethodStatus, type ForkMemoryState, type ForkCompactionState, type ArtifactState, type AgentRegistryState } from '@magnitudedev/agent'
import { textParts } from '@magnitudedev/agent'
import { JsonChatPersistence } from './persistence'
import { MultilineInput, type MultilineInputHandle } from './components/multiline-input'
import { MessageView } from './components/message-view'
import { ErrorBoundary } from './components/error-boundary'
import { StickyWorkingHeader } from './components/think-block'
import { LoadPreviousButton } from './components/chat-controls'
import { ForkDetailOverlay } from './components/fork-detail-overlay'
import { SlashCommandMenu } from './components/slash-command-menu'
import { usePaginatedTimeline } from './hooks/use-paginated-timeline'
import { useCollapsedBlocks } from './hooks/use-collapsed-blocks'
import { useSlashCommands } from './hooks/use-slash-commands'
import { useTheme } from './hooks/use-theme'
import { ArtifactProvider } from './hooks/use-artifacts'
import { readClipboardText } from './utils/clipboard'
import { readClipboardBitmap } from './utils/clipboard'
import { tryReadPastedImageFile } from './utils/pasted-image-path'
import { autoScaleImageAttachmentIfNeeded } from './utils/image-scaling'
import { AttachmentsBar } from './components/attachments-bar'
import { BOX_CHARS } from './utils/ui-constants'
import { AnimatedLogo } from './components/animated-logo'
import { RecentChatsWidget } from './components/recent-chats-widget'
import { RecentChatsOverlay } from './components/recent-chats-overlay'
import { SettingsOverlay } from './components/settings-overlay'
import { SessionLoadingView } from './components/session-loading-view'
import { AuthMethodOverlay } from './components/auth-method-overlay'
import { OAuthOverlay } from './components/oauth-overlay'
import { LocalProviderOverlay } from './components/local-provider-overlay'
import { ApiKeyOverlay } from './components/api-key-overlay'
import { routeSlashCommand, type CommandContext } from './commands/command-router'
import { INIT_PROMPT } from './commands/init-prompt'
import { registerSkillCommands, type SlashCommandDefinition } from './commands/slash-commands'
import { useSelectionAutoCopy } from './utils/clipboard'
import { useRecentChatsNavigation } from './hooks/use-recent-chats-navigation'
import { useModelSelectNavigation } from './hooks/use-model-select-navigation'
import { useProviderSelectNavigation } from './hooks/use-provider-select-navigation'
import { useSettingsNavigation, type SettingsTab } from './hooks/use-settings-navigation'
import { useAuthMethodNavigation } from './hooks/use-auth-method-navigation'
import { useAuthFlow } from './hooks/use-auth-flow'
import { useSetupWizardNavigation } from './hooks/use-setup-wizard-navigation'
import { SetupWizardOverlay, type WizardStep } from './components/setup-wizard-overlay'
import { BrowserSetupOverlay } from './components/browser-setup-overlay'
import { getDefaultModels } from './utils/model-preferences'
import { getRecentChats, type RecentChat } from './data/recent-chats'
import { logger, clearLog, getLogPath, logEvent, clearEventLog, configureSessionLogging, subscribeToLogs, type LogEntry } from '@magnitudedev/logger'
import { readFileSync } from 'fs'
import { executeBashCommand, type BashResult } from './utils/bash-executor'

import { orange } from './utils/theme'
import { BashOutput } from './components/bash-output'
import { Button } from './components/button'

import { ArtifactReaderPanel } from './components/artifact-reader-panel'
import type { ImageAttachment, ImageMediaType } from '@magnitudedev/agent'
import { ContextUsageBar } from './components/context-usage-bar'
import { DebugPanel } from './components/debug-panel'
import type { InputValue } from './types/store'
import { initTelemetry, shutdownTelemetry, trackSessionStart, trackSessionEnd, trackUserMessage, trackTurnCompleted, trackToolUsage, trackAgentSpawned, trackAgentCompleted, trackCompaction, SessionTracker } from '@magnitudedev/telemetry'

import { setSessionTracker } from './utils/telemetry-state'
import { TextAttributes, type KeyEvent } from '@opentui/core'
import stringWidth from 'string-width'
import { createId } from '@magnitudedev/generate-id'
import { applyTextEditWithSegments, insertPasteSegment, reconstituteInputText } from './utils/strings'
import { useProviderRuntime } from './providers/provider-runtime'
import { useProviderUiState } from './hooks/use-provider-ui-state'

type AgentClient = Awaited<ReturnType<typeof createCodingAgentClient>>

export function App({ resume, debug }: { resume: boolean; debug: boolean }) {
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
    />
  )
}

function AppInner({
  debugMode,
  skipAnimation,
  sessionSelection,
  onReset,
  onResumeSession,
}: {
  debugMode: boolean
  skipAnimation: boolean
  sessionSelection: string | null | undefined
  onReset: () => void
  onResumeSession: (sessionId: string) => void
}) {
  const renderer = useRenderer()
  const providerRuntime = useProviderRuntime()
  const { state: providerUiState, reload: reloadProviderState } = useProviderUiState()
  const [client, setClient] = useState<AgentClient | null>(null)
  const [display, setDisplay] = useState<DisplayState | null>(null)
  const [forkState, setForkState] = useState<ForkState | null>(null)
  const [artifactState, setArtifactState] = useState<ArtifactState | null>(null)
  const [agentRegistryState, setAgentRegistryState] = useState<AgentRegistryState | null>(null)
  const [selectedArtifact, setSelectedArtifact] = useState<{ name: string; section?: string } | null>(null)
  const [expandedForkStack, setExpandedForkStack] = useState<string[]>([])
  const expandedForkId = expandedForkStack.length > 0 ? expandedForkStack[expandedForkStack.length - 1] : null
  const pushForkOverlay = (forkId: string) => setExpandedForkStack(s => [...s, forkId])
  const popForkOverlay = () => setExpandedForkStack(s => s.slice(0, -1))
  const [inputValue, setInputValue] = useState<InputValue>({
    text: '',
    cursorPosition: 0,
    lastEditDueToNav: false,
    pasteSegments: [],
    selectedPasteSegmentId: null,
  })
  const [nextCtrlCWillExit, setNextCtrlCWillExit] = useState(false)
  const exitTimeoutRef = useRef<NodeJS.Timeout | null>(null)
  const [nextEscWillKillAll, setNextEscWillKillAll] = useState(false)
  const killAllTimeoutRef = useRef<NodeJS.Timeout | null>(null)
  const [nextEscWillClearInput, setNextEscWillClearInput] = useState(false)
  const clearInputTimeoutRef = useRef<NodeJS.Timeout | null>(null)
  const [systemMessages, setSystemMessages] = useState<Array<{ id: string; text: string; timestamp: number }>>([])
  const [showRecentChatsOverlay, setShowRecentChatsOverlay] = useState(false)
  const [settingsTab, setSettingsTab] = useState<SettingsTab | null>(null)
  const [selectingModelFor, setSelectingModelFor] = useState<'primary' | 'secondary' | 'browser' | null>(null)
  const [primaryModel, setPrimaryModelState] = useState<ModelSelection | null>(null)
  const [secondaryModel, setSecondaryModelState] = useState<ModelSelection | null>(null)
  const [browserModel, setBrowserModelState] = useState<ModelSelection | null>(null)
  const [isProviderHovered, setIsProviderHovered] = useState(false)
  const [isModelHovered, setIsModelHovered] = useState(false)
  const [preferencesSelectedIndex, setPreferencesSelectedIndex] = useState(0)
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
  const [tokenEstimate, setTokenEstimate] = useState(0)
  const [isCompacting, setIsCompacting] = useState(false)
  const returnToProviderDetailRef = useRef<string | null>(null)
  const turnStartTimeRef = useRef<number | null>(null)
  const hasAnimatedRef = useRef(skipAnimation)
  const initClientRef = useRef<(() => Promise<AgentClient>) | null>(null)
  const [modelsLoaded, setModelsLoaded] = useState(0)
  const [attachments, setAttachments] = useState<ImageAttachment[]>([])
  const [, setTerminalResizeTick] = useState(0)

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

  const escHintRenderedText = nextEscWillKillAll
    ? 'Press Esc again to kill all forks'
    : nextEscWillClearInput
      ? 'Press Esc again to clear text'
      : nextCtrlCWillExit
        ? 'Press Ctrl-C again to exit'
        : bashMode
          ? 'Esc to exit Bash mode'
          : ''

  // Always reserve width for the longest possible escape hint so that
  // attachments don't reflow when hints appear/disappear.
  const maxEscHintWidth = stringWidth('Press Esc again to kill all forks')

  const terminalWidth = process.stdout.columns ?? 80
  const footerRightGap = contextRenderedText ? 1 : 0
  const footerHorizontalPadding = 4
  const footerSafetyBuffer = 4
  const attachmentsMaxWidth = Math.max(
    0,
    terminalWidth
      - footerHorizontalPadding
      - maxEscHintWidth
      - stringWidth(contextRenderedText)
      - footerRightGap
      - footerSafetyBuffer,
  )

  const addImageAttachment = useCallback(async () => {
    const result = await readClipboardBitmap()
    if (!result) return false

    const scaled = await autoScaleImageAttachmentIfNeeded({
      base64: result.base64,
      mime: result.mime,
      width: result.width,
      height: result.height,
      filename: 'clipboard-' + Date.now() + '.png',
    })

    logger.info(
      {
        wasScaled: scaled.wasScaled,
        originalBytes: scaled.originalBytes,
        finalBytes: scaled.finalBytes,
        width: scaled.width,
        height: scaled.height,
      },
      'Clipboard image attachment processed',
    )

    const extension = scaled.mime === 'image/jpeg' ? '.jpg' : '.png'
    const attachment: ImageAttachment = {
      type: 'image',
      base64: scaled.base64,
      mediaType: scaled.mime as ImageMediaType,
      width: scaled.width,
      height: scaled.height,
      filename: 'clipboard-' + Date.now() + extension,
    }
    setAttachments(prev => [...prev, attachment])
    return true
  }, [])

  const addImageAttachmentFromFilePath = useCallback(async (rawPasteText: string) => {
    const result = await tryReadPastedImageFile(rawPasteText)
    if (!result) return false

    const scaled = await autoScaleImageAttachmentIfNeeded({
      base64: result.base64,
      mime: result.mediaType,
      width: result.width,
      height: result.height,
      filename: result.filename,
    })

    logger.info(
      {
        wasScaled: scaled.wasScaled,
        originalBytes: scaled.originalBytes,
        finalBytes: scaled.finalBytes,
        width: scaled.width,
        height: scaled.height,
      },
      'File-path image attachment processed',
    )

    const parsed = result.filename.includes('.') ? result.filename.split('.') : [result.filename]
    const stem = parsed.slice(0, -1).join('.') || result.filename
    const filename = scaled.mime === 'image/jpeg' ? `${stem}.jpg` : result.filename

    const attachment: ImageAttachment = {
      type: 'image',
      base64: scaled.base64,
      mediaType: scaled.mime as ImageMediaType,
      width: scaled.width,
      height: scaled.height,
      filename,
    }

    setAttachments(prev => [...prev, attachment])
    return true
  }, [])

  const removeAttachment = useCallback((index: number) => {
    setAttachments(prev => prev.filter((_, i) => i !== index))
  }, [])

  useEffect(() => {
    const handleResize = () => {
      setTerminalResizeTick(prev => prev + 1)
    }

    process.stdout.on('resize', handleResize)
    return () => {
      process.stdout.off('resize', handleResize)
    }
  }, [])

  // Browser setup overlay state
  const [showBrowserSetup, setShowBrowserSetup] = useState(false)

  // Setup wizard state
  const [showSetupWizard, setShowSetupWizard] = useState(false)
  const [wizardStep, setWizardStep] = useState<WizardStep>('provider')
  const [wizardProviderSelectedIndex, setWizardProviderSelectedIndex] = useState(0)
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
    getRecentChats().then(setRecentChats)
  }, [])

  useEffect(() => {
    clearLog()
    clearEventLog()
    logger.info({ logFile: getLogPath() }, 'App started')
    if (debugMode) logger.info('Debug mode enabled - press Ctrl+D to toggle debug panel')
    refreshRecentChats()
  }, [refreshRecentChats])

  useEffect(() => {
    if (!providerUiState) return

    setPrimaryModelState(providerUiState.primaryModel)
    setSecondaryModelState(providerUiState.secondaryModel)
    setBrowserModelState(providerUiState.browserModel)

    initTelemetry({ telemetryEnabled: providerUiState.config.telemetry !== false })

    if (!providerUiState.config.setupComplete) {
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

    setModelsLoaded(n => n + 1)

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
        sessionId = await JsonChatPersistence.findLatestSessionId() ?? undefined
      } else if (sessionSelection === null) {
        sessionId = undefined
      } else {
        sessionId = sessionSelection
      }

      const persistence = new JsonChatPersistence(sessionId)
      configureSessionLogging(persistence.getSessionDir())
      const persistenceLayer = Layer.succeed(ChatPersistence, persistence)
      return createCodingAgentClient({
        persistence: persistenceLayer,
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
        logEvent(event)
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

        if (event.type === 'fork_started') {
          forkRoles.set(event.forkId, { role: event.role, startTime: Date.now() })
          trackAgentSpawned({
            agentType: event.role,
            mode: event.mode,
            blocking: event.blocking ?? false,
          })
          sessionTracker.recordAgentSpawned()
        }

        if (event.type === 'fork_completed') {
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

      // Subscribe to fork state (standard projection)
      client.state.forks.subscribe((state) => {
        if (mounted) {
          setForkState(state)
        }
      })


      // Subscribe to artifact state (global projection)
      client.state.artifacts.subscribe((state) => {
        if (mounted) {
          setArtifactState(state)
        }
      })

      // Subscribe to agent registry state (global projection)
      client.state.agentRegistry.subscribe((state) => {
        if (mounted) {
          setAgentRegistryState(state)
        }
      })

      // Subscribe to restore queued messages signal (only for main/root)
      client.on.restoreQueuedMessages(({ forkId, messages }) => {
        // Only restore if this is for the main agent (not a fork)
        if (mounted && forkId === null && messages.length > 0) {
          const restored = messages.join('\n')
          logger.info({ restored, length: restored.length }, 'Restoring queued messages to input')
          setInputValue(prev => {
            logger.info({ prev, next: restored }, 'setInputValue functional update')
            return {
              text: restored,
              cursorPosition: restored.length,
              lastEditDueToNav: false,
              pasteSegments: [],
              selectedPasteSegmentId: null,
            }
          })
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
      c?.dispose()
    }
  }, [sessionSelection])

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

  const hasRunningForks = forkState
    ? Array.from(forkState.forks.values()).some(f => f.status === 'running')
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
    setSystemMessages(prev => [...prev, { id, text: message, timestamp: Date.now() }])
    setTimeout(() => {
      setSystemMessages(prev => prev.filter(m => m.id !== id))
    }, durationMs)
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
    setInputValue({ text: '', cursorPosition: 0, lastEditDueToNav: false, pasteSegments: [], selectedPasteSegmentId: null })
  }, [])

  const handleResumeChat = useCallback((chat: RecentChat) => {
    hasAnimatedRef.current = true
    setShowRecentChatsOverlay(false)
    onResumeSession(chat.id)
  }, [onResumeSession])

  // Navigation for startup widget (active when no messages, overlay closed, and input empty)
  const widgetNavActive = !showRecentChatsOverlay && (display?.messages ?? []).length === 0 && inputValue.text.length === 0 && attachments.length === 0
  const widgetNavigation = useRecentChatsNavigation(
    recentChats ? recentChats.slice(0, 5) : [],
    handleResumeChat,
    widgetNavActive,
  )

  // Navigation for full-screen overlay
  const overlayNavigation = useRecentChatsNavigation(
    recentChats ?? [],
    handleResumeChat,
    showRecentChatsOverlay,
  )

  // Model selection overlay handlers
  const handleModelSelect = useCallback(async (providerId: string, modelId: string) => {
    if (!selectingModelFor || !providerUiState) return

    const selection: ModelSelection = { providerId, modelId }
    const cfg = { ...providerUiState.config }

    if (selectingModelFor === 'primary') {
      setPrimaryModelState(selection)
      cfg.primaryModel = selection
      const auth = await providerRuntime.auth.getAuth(providerId)
      await providerRuntime.state.setSelection('primary', providerId, modelId, auth ?? null, { persist: false })
    } else if (selectingModelFor === 'secondary') {
      setSecondaryModelState(selection)
      cfg.secondaryModel = selection
      const auth = await providerRuntime.auth.getAuth(providerId)
      await providerRuntime.state.setSelection('secondary', providerId, modelId, auth ?? null)
    } else if (selectingModelFor === 'browser') {
      setBrowserModelState(selection)
      cfg.browserModel = selection
      const auth = await providerRuntime.auth.getAuth(providerId)
      await providerRuntime.state.setSelection('browser', providerId, modelId, auth ?? null, { persist: false })
    }

    await providerRuntime.config.saveConfig(cfg)
    await reloadProviderState()
    setSelectingModelFor(null)
  }, [selectingModelFor, providerUiState, providerRuntime, reloadProviderState])

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

  const connectedProviders = useMemo(() => {
    const connectedIds = new Set(detectedProviders.map(d => d.provider.id))
    return PROVIDERS.filter(p => connectedIds.has(p.id))
  }, [detectedProviders, modelsLoaded])

  const modelItems = useMemo(() => {
    const result: Array<{ type: 'model'; providerId: string; providerName: string; modelId: string; modelName: string }> = []
    for (const provider of connectedProviders) {
      if (provider.id === 'local') {
        if (providerUiState?.localProviderConfig.baseUrl && providerUiState?.localProviderConfig.modelId) {
          result.push({
            type: 'model',
            providerId: 'local',
            providerName: provider.name,
            modelId: providerUiState.localProviderConfig.modelId,
            modelName: providerUiState.localProviderConfig.baseUrl,
          })
        }
        continue
      }

      const oauthOnly = provider.oauthOnlyModelIds
      const oauthOnlySet = oauthOnly ? new Set(oauthOnly) : null
      const detected = detectedProviders.find(d => d.provider.id === provider.id)
      const isOAuth = oauthOnlySet ? detected?.auth?.type === 'oauth' : false

      for (const model of provider.models) {
        if (oauthOnlySet && !isOAuth && oauthOnlySet.has(model.id)) continue
        result.push({
          type: 'model',
          providerId: provider.id,
          providerName: provider.name,
          modelId: model.id,
          modelName: model.name,
        })
      }
    }
    return result
  }, [connectedProviders, detectedProviders, providerUiState])

  const filteredModelItems = useMemo(() => {
    if (selectingModelFor !== 'browser') return modelItems
    return modelItems.filter(item => isBrowserCompatible(item.providerId, item.modelId))
  }, [modelItems, selectingModelFor])

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
        const detected = detectedProviders
        const match = detected.find(d => d.provider.id === providerId)
        const isOAuth = match?.auth?.type === 'oauth'
        const defaults = getDefaultModels(providerId, isOAuth)
        const provider = getProvider(providerId)
        const primaryModelId = defaults.primary || provider?.defaultModel || provider?.models[0]?.id || ''
        const secondaryModelId = defaults.secondary || provider?.defaultSecondaryModel || provider?.defaultModel || provider?.models[0]?.id || ''
        if (primaryModelId) setWizardPrimaryModel({ providerId, modelId: primaryModelId })
        if (secondaryModelId) setWizardSecondaryModel({ providerId, modelId: secondaryModelId })
        const browserModelId = defaults.browser || provider?.defaultBrowserModel
        const browserSel = browserModelId
          ? { providerId, modelId: browserModelId }
          : detectBrowserModel(detected, providerId)
        setWizardBrowserModel(browserSel)
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
      const isOAuth = match.auth?.type === 'oauth'
      const defaults = getDefaultModels(providerId, isOAuth)
      const primaryModelId = defaults.primary || provider.defaultModel || provider.models[0]?.id || ''
      const secondaryModelId = defaults.secondary || provider.defaultSecondaryModel || provider.defaultModel || provider.models[0]?.id || ''
      if (primaryModelId) setWizardPrimaryModel({ providerId, modelId: primaryModelId })
      if (secondaryModelId) setWizardSecondaryModel({ providerId, modelId: secondaryModelId })
      const browserModelId = defaults.browser || provider.defaultBrowserModel
      const browserSel = browserModelId
        ? { providerId, modelId: browserModelId }
        : detectBrowserModel(detected, providerId)
      setWizardBrowserModel(browserSel)
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
    const cfg = { ...providerUiState.config }
    cfg.primaryModel = result.primaryModel
    cfg.secondaryModel = result.secondaryModel
    if (result.browserModel) {
      cfg.browserModel = result.browserModel
    }

    setPrimaryModelState(result.primaryModel)
    setSecondaryModelState(result.secondaryModel)
    if (result.browserModel) setBrowserModelState(result.browserModel)

    const primaryAuth = await providerRuntime.auth.getAuth(result.primaryModel.providerId)
    await providerRuntime.state.setSelection('primary', result.primaryModel.providerId, result.primaryModel.modelId, primaryAuth ?? null, { persist: false })
    const secondaryAuth = await providerRuntime.auth.getAuth(result.secondaryModel.providerId)
    await providerRuntime.state.setSelection('secondary', result.secondaryModel.providerId, result.secondaryModel.modelId, secondaryAuth ?? null)
    if (result.browserModel) {
      const browserAuth = await providerRuntime.auth.getAuth(result.browserModel.providerId)
      await providerRuntime.state.setSelection('browser', result.browserModel.providerId, result.browserModel.modelId, browserAuth ?? null, { persist: false })
    }

    if (wizardNeedsChromium) {
      await providerRuntime.config.saveConfig(cfg)
      await reloadProviderState()
      setWizardStep('browser')
    } else {
      cfg.setupComplete = true
      await providerRuntime.config.saveConfig(cfg)
      await reloadProviderState()
      finishWizard()
    }
  }, [authFlow.cancelAll, wizardNeedsChromium, finishWizard, providerUiState, providerRuntime, reloadProviderState])

  const handleWizardBrowserComplete = useCallback(async () => {
    if (!providerUiState) return
    const cfg = { ...providerUiState.config, setupComplete: true }
    await providerRuntime.config.saveConfig(cfg)
    await reloadProviderState()
    finishWizard()
  }, [finishWizard, providerUiState, providerRuntime, reloadProviderState])

  const handleWizardSkip = useCallback(async () => {
    if (!providerUiState) return
    authFlow.cancelAll()
    const cfg = { ...providerUiState.config, setupComplete: true }
    await providerRuntime.config.saveConfig(cfg)
    await reloadProviderState()
    finishWizard()
  }, [authFlow.cancelAll, finishWizard, providerUiState, providerRuntime, reloadProviderState])

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

  // Wizard provider list navigation
  const wizardProviderNavigation = useProviderSelectNavigation(
    WIZARD_PROVIDERS,
    handleWizardProviderSelected,
    showSetupWizard && wizardStep === 'provider',
  )

  // Wizard model confirmation navigation
  const wizardModelNavigation = useSetupWizardNavigation(
    () => {
      if (wizardPrimaryModel && wizardSecondaryModel) {
        handleWizardComplete({ primaryModel: wizardPrimaryModel, secondaryModel: wizardSecondaryModel, browserModel: wizardBrowserModel })
      }
    },
    () => {}, // onChangePrimary — not used yet
    () => {}, // onChangeSecondary — not used yet
    () => {}, // onChangeBrowser — not used yet
    showSetupWizard && wizardStep === 'models',
  )

  const handleProviderSelect = useCallback((providerId: string) => {
    setProviderDetailId(providerId)
    setProviderDetailSelectedIndex(0)
  }, [])


  const handleProviderDisconnect = useCallback(async (providerId: string) => {
    if (!providerUiState) return
    await providerRuntime.auth.removeAuth(providerId)

    const cfg = { ...providerUiState.config }
    let modelChanged = false
    if (primaryModel?.providerId === providerId) {
      setPrimaryModelState(null)
      await providerRuntime.state.clear('primary')
      cfg.primaryModel = null
      modelChanged = true
    }
    if (secondaryModel?.providerId === providerId) {
      setSecondaryModelState(null)
      await providerRuntime.state.clear('secondary')
      cfg.secondaryModel = null
      modelChanged = true
    }
    if (browserModel?.providerId === providerId) {
      setBrowserModelState(null)
      await providerRuntime.state.clear('browser')
      cfg.browserModel = null
      modelChanged = true
    }
    if (modelChanged) {
      await providerRuntime.config.saveConfig(cfg)
    }
    if (providerId === 'local') {
      const localCfg = { ...cfg }
      if (localCfg.providerOptions?.local) {
        delete localCfg.providerOptions.local
        await providerRuntime.config.saveConfig(localCfg)
      }
      await providerRuntime.config.setLocalProviderConfig('', '')
    }

    await reloadProviderState()
    setProviderDetailSelectedIndex(0)
    setProviderRefreshKey(prev => prev + 1)
    showEphemeral(`Disconnected ${getProvider(providerId)?.name ?? providerId}`, theme.warning)
  }, [primaryModel, secondaryModel, browserModel, showEphemeral, theme.warning, providerUiState, providerRuntime, reloadProviderState])

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
    setSelectingModelFor('primary')
  }, [])

  const handleChangeSecondary = useCallback(() => {
    setSelectingModelFor('secondary')
  }, [])

  const handleChangeBrowser = useCallback(() => {
    setSelectingModelFor('browser')
  }, [])

  // Combined model tab keyboard handler — switches between primary/secondary/browser view and model picker
  const modelTabHandleKeyEvent = useCallback((key: KeyEvent): boolean => {
    if (settingsTab !== 'model') return false
    const plain = !key.ctrl && !key.meta && !key.option

    // When in model picker sub-view
    if (selectingModelFor) {
      // 'b' goes back to primary/secondary/browser view
      if (key.name === 'b' && plain) {
        setSelectingModelFor(null)
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
  }, [settingsTab, selectingModelFor, modelNavigation.handleKeyEvent, preferencesSelectedIndex, handleChangePrimary, handleChangeSecondary, handleChangeBrowser])

  const providerNavigation = useProviderSelectNavigation(
    PROVIDERS,
    handleProviderSelect,
    settingsTab === 'provider',
  )

  const authMethodNavigation = useAuthMethodNavigation(
    authFlow.authMethodProvider?.authMethods ?? [],
    (methodIndex: number) => {
      if (authFlow.authMethodProvider) {
        authFlow.startAuthForProvider(authFlow.authMethodProvider, methodIndex)
      }
    },
    authFlow.closeAuthMethodPicker,
    authFlow.showAuthMethodOverlay,
  )

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
      setSelectingModelFor(null)
    }
    setProviderDetailId(null)
    setSettingsTab(tab)
  }, [selectingModelFor])

  const settingsNavigation = useSettingsNavigation(
    settingsTab ?? 'provider',
    handleSettingsTabChange,
    modelTabHandleKeyEvent,
    providerTabHandleKeyEvent,
    settingsTab !== null,
  )

  const openSetup = useCallback(() => {
    setWizardStep('provider')
    setWizardProviderSelectedIndex(0)
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

  const executeSlashCommand = useCallback((commandText: string) => {
    routeSlashCommand(commandText, commandContext)
    setInputValue({ text: '', cursorPosition: 0, lastEditDueToNav: false, pasteSegments: [], selectedPasteSegmentId: null })
  }, [commandContext])

  const slashCommands = useSlashCommands(inputValue.text, executeSlashCommand)

  // Combined key intercept: slash commands + widget navigation
  const handleKeyIntercept = useCallback((key: KeyEvent): boolean => {
    // Block all input during pending approval
    if (pendingApproval) return true
    if (!bashMode && slashCommands.handleKeyIntercept(key)) return true
    if (widgetNavActive && widgetNavigation.handleKeyEvent(key)) return true
    return false
  }, [pendingApproval, bashMode, slashCommands.handleKeyIntercept, widgetNavActive, widgetNavigation.handleKeyEvent])

  const handleSubmit = useCallback((message: string, visibleMessage?: string) => {
    const slashText = visibleMessage ?? message
    // Check for slash commands first (skip in bash mode)
    if (!bashMode && routeSlashCommand(slashText, commandContext)) {
      setInputValue({ text: '', cursorPosition: 0, lastEditDueToNav: false, pasteSegments: [], selectedPasteSegmentId: null })
      return
    }

    // Bash mode: execute terminal command
    if (bashMode) {
      const trimmed = message.trim()
      if (!trimmed) return
      const result = executeBashCommand(trimmed)
      setBashOutputs(prev => [...prev, result])
      setInputValue({ text: '', cursorPosition: 0, lastEditDueToNav: false, pasteSegments: [], selectedPasteSegmentId: null })
      return
    }

    // Guard: all model slots must be configured (banner shows above input)
    if (!primaryModel || !secondaryModel || !browserModel) {
      return
    }

    // Normal message — send to agent
    setSystemMessages([])
    if (ephemeralTimerRef.current) clearTimeout(ephemeralTimerRef.current)
    setEphemeralMessage(null)
    setInputValue({ text: '', cursorPosition: 0, lastEditDueToNav: false, pasteSegments: [], selectedPasteSegmentId: null })

    const currentAttachments = [...attachments]
    const sendMessage = (c: AgentClient) => {
      c.send({ type: 'user_message', forkId: null, content: textParts(message), attachments: currentAttachments, mode: 'text', synthetic: false, taskMode: false })
    }
    setAttachments([])

    if (client) {
      sendMessage(client)
    } else if (initClientRef.current) {
      // Lazy init: create client now, then send message
      const initFn = initClientRef.current
      initClientRef.current = null  // Prevent double init
      initFn().then((newClient) => {
        sendMessage(newClient)
      }).catch((err) => {
        logger.error({ error: err.message, stack: err.stack }, 'Failed to create agent client')
      })
    }
  }, [client, commandContext, bashMode, attachments, primaryModel, secondaryModel, browserModel])

  const handleInputChange = useCallback((value: InputValue) => {
    // Typing '!' as first character enters bash mode
    if (!bashMode && value.text === '!') {
      setBashMode(true)
      setInputValue({ text: '', cursorPosition: 0, lastEditDueToNav: false, pasteSegments: [], selectedPasteSegmentId: null })
      return
    }
    // Reset escape-to-clear hint when user types
    if (nextEscWillClearInput) {
      setNextEscWillClearInput(false)
      if (clearInputTimeoutRef.current) {
        clearTimeout(clearInputTimeoutRef.current)
      }
    }
    setInputValue(value)
  }, [bashMode, nextEscWillClearInput])

  const INLINE_PASTE_PILL_CHAR_LIMIT = 1000

  const handlePaste = useCallback(async (eventText?: string) => {
    logger.debug({ eventText: eventText ? eventText.substring(0, 50) : null }, 'handlePaste called')

    // Try clipboard image first
    const wasClipboardImage = await addImageAttachment()
    if (wasClipboardImage) {
      logger.debug('Added clipboard image attachment')
      return
    }

    const pasteText = eventText || readClipboardText()
    logger.debug({ pasteText: pasteText ? pasteText.substring(0, 50) : null }, 'pasteText from clipboard')
    if (!pasteText) {
      logger.debug('No paste text available, returning')
      return
    }

    const wasImagePath = await addImageAttachmentFromFilePath(pasteText)
    if (wasImagePath) {
      logger.debug('Added image attachment from pasted file path')
      return
    }

    logger.debug({ cursorPosition: inputValue.cursorPosition }, 'Handling text paste with inline pill logic')
    setInputValue(prev => {
      if (pasteText.length > INLINE_PASTE_PILL_CHAR_LIMIT) {
        return insertPasteSegment(prev, pasteText, createId())
      }
      return applyTextEditWithSegments(
        prev,
        prev.cursorPosition,
        prev.cursorPosition,
        pasteText,
      )
    })
  }, [inputValue.cursorPosition, addImageAttachment, addImageAttachmentFromFilePath])

  const handleInterrupt = useCallback(() => {
    if (!client) return
    logger.info({ forkId: null }, 'Sending interrupt event')
    client.send({ type: 'interrupt', forkId: null })
  }, [client])

  const handleInterruptAll = useCallback(() => {
    if (!client) return
    logger.info('Interrupt all: killing all forks')
    // Interrupt root with allKilled flag
    client.send({ type: 'interrupt', forkId: null, allKilled: true })
    // Interrupt every running fork
    if (forkState) {
      for (const [forkId, fork] of forkState.forks) {
        if (fork.status === 'running') {
          client.send({ type: 'interrupt', forkId })
        }
      }
    }
  }, [client, forkState])

  useKeyboard(
    useCallback(
      (key: KeyEvent) => {
        logger.debug({
          name: key.name,
          ctrl: key.ctrl,
          meta: key.meta,
          shift: key.shift,
          option: key.option,
          sequence: key.sequence
        }, 'KEY EVENT')

        const isEscape = key.name === 'escape'
        const isCtrlC = key.ctrl && key.name === 'c' && !key.meta && !key.option
        const isCtrlV = key.ctrl && key.name === 'v' && !key.meta && !key.option
        const isCmdV = key.meta && key.name === 'v' && !key.ctrl && !key.option
        // Tab: toggle task panel + plan mode
        if (key.name === 'tab' && !key.shift && !key.ctrl && !key.meta && !key.option) {
          toggleTaskPanel()
          return
        }

        // Priority 0.5: Setup wizard keyboard handling
        if (showSetupWizard) {
          if (isCtrlC) {
            if ('preventDefault' in key && typeof key.preventDefault === 'function') {
              key.preventDefault()
            }
            // Single-tap exit during wizard — mark setup complete and exit
            if (providerUiState) {
              const cfg = { ...providerUiState.config, setupComplete: true }
              void providerRuntime.config.saveConfig(cfg).then(() => reloadProviderState())
            }
            authFlow.cancelAll()
            process.kill(process.pid, 'SIGINT')
            return
          }
          if (isEscape) {
            if ('preventDefault' in key && typeof key.preventDefault === 'function') {
              key.preventDefault()
            }
            handleWizardSkip()
            return
          }
          // Delegate arrow/enter to the appropriate wizard navigation hook
          if (wizardStep === 'provider') {
            if (wizardProviderNavigation.handleKeyEvent(key)) {
              if ('preventDefault' in key && typeof key.preventDefault === 'function') {
                key.preventDefault()
              }
              return
            }
          } else if (wizardStep === 'models') {
            // B goes back to provider selection
            if (key.name === 'b' && !key.ctrl && !key.meta && !key.option && !key.shift) {
              if ('preventDefault' in key && typeof key.preventDefault === 'function') {
                key.preventDefault()
              }
              handleWizardBack()
              return
            }
            if (wizardModelNavigation.handleKeyEvent(key)) {
              if ('preventDefault' in key && typeof key.preventDefault === 'function') {
                key.preventDefault()
              }
              return
            }
          } else if (wizardStep === 'browser') {
            // B goes back to models step
            if (key.name === 'b' && !key.ctrl && !key.meta && !key.option && !key.shift) {
              if ('preventDefault' in key && typeof key.preventDefault === 'function') {
                key.preventDefault()
              }
              handleWizardBack()
              return
            }
            // Enter triggers install (handled by BrowserSetupOverlay internally)
            if ((key.name === 'return' || key.name === 'enter') && !key.shift && !key.ctrl && !key.meta && !key.option) {
              // Let it pass through to the overlay
              return
            }
          }
          // Block all other keys when wizard is showing
          return
        }

        // Priority 1: Clear input with Ctrl+C when there's text
        if (isCtrlC && inputValue.text.trim().length > 0) {
          logger.debug('Ctrl+C clearing input')
          if ('preventDefault' in key && typeof key.preventDefault === 'function') {
            key.preventDefault()
          }
          setInputValue({ text: '', cursorPosition: 0, lastEditDueToNav: false, pasteSegments: [], selectedPasteSegmentId: null })
          return
        }

        // Escape closes fork overlay first (before any interrupt logic)
        if (isEscape && expandedForkStack.length > 0) {
          if ('preventDefault' in key && typeof key.preventDefault === 'function') {
            key.preventDefault()
          }
          popForkOverlay()
          return
        }

        // Priority 2a: Second Esc while "kill all" prompt is active — kill all forks
        if (isEscape && nextEscWillKillAll) {
          if ('preventDefault' in key && typeof key.preventDefault === 'function') {
            key.preventDefault()
          }
          logger.debug('Second Esc: killing all forks')
          handleInterruptAll()
          setNextEscWillKillAll(false)
          if (killAllTimeoutRef.current) {
            clearTimeout(killAllTimeoutRef.current)
          }
          return
        }

        // Priority 2b: Esc while forks are running — interrupt viewed fork (if streaming) + prompt to kill all
        if (isEscape && hasRunningForks) {
          if ('preventDefault' in key && typeof key.preventDefault === 'function') {
            key.preventDefault()
          }
          // Interrupt viewed fork if it's streaming
          if (display?.status === 'streaming') {
            logger.debug('Interrupting stream (forks running)')
            handleInterrupt()
          }
          // Show "press Esc again to kill all"
          setNextEscWillKillAll(true)
          if (killAllTimeoutRef.current) {
            clearTimeout(killAllTimeoutRef.current)
          }
          killAllTimeoutRef.current = setTimeout(() => {
            setNextEscWillKillAll(false)
          }, 2000)
          return
        }

        // Priority 2b2: Double-tap Escape to clear input text
        if (isEscape && inputValue.text.length > 0) {
          if ('preventDefault' in key && typeof key.preventDefault === 'function') {
            key.preventDefault()
          }
          if (nextEscWillClearInput) {
            // Second tap — clear the input
            setInputValue({ text: '', cursorPosition: 0, lastEditDueToNav: false, pasteSegments: [], selectedPasteSegmentId: null })
            setNextEscWillClearInput(false)
            if (clearInputTimeoutRef.current) {
              clearTimeout(clearInputTimeoutRef.current)
            }
          } else {
            // First tap — show hint
            setNextEscWillClearInput(true)
            if (clearInputTimeoutRef.current) {
              clearTimeout(clearInputTimeoutRef.current)
            }
            clearInputTimeoutRef.current = setTimeout(() => {
              setNextEscWillClearInput(false)
            }, 2000)
          }
          return
        }

        // Priority 2c: Interrupt streaming with Escape or Ctrl+C (no forks running)
        if ((isEscape || isCtrlC) && display?.status === 'streaming') {
          if ('preventDefault' in key && typeof key.preventDefault === 'function') {
            key.preventDefault()
          }
          logger.debug({ trigger: isEscape ? 'Escape' : 'Ctrl+C' }, 'Interrupting stream')
          handleInterrupt()
          return
        }

        // Priority 3: Two-tap Ctrl+C to exit
        if (isCtrlC) {
          if ('preventDefault' in key && typeof key.preventDefault === 'function') {
            key.preventDefault()
          }

          if (nextCtrlCWillExit) {
            // Second tap - send SIGINT to trigger proper cleanup handlers
            process.kill(process.pid, 'SIGINT')
          } else {
            // First tap - show warning
            setNextCtrlCWillExit(true)

            // Clear any existing timeout
            if (exitTimeoutRef.current) {
              clearTimeout(exitTimeoutRef.current)
            }

            // Reset after 2 seconds
            exitTimeoutRef.current = setTimeout(() => {
              setNextCtrlCWillExit(false)
            }, 2000)
          }
          return
        }

        // Priority 4.5: Ctrl+D toggles debug panel (only when debug mode enabled)
        const isCtrlD = key.ctrl && key.name === 'd' && !key.meta && !key.option
        if (isCtrlD && debugMode) {
          if ('preventDefault' in key && typeof key.preventDefault === 'function') {
            key.preventDefault()
          }
          setDebugPanelVisible(prev => !prev)
          return
        }

        // Priority 4: Ctrl+R toggles recent chats overlay
        const isCtrlR = key.ctrl && key.name === 'r' && !key.meta && !key.option
        if (isCtrlR) {
          if ('preventDefault' in key && typeof key.preventDefault === 'function') {
            key.preventDefault()
          }
          hasAnimatedRef.current = true
          setShowRecentChatsOverlay(prev => !prev)
          return
        }

        // Priority 5: Escape closes overlays (when not streaming)
        if (isEscape && showBrowserSetup) {
          if ('preventDefault' in key && typeof key.preventDefault === 'function') {
            key.preventDefault()
          }
          setShowBrowserSetup(false)
          return
        }
        if (isEscape && showRecentChatsOverlay) {
          if ('preventDefault' in key && typeof key.preventDefault === 'function') {
            key.preventDefault()
          }
          hasAnimatedRef.current = true
          setShowRecentChatsOverlay(false)
          return
        }
        if (isEscape && settingsTab !== null) {
          if ('preventDefault' in key && typeof key.preventDefault === 'function') {
            key.preventDefault()
          }
          setSettingsTab(null)
          setSelectingModelFor(null)
          return
        }
        if (isEscape && authFlow.oauthState) {
          if ('preventDefault' in key && typeof key.preventDefault === 'function') {
            key.preventDefault()
          }
          authFlow.handleOAuthCancel()
          return
        }
        if (isEscape && authFlow.showAuthMethodOverlay) {
          if ('preventDefault' in key && typeof key.preventDefault === 'function') {
            key.preventDefault()
          }
          authFlow.closeAuthMethodPicker()
          return
        }
        // Priority 5.4: Escape closes artifact panel
        if (isEscape && selectedArtifact) {
          if ('preventDefault' in key && typeof key.preventDefault === 'function') {
            key.preventDefault()
          }
          setSelectedArtifact(null)
          return
        }
        // Priority 5.5: Escape exits bash mode
        if (isEscape && bashMode) {
          if ('preventDefault' in key && typeof key.preventDefault === 'function') {
            key.preventDefault()
          }
          exitBashMode()
          return
        }

        // Priority 5.7: Approval keyboard shortcuts
        if (pendingApproval) {
          const isY = key.name === 'a' && !key.ctrl && !key.meta && !key.option
          const isN = key.name === 'd' && !key.ctrl && !key.meta && !key.option
          const isEnter = key.name === 'return' && !key.ctrl && !key.meta && !key.option

          if (isY || isEnter) {
            if ('preventDefault' in key && typeof key.preventDefault === 'function') {
              key.preventDefault()
            }
            handleApprove()
            return
          }
          if (isN || isEscape) {
            if ('preventDefault' in key && typeof key.preventDefault === 'function') {
              key.preventDefault()
            }
            handleReject()
            return
          }
        }

        // Priority 6: Arrow/Enter navigation for overlays
        if (showRecentChatsOverlay) {
          if (overlayNavigation.handleKeyEvent(key)) {
            if ('preventDefault' in key && typeof key.preventDefault === 'function') {
              key.preventDefault()
            }
            return
          }
        }
        if (settingsTab !== null) {
          if (settingsNavigation.handleKeyEvent(key)) {
            if ('preventDefault' in key && typeof key.preventDefault === 'function') {
              key.preventDefault()
            }
            return
          }
        }
        if (authFlow.showAuthMethodOverlay) {
          if (authMethodNavigation.handleKeyEvent(key)) {
            if ('preventDefault' in key && typeof key.preventDefault === 'function') {
              key.preventDefault()
            }
            return
          }
        }
        if (authFlow.oauthState || authFlow.showLocalSetup || authFlow.apiKeySetup) {
          // Block all keys when OAuth/local/API key setup overlay is active (overlay handles its own keyboard)
          return
        }

        // Priority 7: Paste
        if (isCtrlV || isCmdV) {
          logger.debug({ shortcut: isCtrlV ? 'Ctrl+V' : 'CMD+V' }, 'PASTE SHORTCUT DETECTED')
          if ('preventDefault' in key && typeof key.preventDefault === 'function') {
            key.preventDefault()
          }
          handlePaste()
          return
        }
      },
      [handlePaste, handleInterrupt, handleInterruptAll, inputValue.text, display?.status, nextCtrlCWillExit, nextEscWillKillAll, nextEscWillClearInput,
       showRecentChatsOverlay, overlayNavigation.handleKeyEvent,
       settingsTab, settingsNavigation.handleKeyEvent, selectingModelFor,
       authFlow.showAuthMethodOverlay, authMethodNavigation.handleKeyEvent, authFlow.closeAuthMethodPicker,
       authFlow.oauthState, authFlow.handleOAuthCancel, authFlow.showLocalSetup, authFlow.apiKeySetup,
       bashMode, exitBashMode, debugMode, hasRunningForks, toggleTaskPanel, expandedForkId,
       pendingApproval, handleApprove, handleReject,
       showSetupWizard, wizardStep, handleWizardSkip, handleWizardBack, wizardProviderNavigation.handleKeyEvent, wizardModelNavigation.handleKeyEvent, selectedArtifact,
       showBrowserSetup]
    )
  )

  // Ensure input is focused for paste events to work reliably
  const multilineInputRef = useRef<MultilineInputHandle | null>(null)
  useEffect(() => {
    // Focus input when not showing overlays
    if (!showSetupWizard && !showBrowserSetup && !showRecentChatsOverlay && settingsTab === null && !authFlow.showAuthMethodOverlay && !authFlow.oauthState && !authFlow.showLocalSetup && !authFlow.apiKeySetup && !bashMode && expandedForkId === null) {
      multilineInputRef.current?.focus()
    }
  }, [showSetupWizard, showBrowserSetup, showRecentChatsOverlay, settingsTab, authFlow.showAuthMethodOverlay, authFlow.oauthState, authFlow.showLocalSetup, authFlow.apiKeySetup, bashMode, expandedForkId])

  const handleInputSubmit = useCallback(() => {
    // Normal message submission
    if (inputValue.text.trim() || attachments.length > 0) {
      handleSubmit(reconstituteInputText(inputValue), inputValue.text)
    }
  }, [inputValue, handleSubmit, attachments.length])

  const selectedArtifactContent = useMemo(() => {
    if (!selectedArtifact || !artifactState) return null
    const artifact = artifactState.artifacts.get(selectedArtifact.name)
    return artifact?.content ?? null
  }, [selectedArtifact, artifactState])

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

  if (!display) {
    return (
      <SessionLoadingView
        sessionSelection={sessionSelection}
        recentChats={recentChats}
      />
    )
  }

  if (showSetupWizard && wizardStep === 'browser') {
    return (
      <box style={{ flexDirection: 'column', height: '100%' }}>
        <BrowserSetupOverlay
          onClose={() => handleWizardBrowserComplete()}
          onResult={() => handleWizardBrowserComplete()}
          wizardMode={{
            stepLabel: `Browser (${wizardTotalSteps} of ${wizardTotalSteps})`,
            subtitle: 'The browser agent requires Chromium to control web pages.',
            onSkip: handleWizardSkip,
            onBack: handleWizardBack,
          }}
        />
      </box>
    )
  }

  if (showSetupWizard && !authFlow.oauthState && !authFlow.apiKeySetup && !authFlow.showLocalSetup && !authFlow.showAuthMethodOverlay) {
    return (
      <box style={{ flexDirection: 'column', height: '100%' }}>
        <SetupWizardOverlay
          step={wizardStep}
          allProviders={WIZARD_PROVIDERS}
          detectedProviders={detectedProviders}
          primaryModel={wizardPrimaryModel}
          secondaryModel={wizardSecondaryModel}
          browserModel={wizardBrowserModel}
          connectedProviderName={wizardConnectedProvider}
          totalSteps={wizardTotalSteps}
          onProviderSelected={handleWizardProviderSelected}
          onComplete={handleWizardComplete}
          onBack={handleWizardBack}
          onSkip={handleWizardSkip}
          providerSelectedIndex={wizardProviderNavigation.selectedIndex}
          onProviderHoverIndex={wizardProviderNavigation.setSelectedIndex}
          modelNavSelectedIndex={wizardModelNavigation.selectedIndex}
          onModelNavHoverIndex={wizardModelNavigation.setSelectedIndex}
        />
      </box>
    )
  }

  if (showRecentChatsOverlay) {
    return (
      <box style={{ flexDirection: 'column', height: '100%' }}>
        <RecentChatsOverlay
          chats={recentChats ?? []}
          selectedIndex={overlayNavigation.selectedIndex}
          onSelect={handleResumeChat}
          onHoverIndex={overlayNavigation.setSelectedIndex}
          onClose={() => setShowRecentChatsOverlay(false)}
        />
      </box>
    )
  }

  if (expandedForkId && client) {
    const fork = forkState?.forks.get(expandedForkId)
    const initialPrompt = Array.from(agentRegistryState?.agents.values() ?? []).find(
      (entry) => entry.forkId === expandedForkId
    )?.message ?? null
    return (
      <box style={{ flexDirection: 'column', height: '100%' }}>
        <ForkDetailOverlay
          forkId={expandedForkId}
          forkName={fork?.name ?? 'Agent'}
          forkRole={fork?.role ?? 'agent'}
          initialPrompt={initialPrompt}
          onClose={popForkOverlay}
          onForkExpand={pushForkOverlay}
          subscribeForkDisplay={(fId, cb) => client.state.display.subscribeFork(fId, cb)}
        />
      </box>
    )
  }

  if (showBrowserSetup) {
    return (
      <box style={{ flexDirection: 'column', height: '100%' }}>
        <BrowserSetupOverlay
          onClose={() => setShowBrowserSetup(false)}
          onResult={() => setShowBrowserSetup(false)}
        />
      </box>
    )
  }

  if (settingsTab !== null) {
    return (
      <box style={{ flexDirection: 'column', height: '100%' }}>
        <SettingsOverlay
          activeTab={settingsTab}
          onTabChange={handleSettingsTabChange}
          onClose={() => { setSettingsTab(null); setSelectingModelFor(null); setProviderDetailId(null) }}
          modelProviders={connectedProviders}
          modelItems={modelNavigation.items}
          modelSelectedIndex={modelNavigation.selectedIndex}
          onModelSelect={handleModelSelect}
          onModelHoverIndex={modelNavigation.setSelectedIndex}
          allProviders={PROVIDERS}
          detectedProviders={detectedProviders}
          providerSelectedIndex={providerNavigation.selectedIndex}
          onProviderSelect={handleProviderSelect}
          onProviderHoverIndex={providerNavigation.setSelectedIndex}
          providerDetailStatus={providerDetailStatus}
          providerDetailActions={providerDetailActions}
          providerDetailSelectedIndex={providerDetailSelectedIndex}
          onProviderDetailAction={handleProviderDetailAction}
          onProviderDetailHoverIndex={setProviderDetailSelectedIndex}
          primaryModel={primaryModel}
          secondaryModel={secondaryModel}
          browserModel={browserModel}
          selectingModelFor={selectingModelFor}
          onChangePrimary={handleChangePrimary}
          onChangeSecondary={handleChangeSecondary}
          onChangeBrowser={handleChangeBrowser}
          modelPrefsSelectedIndex={preferencesSelectedIndex}
          onModelPrefsHoverIndex={setPreferencesSelectedIndex}
          localProviderConfig={providerUiState?.localProviderConfig
            ? {
                baseUrl: providerUiState.localProviderConfig.baseUrl ?? undefined,
                modelId: providerUiState.localProviderConfig.modelId ?? undefined,
              }
            : null}
          localProviderAuth={(() => {
            const localDetected = detectedProviders.find((d) => d.provider.id === 'local')
            return localDetected?.auth?.type === 'api' ? localDetected.auth : null
          })()}
        />
      </box>
    )
  }

  if (authFlow.showAuthMethodOverlay && authFlow.authMethodProvider) {
    return (
      <box style={{ flexDirection: 'column', height: '100%' }}>
        <AuthMethodOverlay
          providerName={authFlow.authMethodProvider.name}
          methods={authFlow.authMethodProvider.authMethods}
          selectedIndex={authMethodNavigation.selectedIndex}
          onSelect={(methodIndex) => authFlow.startAuthForProvider(authFlow.authMethodProvider!, methodIndex)}
          onHoverIndex={authMethodNavigation.setSelectedIndex}
          onBack={authFlow.closeAuthMethodPicker}
          wizardMode={showSetupWizard ? {
            stepLabel: `Providers (1 of ${wizardTotalSteps})`,
            subtitle: `Connect to ${authFlow.authMethodProvider.name} to get started.`,
            onSkip: handleWizardSkip,
            onBack: authFlow.closeAuthMethodPicker,
          } : undefined}
        />
      </box>
    )
  }

  if (authFlow.showLocalSetup) {
    const localConfig = providerUiState?.localProviderConfig
    return (
      <box style={{ flexDirection: 'column', height: '100%' }}>
        <LocalProviderOverlay
          initialConfig={{ url: localConfig?.baseUrl ?? '', modelId: localConfig?.modelId ?? '' }}
          onSubmit={authFlow.handleLocalSetupSubmit}
          onCancel={authFlow.handleLocalSetupCancel}
          wizardMode={showSetupWizard ? {
            stepLabel: `Providers (1 of ${wizardTotalSteps})`,
            subtitle: 'Configure your local provider to get started.',
            onSkip: handleWizardSkip,
            onBack: authFlow.handleLocalSetupCancel,
          } : undefined}
        />
      </box>
    )
  }

  if (authFlow.apiKeySetup) {
    return (
      <box style={{ flexDirection: 'column', height: '100%' }}>
        <ApiKeyOverlay
          providerName={authFlow.apiKeySetup.provider.name}
          envKeyHint={authFlow.apiKeySetup.envKeyHint}
          initialKey={authFlow.apiKeySetup.existingKey}
          onSubmit={authFlow.handleApiKeySubmit}
          onCancel={authFlow.handleApiKeyCancel}
          wizardMode={showSetupWizard ? {
            stepLabel: `Providers (1 of ${wizardTotalSteps})`,
            subtitle: `Connect to ${authFlow.apiKeySetup.provider.name} to get started.`,
            onSkip: handleWizardSkip,
            onBack: authFlow.handleApiKeyCancel,
          } : undefined}
        />
      </box>
    )
  }

  if (authFlow.oauthState) {
    return (
      <box style={{ flexDirection: 'column', height: '100%' }}>
        <OAuthOverlay
          providerName={authFlow.oauthState.provider.name}
          mode={authFlow.oauthState.mode}
          url={authFlow.oauthState.url}
          onSubmitCode={authFlow.handleOAuthCodeSubmit}
          codeError={authFlow.oauthState.codeError}
          isSubmitting={authFlow.oauthState.isSubmitting}
          userCode={authFlow.oauthState.userCode}
          onCancel={authFlow.handleOAuthCancel}
          onCopyUrl={authFlow.handleOAuthCopyUrl}
          onCopyCode={authFlow.handleOAuthCopyCode}
          wizardMode={showSetupWizard ? {
            stepLabel: `Providers (1 of ${wizardTotalSteps})`,
            subtitle: `Connect to ${authFlow.oauthState.provider.name} to get started.`,
            onSkip: handleWizardSkip,
            onBack: authFlow.handleOAuthCancel,
          } : undefined}
        />
      </box>
    )
  }

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

                    inputHasText={!!inputValue.text.trim()}
                    onArtifactClick={handleArtifactClick}
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
    </scrollbox>
  )

  const debugVisible = debugMode && debugPanelVisible

  return (
    <ArtifactProvider value={artifactState}>
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
          {chatScrollbox}
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


        {(!primaryModel || !secondaryModel || !browserModel) && (
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

        <box style={{ paddingLeft: 1, paddingRight: 1, flexShrink: 0 }}>
          <box
            style={{
              borderStyle: 'single',
              border: ['left'],
              borderColor: bashMode ? orange[400] : modeColor,
              customBorderChars: { ...BOX_CHARS, vertical: '┃' },
            }}
          >
            <box
              style={{
                backgroundColor: theme.inputBg,
                paddingTop: 1,
                paddingLeft: 1,
                paddingRight: 2,
                flexDirection: 'column',
                flexGrow: 1,
              }}
            >
              {!bashMode && slashCommands.isSlashMenuOpen && (
                <SlashCommandMenu
                  commands={slashCommands.filteredCommands}
                  selectedIndex={slashCommands.selectedIndex}
                  onSelect={(cmd) => executeSlashCommand(`/${cmd.id}`)}
                />
              )}
              <box
                style={{
                  flexDirection: 'column',
                }}
              >
                <box
                  style={{
                    flexDirection: 'row',
                    alignItems: 'center',
                    width: '100%',
                  }}
                >
                  <box style={{ flexGrow: 1, minWidth: 0 }}>
                    <MultilineInput
                      ref={multilineInputRef}
                      value={inputValue.text}
                      cursorPosition={inputValue.cursorPosition}
                      pasteSegments={inputValue.pasteSegments}
                      selectedPasteSegmentId={inputValue.selectedPasteSegmentId}
                      onChange={handleInputChange}
                      onSubmit={handleInputSubmit}
                      onPaste={handlePaste}
                      onKeyIntercept={handleKeyIntercept}
                      focused={!pendingApproval}
                      highlightColor={bashMode ? orange[400] : undefined}
                      placeholder={
                        pendingApproval
                          ? 'Approve or reject the pending action...'
                          : bashMode
                            ? 'Enter a command...'
                            : display.status === 'streaming'
                              ? 'Type to queue a message...'
                              : 'Type a message...'
                      }
                      maxHeight={10}
                      minHeight={1}
                    />

                  </box>
                </box>
                <box style={{ flexDirection: 'row', justifyContent: 'flex-end' }}>
                  {bashMode ? (
                    <text style={{ fg: orange[400] }} attributes={TextAttributes.BOLD}>Bash Mode</text>
                  ) : (() => {
                    const summary = primaryModel ? {
                      provider: getProvider(primaryModel.providerId)?.name ?? primaryModel.providerId,
                      model: modelItems.find((item) => item.providerId === primaryModel.providerId && item.modelId === primaryModel.modelId)?.modelName ?? primaryModel.modelId,
                    } : null
                    return (
                      <>
                        <Button
                          onClick={() => openSettings('provider')}
                          onMouseOver={() => setIsProviderHovered(true)}
                          onMouseOut={() => setIsProviderHovered(false)}
                        >
                          <text style={{ fg: isProviderHovered ? theme.primary : theme.muted }}>{summary?.provider ?? 'No provider'}</text>
                        </Button>
                        <text style={{ fg: theme.muted }}> {'\u00b7'} </text>
                        <Button
                          onClick={() => openSettings('model')}
                          onMouseOver={() => setIsModelHovered(true)}
                          onMouseOut={() => setIsModelHovered(false)}
                        >
                          <text style={{ fg: isModelHovered ? theme.primary : theme.foreground }}>{summary?.model ?? 'No model'}</text>
                        </Button>
                      </>
                    )
                  })()}
                </box>
              </box>
            </box>
          </box>
          <box
            style={{
              height: 1,
              borderStyle: 'single',
              border: ['left'],
              borderColor: bashMode ? orange[400] : modeColor,
              customBorderChars: {
                topLeft: '', bottomLeft: '', topRight: '', bottomRight: '',
                horizontal: ' ', vertical: '╹',
                topT: '', bottomT: '', leftT: '', rightT: '', cross: '',
              },
            }}
          >
            <box
              style={{
                height: 1,
                borderStyle: 'single',
                border: ['bottom'],
                borderColor: theme.inputBg,
                customBorderChars: {
                  topLeft: '', bottomLeft: '', topRight: '', bottomRight: '',
                  horizontal: '▀', vertical: ' ',
                  topT: '', bottomT: '', leftT: '', rightT: '', cross: '',
                },
              }}
            />
          </box>
        </box>
        <box style={{ paddingLeft: 2, paddingRight: 2, flexShrink: 0, height: 1, minHeight: 1, maxHeight: 1, flexDirection: 'row', justifyContent: 'space-between' }}>
          <box style={{ flexDirection: 'row', alignItems: 'center', gap: 1 }}>
            {attachments.length > 0 ? (
              <AttachmentsBar attachments={attachments} onRemove={removeAttachment} maxWidth={attachmentsMaxWidth} />
            ) : nextEscWillKillAll ? (
              <text style={{ fg: theme.secondary }}>Press Esc again to kill all forks</text>
            ) : nextEscWillClearInput ? (
              <text style={{ fg: theme.secondary }}>Press Esc again to clear text</text>
            ) : nextCtrlCWillExit ? (
              <text style={{ fg: theme.secondary }}>Press Ctrl-C again to exit</text>
            ) : bashMode ? (
              <text style={{ fg: theme.muted }}><span attributes={TextAttributes.BOLD}>Esc</span> to exit Bash mode</text>
            ) : null}
          </box>
          <box style={{ flexDirection: 'row', alignItems: 'center', gap: 1 }}>
            {attachments.length > 0 && (nextEscWillKillAll ? (
              <text style={{ fg: theme.secondary }}>Press Esc again to kill all forks</text>
            ) : nextEscWillClearInput ? (
              <text style={{ fg: theme.secondary }}>Press Esc again to clear text</text>
            ) : nextCtrlCWillExit ? (
              <text style={{ fg: theme.secondary }}>Press Ctrl-C again to exit</text>
            ) : bashMode ? (
              <text style={{ fg: theme.muted }}><span attributes={TextAttributes.BOLD}>Esc</span> to exit Bash mode</text>
            ) : null)}
            {tokenEstimate > 0 && (
              <ContextUsageBar
                tokenEstimate={tokenEstimate}
                hardCap={contextHardCap ?? tokenEstimate}
                isCompacting={isCompacting}
              />
            )}
          </box>
        </box>
      </box>

      {selectedArtifact && selectedArtifactContent !== null && (
        <box style={{ width: '45%', flexShrink: 0, paddingRight: 1, paddingBottom: 1 }}>
          <ArtifactReaderPanel
            artifactName={selectedArtifact.name}
            content={selectedArtifactContent}
            scrollToSection={selectedArtifact.section}
            onClose={() => setSelectedArtifact(null)}
            onOpenArtifact={handleArtifactClick}
          />
        </box>
      )}

    </box>
    </ArtifactProvider>
  )
}
