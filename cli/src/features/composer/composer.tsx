import { TextAttributes, type KeyEvent } from '@opentui/core'
import { useCallback, useRef, useState, useMemo } from 'react'
import stringWidth from 'string-width'
import { Effect } from 'effect'
import { Atom, useAtomMount, useAtomSet, useAtomValue as useAtomValueClientCommon } from '@effect-atom/atom-react'
import type { RawImageAttachment, RawMentionOccurrence } from '@magnitudedev/sdk'
import { filenameWithImageExtension, useAgentClient, mentionOccurrenceFromInputSegment, imageMediaTypeFromMime } from '@magnitudedev/client-common'
import { createId } from '@magnitudedev/generate-id'
import { orange } from '../../utils/theme'
import { Button } from '../../components/button'
import { ChatSurfaceKeyboard } from './chat-surface-keyboard'
import { FileMentionMenu } from './mention-menu'
import { SlashCommandMenu } from './slash-menu'
import { MultilineInput, type MultilineInputHandle } from './multiline-input'
import { AttachmentsBar } from './attachment-bar'
import { AutopilotIndicator } from './autopilot-indicator'
import { deriveLocalInferenceFooterView } from '../local-inference/footer-status'
import { useFileMentions, type MentionSearchClient } from '@magnitudedev/client-common'
import { useSlashCommands } from '@magnitudedev/client-common'
import { readClipboardBitmap, readClipboardText } from '../../utils/clipboard'
import { extractPastedPathCandidates, tryReadPastedImageFileCandidate, type ReadPastedImageFileParams } from '../../utils/pasted-image-path'
import { autoScaleImageAttachmentIfNeeded } from '../../utils/image-scaling'
import {
  applyTextEditWithPastesAndMentions,
  insertMentionSegment,
  reconstituteInputTextWithMentions,
} from '@magnitudedev/client-common'
import { resolvePasteIntent, resolvePasteOutcomeFromApplyResult } from '@magnitudedev/client-common'
import { applyPasteIntent } from '@magnitudedev/client-common'
import { composerTextAtom, composerAttachmentsAtom, composerHistoryIndexAtom, composerHasContentAtom } from '@magnitudedev/client-common'
import type { InputValue } from '@magnitudedev/client-common'
import type { ComposerProps } from './types'
import { shouldHandleSlashCommandInTab } from '@magnitudedev/client-common'
import { allowProviderMessageSend } from './provider-send-guard'
import { ContextUsage, contextUsageWidth } from './context-usage'
import { ResidencyIndicator } from './residency-indicator'
import { BOX_CHARS } from '../../utils/ui-constants'

const displayWorkingDirectory = (cwd: string): string => {
  const home = process.env.HOME
  return home && (cwd === home || cwd.startsWith(home + '/'))
    ? `~${cwd.slice(home.length)}`
    : cwd
}

export type PendingImageAttachment = RawImageAttachment

const EMPTY_INPUT: InputValue = {
  text: '',
  cursorPosition: 0,
  lastEditDueToNav: false,
  pasteSegments: [],
  mentionSegments: [],
  selectedPasteSegmentId: null,
  selectedMentionSegmentId: null,
}

const INLINE_PASTE_PILL_CHAR_LIMIT = 1000
const MAX_HISTORY = 200
export const COMPOSER_BORDER_CHARS = { ...BOX_CHARS, vertical: '┃' } as const

export async function handleChatControllerPaste(args: {
  eventText?: string
  addClipboardImage: () => Promise<boolean>
  addImageFromFilePath: (rawPasteText: string) => Promise<boolean>
  setInputValue: (updater: (prev: InputValue) => InputValue) => void
}): Promise<{ didInsert: boolean }> {
  const intent = await resolvePasteIntent({
    eventText: args.eventText,
    readClipboardText,
    tryAddClipboardImage: args.addClipboardImage,
    tryAddImageFromFilePath: args.addImageFromFilePath,
    inlinePastePillCharLimit: INLINE_PASTE_PILL_CHAR_LIMIT,
  })

  const applyResult = applyPasteIntent({
    intent,
    setInputValue: args.setInputValue,
  })

  return {
    didInsert: resolvePasteOutcomeFromApplyResult(applyResult),
  }
}

type ReadPastedImageCandidate = (
  candidate: string,
  params: ReadPastedImageFileParams,
) => Promise<{
  path: string
  filename: string
  base64: string
  mediaType: string
  width: number
  height: number
} | null>

type ScaleImageAttachment = (args: {
  base64: string
  mime: string
  width: number
  height: number
  filename: string
}) => Promise<{
  base64: string
  mime: string
  width: number
  height: number
}>

export async function addImageAttachmentsFromPastedText(args: {
  rawPasteText: string
  appendAttachments: (attachments: PendingImageAttachment[]) => void
  readPastedImageParams: ReadPastedImageFileParams
  extractCandidates?: (rawPasteText: string) => string[]
  readCandidate?: ReadPastedImageCandidate
  scaleAttachment?: ScaleImageAttachment
}): Promise<boolean> {
  const extractCandidates = args.extractCandidates ?? extractPastedPathCandidates
  const readCandidate = args.readCandidate ?? tryReadPastedImageFileCandidate
  const scaleAttachment = args.scaleAttachment ?? autoScaleImageAttachmentIfNeeded

  const candidates = extractCandidates(args.rawPasteText)
  if (candidates.length === 0) return false

  const newAttachments: PendingImageAttachment[] = []

  for (const candidate of candidates) {
    const result = await readCandidate(candidate, args.readPastedImageParams)
    if (!result) continue

    const scaled = await scaleAttachment({
      base64: result.base64,
      mime: result.mediaType,
      width: result.width,
      height: result.height,
      filename: result.filename,
    })
    const mediaType = imageMediaTypeFromMime(scaled.mime)
    if (!mediaType) continue

    newAttachments.push({
      type: 'raw_image_file',
      data: scaled.base64,
      filename: filenameWithImageExtension(result.filename, mediaType),
      mediaType,
      width: scaled.width,
      height: scaled.height,
    })
  }

  if (newAttachments.length === 0) return false

  args.appendAttachments(newAttachments)
  return true
}

export function Composer(props: ComposerProps) {
  const {
    sessionId,
    cwd,
    clientWorkingDirectory,
    status,
    hasRunningForks,
    bashMode,
    modelsConfigured,
    modelSummary,
    localInferenceState,
    selectedProviderId,
    selectedSlotId,
    tokenUsage,
    contextHardCap,
    isCompacting,
    displayMode,
    theme,
    modeColor,
    chatColumnWidth,
    attachmentsMaxWidth,
    composerCanFocus,
    widgetNavActive,
    isWorkerView,
    enableAutopilot,
    autopilotEnabled,
    autopilotGenerating,
    submitUserMessage,
    runSlashCommand,
    executeBash,
    clearSystemBanners,
    interruptFork,
    interruptAll,
    openSettings,
    openHardware,
    thinkingOptions,
    applyThinking,
    handleWidgetKeyEvent,
    enterBashMode,
    exitBashMode,
    showToast,
    toggleAutopilot,
    displayMessages,
    selectedForkId,
    isBlockingOverlayActive,
    selectedFileOpen,
    onCloseFilePanel,
  } = props

  const atomClient = useAgentClient()
  const resolvePathMutation = useAtomSet(atomClient.mutation('ResolvePath'), { mode: 'promise' })
  const readFileMutation = useAtomSet(atomClient.mutation('ReadFile'), { mode: 'promise' })
  const searchMentionsMutation = useAtomSet(atomClient.mutation('SearchMentions'), { mode: 'promise' })

  const composerText = useAtomValueClientCommon(composerTextAtom)
  const setComposerText = useAtomSet(composerTextAtom)
  const setComposerAttachments = useAtomSet(composerAttachmentsAtom)
  const composerHistoryIndex = useAtomValueClientCommon(composerHistoryIndexAtom)
  const setComposerHistoryIndex = useAtomSet(composerHistoryIndexAtom)
  const setComposerHasContent = useAtomSet(composerHasContentAtom)
  // historyIndex: null = not navigating history, number = current index
  const historyIndex = composerHistoryIndex === -1 ? null : composerHistoryIndex

  // Mention search client — uses mutation setter
  const mentionClient = useMemo<MentionSearchClient>(() => ({
    searchMentions(payload) {
      return searchMentionsMutation({
        payload: {
          cwd: payload.cwd,
          query: payload.query,
          ...(payload.limit !== undefined ? { limit: payload.limit } : {}),
          ...(payload.visibleLimit !== undefined ? { visibleLimit: payload.visibleLimit } : {}),
          ...(payload.includeRecent !== undefined ? { includeRecent: payload.includeRecent } : {}),
        },
      })
    },
  }), [searchMentionsMutation])

  const [inputValue, setInputValue] = useState<InputValue>({
    ...EMPTY_INPUT,
    text: composerText,
    cursorPosition: composerText.length,
  })
  const [attachments, setAttachments] = useState<PendingImageAttachment[]>([])
  const [history, setHistory] = useState<string[]>([])
  const [savedDraft, setSavedDraft] = useState('')
  const historySeededRef = useRef(false)
  const historyNavRef = useRef(false)
  const [nextEscWillKillAll, setNextEscWillKillAll] = useState(false)
  const killAllTimeoutRef = useRef<NodeJS.Timeout | null>(null)
  const [modelLabelHovered, setModelLabelHovered] = useState(false)
  const [thinkingLabelHovered, setThinkingLabelHovered] = useState(false)
  const [memoryLabelHovered, setMemoryLabelHovered] = useState(false)
  const [thinkingOpen, setThinkingOpen] = useState(false)
  const currentThinkingIndex = Math.max(
    0,
    thinkingOptions.findIndex((option) =>
      option.label.toLowerCase() === modelSummary?.thinkingLevel.toLowerCase()),
  )
  const [thinkingIndex, setThinkingIndex] = useState(currentThinkingIndex)
  const modelFooter = useMemo(
    () => deriveLocalInferenceFooterView(
      localInferenceState,
      modelSummary?.model ?? null,
      selectedProviderId,
      selectedSlotId,
    ),
    [localInferenceState, modelSummary?.model, selectedProviderId, selectedSlotId],
  )
  const workingDirectoryLabel = displayWorkingDirectory(clientWorkingDirectory)
  const modelNameLabel = modelFooter.modelName ?? modelSummary?.model ?? 'Choose a model'
  const thinkingLevelLabel = modelSummary?.thinkingLevel ?? '-'
  const footerTransientWidth = nextEscWillKillAll
    ? 3 + 'Press Esc again to interrupt all workers'.length
    : bashMode
      ? 3 + 'Esc to exit Bash mode'.length
      : 0
  const footerModeWidth = displayMode === 'transcript'
    ? 3 + 'Transcript Mode'.length
    : 0
  const footerLeftWidth = (bashMode
    ? 'Bash Mode'.length
    : !modelsConfigured
      ? 'No provider configured'.length
      : (modelFooter.residency === null ? 0 : 2)
        + stringWidth(modelNameLabel)
        + 2
        + stringWidth(thinkingLevelLabel)
        + 3
        + contextUsageWidth(tokenUsage, contextHardCap, isCompacting)
        + (modelFooter.memoryLabel === null
          ? 0
          : 3 + 'Memory: '.length + stringWidth(modelFooter.memoryLabel)))
    + footerModeWidth
    + footerTransientWidth
  const footerRightWidth = stringWidth(workingDirectoryLabel)
  const footerStacks = footerLeftWidth + footerRightWidth + 8 > chatColumnWidth
  const openThinking = useCallback(() => {
    if (thinkingOptions.length === 0) return
    setThinkingIndex(currentThinkingIndex)
    setThinkingOpen(true)
  }, [currentThinkingIndex, thinkingOptions.length])
  const toggleThinking = useCallback(() => {
    if (thinkingOpen) {
      setThinkingOpen(false)
    } else {
      openThinking()
    }
  }, [openThinking, thinkingOpen])
  const moveThinking = useCallback((direction: -1 | 1) => {
    setThinkingIndex((index) =>
      (index + direction + thinkingOptions.length) % thinkingOptions.length)
  }, [thinkingOptions.length])
  const commitThinking = useCallback(() => {
    const option = thinkingOptions[thinkingIndex]
    if (option) applyThinking(option.value)
    setThinkingOpen(false)
  }, [applyThinking, thinkingIndex, thinkingOptions])
  const multilineInputRef = useRef<MultilineInputHandle | null>(null)

  // Refs to mirror local state so the sync Effect reads the latest values
  // without depending on them in the useMemo dep array (which would recreate
  // the atom on every keystroke).
  const inputValueRef = useRef(inputValue)
  inputValueRef.current = inputValue
  const attachmentsRef = useRef(attachments)
  attachmentsRef.current = attachments

  // Sync inputValue from composerText atom when the atom changes externally
  // (e.g. queued-input restore, send-failure rollback). useAtomMount is the
  // sanctioned pattern for atom→local reconciliation when no single user
  // action is the trigger. Gates strictly on text divergence (never cursor
  // position) to avoid the feedback loop; preserves segments via ...prev.
  const syncFromAtomAtom = useMemo(
    () =>
      Atom.make(
        Effect.gen(function* () {
          if (composerText !== inputValueRef.current.text) {
            setInputValue((prev) => ({
              ...prev,
              text: composerText,
              cursorPosition: composerText.length,
            }))
            setComposerHasContent(composerText.trim().length > 0 || attachmentsRef.current.length > 0)
          }
        }),
      ),
    [composerText, setComposerHasContent],
  )
  useAtomMount(syncFromAtomAtom)

  // Composer focus — imperative, no useEffect
  if (composerCanFocus) multilineInputRef.current?.focus()

  // History seeding — ref-based imperative (no useEffect)
  if (!historySeededRef.current && displayMessages && displayMessages.length > 0) {
    const extractUserMessageText = (message: unknown): string => {
      if (!message || typeof message !== 'object') return ''
      const value = message as {
        type?: string
        message?: string
        visibleMessage?: string
        content?: unknown
      }
      if (value.type !== 'user_message') return ''
      if (typeof value.visibleMessage === 'string') return value.visibleMessage.trim()
      if (typeof value.message === 'string') return value.message.trim()

      const content = value.content
      if (typeof content === 'string') return content.trim()
      if (Array.isArray(content)) {
        return content
          .map((part) => {
            if (typeof part === 'string') return part
            if (!part || typeof part !== 'object') return ''
            const p = part as { text?: string; type?: string }
            if (typeof p.text === 'string') return p.text
            return ''
          })
          .join('')
          .trim()
      }
      return ''
    }

    const seededHistory = displayMessages
      .map((message) => extractUserMessageText(message))
      .filter((message) => message.length > 0)
      .slice(-MAX_HISTORY)

    setHistory(seededHistory)
    historySeededRef.current = true
  }

  const setComposerTextValue = useCallback((text: string) => {
    setInputValue({
      ...EMPTY_INPUT,
      text,
      cursorPosition: text.length,
    })
    setComposerText(text)
  }, [setComposerText])

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
    const mediaType = imageMediaTypeFromMime(scaled.mime)
    if (!mediaType) return false
    const newAttachment: PendingImageAttachment = {
      type: 'raw_image_clipboard',
      data: scaled.base64,
      mediaType,
      width: scaled.width,
      height: scaled.height,
    }
    setAttachments(prev => {
      const next = [...prev, newAttachment]
      setComposerHasContent(next.length > 0 || inputValueRef.current.text.trim().length > 0)
      return next
    })
    return true
  }, [setComposerHasContent])

  const addImageAttachmentFromFilePath = useCallback(async (rawPasteText: string) => {
    return addImageAttachmentsFromPastedText({
      rawPasteText,
      appendAttachments: (newAttachments) => {
        setAttachments(prev => [...prev, ...newAttachments])
      },
      readPastedImageParams: {
        cwd,
        resolvePath: (params) => resolvePathMutation({ payload: params }),
        readFile: (params) => readFileMutation({ payload: params }),
      },
    })
  }, [cwd, resolvePathMutation, readFileMutation])

  const removeAttachment = useCallback((index: number) => {
    setAttachments(prev => {
      const next = prev.filter((_, i) => i !== index)
      setComposerHasContent(next.length > 0 || inputValueRef.current.text.trim().length > 0)
      return next
    })
  }, [setComposerHasContent])

  const executeSlashCommand = useCallback((commandText: string) => {
    const handled = runSlashCommand(commandText)
    if (handled) {
      setInputValue(EMPTY_INPUT)
      setComposerText('')
      setComposerAttachments([])
      setAttachments([])
      setComposerHasContent(false)
    }
  }, [runSlashCommand, setComposerText, setComposerAttachments, setComposerHasContent])

  const onSelectMention = useCallback((item: { path: string; contentType: 'text' | 'directory'; lineRange?: { start: number; end: number } }) => {
    setInputValue(prev => {
      const left = prev.text.slice(0, Math.max(0, prev.cursorPosition))
      const match = left.match(/(?:^|\s)@([^\s@]*)$/)
      if (!match) return prev
      const atIndex = left.lastIndexOf('@')
      if (atIndex < 0) return prev
      return insertMentionSegment(prev, { path: item.path, contentType: item.contentType, lineRange: item.lineRange }, createId(), atIndex, left.length)
    })
  }, [])

  const onExpandDirectoryMention = useCallback((item: { path: string }) => {
    setInputValue(prev => {
      const left = prev.text.slice(0, Math.max(0, prev.cursorPosition))
      const match = left.match(/(?:^|\s)@([^\s@]*)$/)
      if (!match) return prev
      const atIndex = left.lastIndexOf('@')
      if (atIndex < 0) return prev
      return applyTextEditWithPastesAndMentions(prev, atIndex, left.length, `@${item.path}`)
    })
  }, [])

  const fileMentions = useFileMentions({
    inputText: inputValue.text,
    cursorPosition: inputValue.cursorPosition,
    client: mentionClient,
    cwd,
    onConfirm: onSelectMention,
    onExpandDirectory: onExpandDirectoryMention,
  })
  const slashCommands = useSlashCommands(inputValue.text, executeSlashCommand)

  const handleInterrupt = useCallback(() => interruptFork(selectedForkId), [interruptFork, selectedForkId])
  const handleInterruptAll = useCallback(() => interruptAll(), [interruptAll])

  const handleKeyIntercept = useCallback((key: KeyEvent): boolean => {
    if (!bashMode && fileMentions.handleKeyIntercept(key)) return true
    if (!bashMode && shouldHandleSlashCommandInTab(selectedForkId) && slashCommands.handleKeyIntercept(key)) return true
    const hasContent = inputValue.text.trim().length > 0 || attachments.length > 0
    if (widgetNavActive && !hasContent && handleWidgetKeyEvent(key)) return true

    const isPlainArrow = !key.ctrl && !key.meta && !key.option && !key.shift
    if (!isPlainArrow) return false

    if (key.name === 'up') {
      if (history.length === 0) return false
      if (inputValue.text.length > 0 && historyIndex == null) return false

      if (historyIndex == null) {
        const nextIndex = history.length - 1
        setSavedDraft(inputValue.text)
        setComposerHistoryIndex(nextIndex)
        historyNavRef.current = true
        setComposerTextValue(history[nextIndex] ?? '')
        return true
      }

      const nextIndex = Math.max(0, historyIndex - 1)
      setComposerHistoryIndex(nextIndex)
      historyNavRef.current = true
      setComposerTextValue(history[nextIndex] ?? '')
      return true
    }

    if (key.name === 'down') {
      if (historyIndex == null) return false
      if (history.length === 0) {
        setComposerHistoryIndex(-1)
        setSavedDraft('')
        historyNavRef.current = true
        setComposerTextValue(savedDraft)
        return true
      }

      if (historyIndex < history.length - 1) {
        const nextIndex = historyIndex + 1
        setComposerHistoryIndex(nextIndex)
        historyNavRef.current = true
        setComposerTextValue(history[nextIndex] ?? '')
        return true
      }

      setComposerHistoryIndex(-1)
      historyNavRef.current = true
      setComposerTextValue(savedDraft)
      setSavedDraft('')
      return true
    }

    return false
  }, [bashMode, widgetNavActive, fileMentions, slashCommands, handleWidgetKeyEvent, enterBashMode, exitBashMode, history, historyIndex, inputValue.text, savedDraft, setComposerTextValue, setComposerHistoryIndex, selectedForkId])

  const handleInputChange = useCallback((value: InputValue) => {
    if (!bashMode && value.text === '!') {
      enterBashMode()
      setInputValue(EMPTY_INPUT)
      setComposerText('')
      setComposerHasContent(false)
      setComposerHistoryIndex(-1)
      setSavedDraft('')
      return
    }
    if (historyNavRef.current) {
      historyNavRef.current = false
    } else if (composerHistoryIndex !== -1 && value.text !== (history[composerHistoryIndex] ?? '')) {
      setComposerHistoryIndex(-1)
      setSavedDraft('')
    }
    setInputValue(value)
    setComposerText(value.text)
    setComposerHasContent(value.text.trim().length > 0 || attachmentsRef.current.length > 0)
    setComposerHistoryIndex(composerHistoryIndex === -1 ? -1 : composerHistoryIndex)
  }, [bashMode, composerHistoryIndex, history, enterBashMode, setComposerText, setComposerHasContent, setComposerHistoryIndex])

  const handlePaste = useCallback(async (eventText?: string): Promise<boolean> => {
    const result = await handleChatControllerPaste({
      eventText,
      addClipboardImage: addImageAttachment,
      addImageFromFilePath: addImageAttachmentFromFilePath,
      setInputValue,
    })
    return result.didInsert
  }, [addImageAttachment, addImageAttachmentFromFilePath])

  const clearComposer = useCallback(() => {
    setInputValue(EMPTY_INPUT)
    setComposerText('')
    setComposerAttachments([])
    setAttachments([])
    setComposerHasContent(false)
    setComposerHistoryIndex(-1)
    setSavedDraft('')
  }, [setComposerText, setComposerAttachments, setComposerHasContent, setComposerHistoryIndex])

  const handleSubmit = useCallback(async (message: string, visibleMessage?: string, mentionInputs: RawMentionOccurrence[] = []) => {
    if (bashMode) {
      const trimmed = message.trim()
      if (!trimmed) return
      const didRun = await Promise.resolve(executeBash(trimmed))
      if (!didRun) return
      exitBashMode()
      clearComposer()
      return
    }
    if (!allowProviderMessageSend(modelsConfigured, showToast)) return

    clearSystemBanners()

    const content = message
    try {
      submitUserMessage({
        message: content,
        visibleMessage,
        imageAttachments: attachments,
        mentions: mentionInputs,
      })
    } catch (error) {
      showToast(`Message was not sent: ${error instanceof Error ? error.message : String(error)}`)
      return
    }

    const historyText = (visibleMessage ?? message).trim()
    if (historyText.length > 0) {
      setHistory(prev => [...prev, historyText].slice(-MAX_HISTORY))
    }
    setComposerHistoryIndex(-1)
    setSavedDraft('')
    clearComposer()
  }, [bashMode, modelsConfigured, submitUserMessage, executeBash, clearSystemBanners, showToast, attachments, clearComposer])

  const handleInputSubmit = useCallback(async () => {
    setComposerHistoryIndex(-1)
    setSavedDraft('')
    if (inputValue.text.trim() || attachments.length > 0) {
      const { text, mentions } = reconstituteInputTextWithMentions(inputValue)
      const mentionInputs = mentions.map(mentionOccurrenceFromInputSegment)
      await handleSubmit(text, inputValue.text, mentionInputs)
    }
  }, [inputValue, attachments.length, handleSubmit, setComposerHistoryIndex])

  const footerStatus = (
    <box style={{ flexDirection: 'row', alignItems: 'center' }}>
      {bashMode ? (
        <text style={{ fg: orange[400] }} attributes={TextAttributes.BOLD}>Bash Mode</text>
      ) : !modelsConfigured ? (
        <Button onClick={openSettings}>
          <text style={{ fg: theme.foreground }}>No provider configured</text>
        </Button>
      ) : (
        <>
          {modelFooter.residency !== null && (
            <box style={{ marginRight: 1 }}>
              <ResidencyIndicator residency={modelFooter.residency} />
            </box>
          )}
          <Button
            onClick={openSettings}
            onMouseOver={() => setModelLabelHovered(true)}
            onMouseOut={() => setModelLabelHovered(false)}
          >
            <text
              style={{ fg: modelLabelHovered ? theme.primary : theme.foreground }}
            >
              <span attributes={modelLabelHovered ? TextAttributes.UNDERLINE : TextAttributes.NONE}>
                {modelNameLabel}
              </span>
            </text>
          </Button>
          <box style={{ width: 2, flexShrink: 0 }} />
          <Button
            onClick={openThinking}
            onMouseOver={() => setThinkingLabelHovered(true)}
            onMouseOut={() => setThinkingLabelHovered(false)}
          >
            <text style={{ fg: thinkingLabelHovered ? theme.primary : theme.modePlan }}>
              <span attributes={thinkingLabelHovered ? TextAttributes.UNDERLINE : TextAttributes.NONE}>
                {thinkingLevelLabel}
              </span>
            </text>
          </Button>
          <box style={{ width: 3, flexShrink: 0 }} />
          <ContextUsage
            tokenUsage={tokenUsage}
            hardCap={contextHardCap}
            isCompacting={isCompacting}
          />
          {modelFooter.memoryLabel && (
            <>
              <box style={{ width: 3, flexShrink: 0 }} />
              <Button
                onClick={openHardware}
                onMouseOver={() => setMemoryLabelHovered(true)}
                onMouseOut={() => setMemoryLabelHovered(false)}
              >
                <text style={{ fg: memoryLabelHovered ? theme.primary : theme.muted }}>
                  <span attributes={memoryLabelHovered ? TextAttributes.UNDERLINE : TextAttributes.NONE}>
                    {`Memory: ${modelFooter.memoryLabel}`}
                  </span>
                </text>
              </Button>
            </>
          )}
        </>
      )}
      {displayMode === 'transcript' && (
        <>
          <box style={{ width: 3, flexShrink: 0 }} />
          <text style={{ fg: theme.info }}>Transcript Mode</text>
        </>
      )}
      {enableAutopilot && (
        <AutopilotIndicator
          enabled={autopilotEnabled}
          generating={autopilotGenerating}
          onToggle={toggleAutopilot}
        />
      )}
      {nextEscWillKillAll ? (
        <>
          <box style={{ width: 3, flexShrink: 0 }} />
          <text style={{ fg: theme.secondary }}>Press Esc again to interrupt all workers</text>
        </>
      ) : bashMode ? (
        <>
          <box style={{ width: 3, flexShrink: 0 }} />
          <text style={{ fg: theme.muted }}><span attributes={TextAttributes.BOLD}>Esc</span> to exit Bash mode</text>
        </>
      ) : null}
    </box>
  )

  const footerEnvironment = (
    <box style={{ flexDirection: 'row', alignItems: 'center' }}>
      <text style={{ fg: theme.muted }}>{workingDirectoryLabel}</text>
    </box>
  )

  return (
    <>
      <ChatSurfaceKeyboard
        status={status}
        hasRunningForks={hasRunningForks}
        isBlockingOverlayActive={isBlockingOverlayActive}
        nextEscWillKillAll={nextEscWillKillAll}
        setNextEscWillKillAll={setNextEscWillKillAll}
        killAllTimeoutRef={killAllTimeoutRef}
        onInterrupt={handleInterrupt}
        onInterruptAll={handleInterruptAll}
        composerHasContent={inputValue.text.trim().length > 0 || attachments.length > 0}
        onClearInput={clearComposer}
        bashMode={bashMode}
        onExitBashMode={() => {
          exitBashMode()
          clearComposer()
        }}
        onToggleAutopilot={enableAutopilot ? toggleAutopilot : undefined}
        thinkingOpen={thinkingOpen}
        thinkingOptionCount={thinkingOptions.length}
        onToggleThinking={toggleThinking}
        onMoveThinking={moveThinking}
        onApplyThinking={commitThinking}
        onCancelThinking={() => setThinkingOpen(false)}
      />

      <box style={{ paddingLeft: 1, paddingRight: 1, flexShrink: 0 }}>
        <box style={{
          borderStyle: 'single',
          border: ['left'],
          borderColor: bashMode ? orange[400] : modeColor,
          customBorderChars: COMPOSER_BORDER_CHARS,
        }}>
          <box style={{
            backgroundColor: theme.inputBg,
            paddingTop: 1,
            paddingBottom: 1,
            paddingLeft: 1,
            paddingRight: 2,
            flexDirection: 'column',
            flexGrow: 1,
          }}>
            {!bashMode && fileMentions.isOpen && (
              <FileMentionMenu
                isOpen={fileMentions.isOpen}
                query={fileMentions.query}
                items={fileMentions.items}
                recentItems={fileMentions.recentItems}
                overflowCount={fileMentions.overflowCount}
                selectedIndex={fileMentions.selectedIndex}
                onSelect={fileMentions.confirmSelection}
                onHoverIndex={fileMentions.setSelectedIndex}
              />
            )}
            {!bashMode && shouldHandleSlashCommandInTab(selectedForkId) && slashCommands.isSlashMenuOpen && (
              <SlashCommandMenu
                commands={slashCommands.filteredCommands}
                selectedIndex={slashCommands.selectedIndex}
                onSelect={(cmd) => executeSlashCommand(`/${cmd.id}`)}
                onHoverIndex={slashCommands.setSelectedIndex}
              />
            )}
            {!bashMode && thinkingOpen && (
              <box style={{
                flexDirection: 'column',
                borderStyle: 'single',
                borderColor: theme.primary,
                backgroundColor: theme.inputBg,
                paddingLeft: 1,
                paddingRight: 1,
                marginBottom: 1,
              }}>
                <text style={{ fg: theme.foreground }} attributes={TextAttributes.BOLD}>Thinking level</text>
                <text style={{ fg: theme.muted }}>↑↓ choose · Enter apply · Esc cancel</text>
                {thinkingOptions.map((option, index) => (
                  <Button
                    key={option.value}
                    onClick={() => {
                      applyThinking(option.value)
                      setThinkingOpen(false)
                    }}
                    onMouseOver={() => setThinkingIndex(index)}
                  >
                    <text style={{ fg: index === thinkingIndex ? theme.primary : theme.foreground }}>
                      {index === thinkingIndex ? '› ' : '  '}{option.label}
                    </text>
                  </Button>
                ))}
              </box>
            )}
            {attachments.length > 0 && (
              <AttachmentsBar
                attachments={attachments}
                onRemove={removeAttachment}
                maxWidth={attachmentsMaxWidth}
              />
            )}
            <box style={{ flexDirection: 'row', alignItems: 'center', width: '100%' }}>
              <box style={{ flexGrow: 1, minWidth: 0 }}>
                <MultilineInput
                  ref={multilineInputRef}
                  value={inputValue.text}
                  cursorPosition={inputValue.cursorPosition}
                  pasteSegments={inputValue.pasteSegments}
                  selectedPasteSegmentId={inputValue.selectedPasteSegmentId}
                  mentionSegments={inputValue.mentionSegments}
                  selectedMentionSegmentId={inputValue.selectedMentionSegmentId}
                  onChange={handleInputChange}
                  onSubmit={handleInputSubmit}
                  onPaste={handlePaste}
                  onKeyIntercept={handleKeyIntercept}
                  focused={composerCanFocus && !thinkingOpen}
                  highlightColor={bashMode ? orange[400] : undefined}
                  placeholder={bashMode ? 'Enter a command...' : status === 'streaming' ? 'Type to queue a message...' : 'Chat with the agent...'}
                  maxHeight={10}
                  minHeight={1}
                />
              </box>
            </box>
          </box>
        </box>
      </box>

      <box style={{ paddingLeft: 2, paddingRight: 2, flexShrink: 0, flexDirection: 'column' }}>
        <box style={{ height: 1, flexDirection: 'row', justifyContent: 'space-between' }}>
          {footerStatus}
          {!footerStacks && footerEnvironment}
        </box>
        {footerStacks && (
          <box style={{ height: 1, flexDirection: 'row', justifyContent: 'flex-end' }}>
            {footerEnvironment}
          </box>
        )}
      </box>

    </>
  )
}
