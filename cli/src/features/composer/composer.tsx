import { RGBA, TextAttributes, type KeyEvent } from '@opentui/core'
import { useCallback, useRef, useState, useMemo } from 'react'
import type {
  MentionAttachment,
  RawImageAttachment,
  RawMessageAttachment,
  ReadFileResult,
} from '@magnitudedev/sdk'
import { imageMediaTypeFromFilename, filenameWithImageExtension, useAgentClient, mentionAttachmentFromSegment, imageMediaTypeFromMime } from '@magnitudedev/client-common'
import { createId } from '@magnitudedev/generate-id'
import path from 'path'
import { BOX_CHARS } from '../../utils/ui-constants'
import { orange } from '../../utils/theme'
import { Button } from '../../components/button'
import { ChatSurfaceKeyboard } from './chat-surface-keyboard'
import { FileMentionMenu } from './mention-menu'
import { SlashCommandMenu } from './slash-menu'
import { MultilineInput, type MultilineInputHandle } from './multiline-input'
import { AttachmentsBar } from './attachment-bar'
import { ContextUsageBar } from '../agent-status/context-usage-bar'
import { AutopilotIndicator } from './autopilot-indicator'
import { useFileMentions, type MentionSearchClient } from '@magnitudedev/client-common'
import { useSlashCommands } from '@magnitudedev/client-common'
import { readClipboardBitmap, readClipboardText, extractImageDimensions } from '../../utils/clipboard'
import { extractPastedPathCandidates, tryReadPastedImageFileCandidate, type ReadPastedImageFileParams } from '../../utils/pasted-image-path'
import { autoScaleImageAttachmentIfNeeded } from '../../utils/image-scaling'
import {
  applyTextEditWithPastesAndMentions,
  insertMentionSegment,
  reconstituteInputTextWithMentions,
} from '@magnitudedev/client-common'
import { resolvePasteIntent, resolvePasteOutcomeFromApplyResult } from '@magnitudedev/client-common'
import { applyPasteIntent } from '@magnitudedev/client-common'
import { derivePasteEffects } from '@magnitudedev/client-common'
import { composerTextAtom, composerAttachmentsAtom, composerHistoryIndexAtom } from '@magnitudedev/client-common'
import { useAtomSet, useAtomValue as useAtomValueClientCommon } from '@effect-atom/atom-react'
import type { InputValue } from '@magnitudedev/client-common'
import type { ComposerProps } from './types'
import { shouldHandleSlashCommandInTab } from '@magnitudedev/client-common'

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

export async function handleChatControllerPaste(args: {
  eventText?: string
  addClipboardImage: () => Promise<boolean>
  addImageFromFilePath: (rawPasteText: string) => Promise<boolean>
  setInputValue: (updater: (prev: InputValue) => InputValue) => void
}): Promise<{ didInsert: boolean; shouldBumpBulkInsertEpoch: boolean }> {
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

  const effects = derivePasteEffects(applyResult)
  return {
    didInsert: resolvePasteOutcomeFromApplyResult(applyResult),
    shouldBumpBulkInsertEpoch: effects.shouldBumpBulkInsertEpoch,
  }
}

export function nextBulkInsertEpochForPaste(previousEpoch: number, shouldBumpBulkInsertEpoch: boolean): number {
  return shouldBumpBulkInsertEpoch ? previousEpoch + 1 : previousEpoch
}

export function buildRestoredQueuedInputValue(restoredQueuedInputText: string): InputValue {
  return {
    ...EMPTY_INPUT,
    text: restoredQueuedInputText,
    cursorPosition: restoredQueuedInputText.length,
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
    status,
    hasRunningForks,
    bashMode,
    modelsConfigured,
    modelSummary,
    tokenUsage,
    contextHardCap,
    isCompacting,
    displayMode,
    theme,
    modeColor,
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
    onInputHasTextChange,
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
  const [bulkInsertEpoch, setBulkInsertEpoch] = useState(0)
  const [modelLabelHovered, setModelLabelHovered] = useState(false)
  const [thinkingLabelHovered, setThinkingLabelHovered] = useState(false)
  const multilineInputRef = useRef<MultilineInputHandle | null>(null)

  // Sync inputValue from composerText atom when the atom changes externally (e.g. restored input).
  // This is the local component's view of the text derived from the shared atom.
  if (inputValue.text !== composerText || inputValue.cursorPosition !== composerText.length) {
    setInputValue({
      ...EMPTY_INPUT,
      text: composerText,
      cursorPosition: composerText.length,
    })
  }

  // onInputHasTextChange — imperative, no useEffect
  onInputHasTextChange?.(inputValue.text.trim().length > 0 || attachments.length > 0)

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
    setAttachments(prev => [...prev, newAttachment])
    return true
  }, [])

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
    setAttachments(prev => prev.filter((_, i) => i !== index))
  }, [])

  const executeSlashCommand = useCallback((commandText: string) => {
    const handled = runSlashCommand(commandText)
    if (handled) {
      setInputValue(EMPTY_INPUT)
      setComposerText('')
      setComposerAttachments([])
      setAttachments([])
    }
  }, [runSlashCommand, setComposerText, setComposerAttachments])

  const onSelectMention = useCallback((item: { path: string; contentType: 'text' | 'directory'; lineRange?: { start: number; end: number } }) => {
    // Image files selected via @mention become pending image inputs
    // (same path as pasted images), not file mentions.
    const mediaType = item.contentType === 'text' ? imageMediaTypeFromFilename(item.path) : null
    if (mediaType) {
      if (cwd) {
        void readFileMutation({
          payload: { cwd, path: item.path, format: 'base64' },
        }).then((read: ReadFileResult) => {
          if (read.format !== 'base64' || read.content.length === 0) return
          const buffer = Buffer.from(read.content, 'base64')
          const dims = extractImageDimensions(buffer)
          if (!dims) return
          const newAttachment: PendingImageAttachment = {
            type: 'raw_image_file',
            data: read.content,
            filename: filenameWithImageExtension(path.basename(item.path), mediaType),
            mediaType,
            width: dims.width,
            height: dims.height,
          }
          setAttachments(prev => [...prev, newAttachment])
        }).catch(() => { /* ignore read errors */ })
      }
      // Remove the @query from the input — the attachment pill replaces it
      setInputValue(prev => {
        const left = prev.text.slice(0, Math.max(0, prev.cursorPosition))
        const match = left.match(/(?:^|\s)@([^\s@]*)$/)
        if (!match) return prev
        const atIndex = left.lastIndexOf('@')
        if (atIndex < 0) return prev
        return applyTextEditWithPastesAndMentions(prev, atIndex, left.length, '')
      })
      return
    }

    setInputValue(prev => {
      const left = prev.text.slice(0, Math.max(0, prev.cursorPosition))
      const match = left.match(/(?:^|\s)@([^\s@]*)$/)
      if (!match) return prev
      const atIndex = left.lastIndexOf('@')
      if (atIndex < 0) return prev
      return insertMentionSegment(prev, { path: item.path, contentType: item.contentType, lineRange: item.lineRange }, createId(), atIndex, left.length)
    })
  }, [cwd, readFileMutation])

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
    setComposerHistoryIndex(composerHistoryIndex === -1 ? -1 : composerHistoryIndex)
  }, [bashMode, composerHistoryIndex, history, enterBashMode, setComposerText, setComposerHistoryIndex])

  const handlePaste = useCallback(async (eventText?: string): Promise<boolean> => {
    const result = await handleChatControllerPaste({
      eventText,
      addClipboardImage: addImageAttachment,
      addImageFromFilePath: addImageAttachmentFromFilePath,
      setInputValue,
    })
    setBulkInsertEpoch((prev) => nextBulkInsertEpochForPaste(prev, result.shouldBumpBulkInsertEpoch))
    return result.didInsert
  }, [addImageAttachment, addImageAttachmentFromFilePath])

  const clearComposer = useCallback(() => {
    setInputValue(EMPTY_INPUT)
    setComposerText('')
    setComposerAttachments([])
    setAttachments([])
    setComposerHistoryIndex(-1)
    setSavedDraft('')
  }, [setComposerText, setComposerAttachments, setComposerHistoryIndex])

  const handleSubmit = useCallback(async (message: string, visibleMessage?: string, mentionInputs: MentionAttachment[] = []) => {
    if (bashMode) {
      const trimmed = message.trim()
      if (!trimmed) return
      await Promise.resolve(executeBash(trimmed))
      exitBashMode()
      clearComposer()
      return
    }
    if (!modelsConfigured) return

    clearSystemBanners()

    const content = message
    const rawMessageAttachments: RawMessageAttachment[] = [...attachments, ...mentionInputs]

    try {
      submitUserMessage({
        message: content,
        visibleMessage,
        attachments: rawMessageAttachments,
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
      const mentionInputs = mentions.map(mentionAttachmentFromSegment)
      await handleSubmit(text, inputValue.text, mentionInputs)
    }
  }, [inputValue, attachments.length, handleSubmit, setComposerHistoryIndex])

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
      />

      <box style={{ paddingLeft: 1, paddingRight: 1, flexShrink: 0 }}>
        <box style={{ height: 1, borderStyle: 'single', border: ['left'], borderColor: bashMode ? orange[400] : modeColor, customBorderChars: { topLeft: '', bottomLeft: '', topRight: '', bottomRight: '', horizontal: ' ', vertical: '╻', topT: '', bottomT: '', leftT: '', rightT: '', cross: '' } }}>
          <box style={{ height: 1, borderStyle: 'single', border: ['top'], borderColor: theme.inputBg, customBorderChars: { topLeft: '', bottomLeft: '', topRight: '', bottomRight: '', horizontal: '▄', vertical: ' ', topT: '', bottomT: '', leftT: '', rightT: '', cross: '' } }} />
        </box>
        <box style={{ borderStyle: 'single', border: ['left'], borderColor: bashMode ? orange[400] : modeColor, customBorderChars: { ...BOX_CHARS, vertical: '┃' } }}>
          <box style={{ backgroundColor: theme.inputBg, paddingTop: 1, paddingLeft: 1, paddingRight: 2, flexDirection: 'column', flexGrow: 1 }}>
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
            <box style={{ flexDirection: 'column' }}>
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
                    focused={composerCanFocus}
                    highlightColor={bashMode ? orange[400] : undefined}
                    placeholder={bashMode ? 'Enter a command...' : status === 'streaming' ? 'Type to queue a message...' : 'Chat with the agent...'}
                    maxHeight={10}
                    minHeight={1}
                    bulkInsertEpoch={bulkInsertEpoch}
                  />
                </box>
              </box>
              <box style={{ flexDirection: 'row', justifyContent: 'flex-end' }}>
                {bashMode ? (
                  <text style={{ fg: orange[400] }} attributes={TextAttributes.BOLD}>Bash Mode</text>
                ) : (
                  <>
                    <Button
                      onClick={openSettings}
                      onMouseOver={() => setModelLabelHovered(true)}
                      onMouseOut={() => setModelLabelHovered(false)}
                    >
                      <text style={{
                        fg: modelLabelHovered ? theme.primary : theme.foreground,
                      }}>
                        <span attributes={modelLabelHovered ? TextAttributes.UNDERLINE : TextAttributes.NONE}>
                          {modelSummary?.model ?? '-'}
                        </span>
                      </text>
                    </Button>
                    <text style={{ fg: theme.muted }}> {'\u00b7'} </text>
                    <Button
                      onClick={openSettings}
                      onMouseOver={() => setThinkingLabelHovered(true)}
                      onMouseOut={() => setThinkingLabelHovered(false)}
                    >
                      <text style={{
                        fg: thinkingLabelHovered ? theme.primary : theme.foreground,
                      }}>
                        <span attributes={thinkingLabelHovered ? TextAttributes.UNDERLINE : TextAttributes.NONE}>
                          {modelSummary?.thinkingLevel ?? '-'}
                        </span>
                      </text>
                    </Button>
                  </>
                )}
              </box>
            </box>
          </box>
        </box>
        <box style={{ height: 1, borderStyle: 'single', border: ['left'], borderColor: bashMode ? orange[400] : modeColor, customBorderChars: { topLeft: '', bottomLeft: '', topRight: '', bottomRight: '', horizontal: ' ', vertical: '╹', topT: '', bottomT: '', leftT: '', rightT: '', cross: '' } }}>
          <box style={{ height: 1, borderStyle: 'single', border: ['bottom'], borderColor: theme.inputBg, customBorderChars: { topLeft: '', bottomLeft: '', topRight: '', bottomRight: '', horizontal: '▀', vertical: ' ', topT: '', bottomT: '', leftT: '', rightT: '', cross: '' } }} />
        </box>
      </box>

      <box style={{ paddingLeft: 2, paddingRight: 2, flexShrink: 0, height: 1, minHeight: 1, maxHeight: 1, flexDirection: 'row', justifyContent: 'space-between' }}>
        <box style={{ flexDirection: 'row', alignItems: 'center', gap: 1 }}>
          {enableAutopilot && (
            <AutopilotIndicator
              enabled={autopilotEnabled}
              generating={autopilotGenerating}
              onToggle={toggleAutopilot}
            />
          )}
          {attachments.length > 0 ? (
            <AttachmentsBar attachments={attachments} onRemove={removeAttachment} maxWidth={attachmentsMaxWidth} />
          ) : nextEscWillKillAll ? (
            <text style={{ fg: theme.secondary }}>Press Esc again to interrupt all workers</text>
          ) : bashMode ? (
            <text style={{ fg: theme.muted }}><span attributes={TextAttributes.BOLD}>Esc</span> to exit Bash mode</text>
          ) : null}
        </box>
        <box style={{ flexDirection: 'row', alignItems: 'center', gap: 1 }}>
          {attachments.length > 0 && (nextEscWillKillAll ? (
            <text style={{ fg: theme.secondary }}>Press Esc again to interrupt all workers</text>
          ) : bashMode ? (
            <text style={{ fg: theme.muted }}><span attributes={TextAttributes.BOLD}>Esc</span> to exit Bash mode</text>
          ) : null)}
          {displayMode === 'transcript' && (
            <>
              <text style={{ fg: theme.info }}>Transcript Mode</text>
              <text style={{ fg: theme.muted }}>{' \u00b7 '}</text>
            </>
          )}
          <ContextUsageBar
            tokenUsage={tokenUsage}
            hardCap={contextHardCap}
            isCompacting={isCompacting}
          />
        </box>
      </box>

    </>
  )
}
