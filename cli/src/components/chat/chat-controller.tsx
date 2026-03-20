import { RGBA, TextAttributes, type KeyEvent } from '@opentui/core'
import { useCallback, useEffect, useRef, useState } from 'react'
import type { Attachment, ImageAttachment, ImageMediaType } from '@magnitudedev/agent'
import { createId } from '@magnitudedev/generate-id'

import { BOX_CHARS } from '../../utils/ui-constants'
import { orange } from '../../utils/theme'
import { Button } from '../button'
import { ChatSurfaceKeyboard } from '../chat-surface-keyboard'
import { FileMentionMenu } from '../file-mention-menu'
import { SlashCommandMenu } from '../slash-command-menu'
import { MultilineInput, type MultilineInputHandle } from '../multiline-input'
import { AttachmentsBar } from '../attachments-bar'
import { ContextUsageBar } from '../context-usage-bar'
import { useFileMentions } from '../../hooks/use-file-mentions'
import { useSlashCommands } from '../../hooks/use-slash-commands'
import { readClipboardBitmap, readClipboardText } from '../../utils/clipboard'
import { tryReadPastedImageFile } from '../../utils/pasted-image-path'
import { autoScaleImageAttachmentIfNeeded } from '../../utils/image-scaling'
import {
  applyTextEditWithPastesAndMentions,
  insertMentionSegment,
  insertPasteSegment,
  reconstituteInputTextWithMentions,
} from '../../utils/strings'
import type { InputValue } from '../../types/store'
import type { ChatControllerProps } from './types'
import { SubagentTabBar } from './subagent-tab-bar'
import { buildSubmitDispatchEvents, shouldHandleSlashCommandInTab } from './submit-routing'

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

export type PasteFlowOutcome = 'clipboard-image' | 'empty' | 'pasted-image-path' | 'text-inline' | 'text-segment'

export async function runChatPasteFlow(args: {
  eventText?: string
  readClipboardText: () => string
  addClipboardImage: () => Promise<boolean>
  addImageFromFilePath: (rawPasteText: string) => Promise<boolean>
  inlinePastePillCharLimit: number
  insertText: (pasteText: string) => void
  insertPasteSegment: (pasteText: string) => void
}): Promise<PasteFlowOutcome> {
  const eventText = args.eventText ?? ''
  const hasNativeEventText = eventText.length > 0
  const pasteText = hasNativeEventText ? eventText : args.readClipboardText()

  if (!pasteText) {
    const wasClipboardImage = await args.addClipboardImage()
    if (wasClipboardImage) return 'clipboard-image'
    return 'empty'
  }

  const wasImagePath = await args.addImageFromFilePath(pasteText)
  if (wasImagePath) return 'pasted-image-path'

  if (pasteText.length > args.inlinePastePillCharLimit) {
    args.insertPasteSegment(pasteText)
    return 'text-segment'
  }

  args.insertText(pasteText)
  return 'text-inline'
}

export async function handleChatControllerPaste(args: {
  eventText?: string
  addClipboardImage: () => Promise<boolean>
  addImageFromFilePath: (rawPasteText: string) => Promise<boolean>
  setInputValue: (updater: (prev: InputValue) => InputValue) => void
}) {
  await runChatPasteFlow({
    eventText: args.eventText,
    readClipboardText,
    addClipboardImage: args.addClipboardImage,
    addImageFromFilePath: args.addImageFromFilePath,
    inlinePastePillCharLimit: INLINE_PASTE_PILL_CHAR_LIMIT,
    insertText: (pasteText) => {
      args.setInputValue((prev) =>
        applyTextEditWithPastesAndMentions(prev, prev.cursorPosition, prev.cursorPosition, pasteText),
      )
    },
    insertPasteSegment: (pasteText) => {
      args.setInputValue((prev) => insertPasteSegment(prev, pasteText, createId()))
    },
  })
}

export function ChatController(props: ChatControllerProps) {
  const {
    env,
    services,
    displayMessages,
    subagentTabs,
    selectedForkId,
    onSubagentTabSelect,
    selectedFileOpen,
    onCloseFilePanel,
    onApprove,
    onReject,
    onInputHasTextChange,
    restoredQueuedInputText,
    onRestoredQueuedInputHandled,
  } = props
  const [inputValue, setInputValue] = useState<InputValue>(EMPTY_INPUT)
  const [attachments, setAttachments] = useState<ImageAttachment[]>([])
  const [history, setHistory] = useState<string[]>([])
  const [historyIndex, setHistoryIndex] = useState<number | null>(null)
  const [savedDraft, setSavedDraft] = useState('')
  const historySeededRef = useRef(false)
  const historyNavRef = useRef(false)
  const [nextEscWillKillAll, setNextEscWillKillAll] = useState(false)
  const killAllTimeoutRef = useRef<NodeJS.Timeout | null>(null)
  const [nextEscWillClearInput, setNextEscWillClearInput] = useState(false)
  const clearInputTimeoutRef = useRef<NodeJS.Timeout | null>(null)
  const [isProviderHovered, setIsProviderHovered] = useState(false)
  const [isModelHovered, setIsModelHovered] = useState(false)
  const [pendingKillForkId, setPendingKillForkId] = useState<string | null>(null)
  const [isKillCancelHovered, setIsKillCancelHovered] = useState(false)
  const [isKillConfirmHovered, setIsKillConfirmHovered] = useState(false)
  const multilineInputRef = useRef<MultilineInputHandle | null>(null)

  useEffect(() => {
    onInputHasTextChange?.(inputValue.text.trim().length > 0 || attachments.length > 0)
  }, [inputValue.text, attachments.length, onInputHasTextChange])

  useEffect(() => {
    if (restoredQueuedInputText == null) return
    setInputValue({
      ...EMPTY_INPUT,
      text: restoredQueuedInputText,
      cursorPosition: restoredQueuedInputText.length,
    })
    onRestoredQueuedInputHandled?.()
  }, [restoredQueuedInputText, onRestoredQueuedInputHandled])

  useEffect(() => {
    if (env.composerCanFocus) multilineInputRef.current?.focus()
  }, [env.composerCanFocus])

  useEffect(() => {
    if (historySeededRef.current) return
    if (!displayMessages || displayMessages.length === 0) return

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
    setHistoryIndex(null)
    historySeededRef.current = true
  }, [displayMessages])

  const setComposerText = useCallback((text: string) => {
    setInputValue({
      ...EMPTY_INPUT,
      text,
      cursorPosition: text.length,
    })
  }, [])

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
    const extension = scaled.mime === 'image/jpeg' ? '.jpg' : '.png'
    setAttachments(prev => [...prev, {
      type: 'image',
      base64: scaled.base64,
      mediaType: scaled.mime as ImageMediaType,
      width: scaled.width,
      height: scaled.height,
      filename: 'clipboard-' + Date.now() + extension,
    }])
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
    const parsed = result.filename.includes('.') ? result.filename.split('.') : [result.filename]
    const stem = parsed.slice(0, -1).join('.') || result.filename
    const filename = scaled.mime === 'image/jpeg' ? `${stem}.jpg` : result.filename
    setAttachments(prev => [...prev, {
      type: 'image',
      base64: scaled.base64,
      mediaType: scaled.mime as ImageMediaType,
      width: scaled.width,
      height: scaled.height,
      filename,
    }])
    return true
  }, [])

  const removeAttachment = useCallback((index: number) => {
    setAttachments(prev => prev.filter((_, i) => i !== index))
  }, [])

  const executeSlashCommand = useCallback((commandText: string) => {
    const handled = services.runSlashCommand(commandText)
    if (handled) {
      setInputValue(EMPTY_INPUT)
      setAttachments([])
    }
  }, [services])

  const onSelectMention = useCallback((item: { path: string; contentType: 'text' | 'image' | 'directory' }) => {
    setInputValue(prev => {
      const left = prev.text.slice(0, Math.max(0, prev.cursorPosition))
      const match = left.match(/(?:^|\s)@([^\s@]*)$/)
      if (!match) return prev
      const atIndex = left.lastIndexOf('@')
      if (atIndex < 0) return prev
      return insertMentionSegment(prev, { path: item.path, contentType: item.contentType }, createId(), atIndex, left.length)
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

  const fileMentions = useFileMentions(
    inputValue.text,
    inputValue.cursorPosition,
    onSelectMention,
    onExpandDirectoryMention,
  )
  const slashCommands = useSlashCommands(inputValue.text, executeSlashCommand)

  const handleInterrupt = useCallback(() => services.interruptFork(selectedForkId), [services, selectedForkId])
  const handleInterruptAll = useCallback(() => services.interruptAll(), [services])

  const handleKeyIntercept = useCallback((key: KeyEvent): boolean => {
    if (env.pendingApproval) return true
    if (!env.bashMode && fileMentions.handleKeyIntercept(key)) return true
    if (!env.bashMode && shouldHandleSlashCommandInTab(selectedForkId) && slashCommands.handleKeyIntercept(key)) return true
    if (env.widgetNavActive && services.handleWidgetKeyEvent(key)) return true

    const isPlainArrow = !key.ctrl && !key.meta && !key.option && !key.shift
    if (!isPlainArrow) return false

    if (key.name === 'up') {
      if (history.length === 0) return false
      if (inputValue.text.length > 0 && historyIndex == null) return false

      if (historyIndex == null) {
        const nextIndex = history.length - 1
        setSavedDraft(inputValue.text)
        setHistoryIndex(nextIndex)
        historyNavRef.current = true
        setComposerText(history[nextIndex] ?? '')
        return true
      }

      const nextIndex = Math.max(0, historyIndex - 1)
      setHistoryIndex(nextIndex)
      historyNavRef.current = true
      setComposerText(history[nextIndex] ?? '')
      return true
    }

    if (key.name === 'down') {
      if (historyIndex == null) return false
      if (history.length === 0) {
        setHistoryIndex(null)
        setSavedDraft('')
        historyNavRef.current = true
        setComposerText(savedDraft)
        return true
      }

      if (historyIndex < history.length - 1) {
        const nextIndex = historyIndex + 1
        setHistoryIndex(nextIndex)
        historyNavRef.current = true
        setComposerText(history[nextIndex] ?? '')
        return true
      }

      setHistoryIndex(null)
      historyNavRef.current = true
      setComposerText(savedDraft)
      setSavedDraft('')
      return true
    }

    return false
  }, [env.pendingApproval, env.bashMode, env.widgetNavActive, fileMentions, slashCommands, services, history, historyIndex, inputValue.text, savedDraft, setComposerText, selectedForkId])

  const handleInputChange = useCallback((value: InputValue) => {
    if (!env.bashMode && value.text === '!') {
      services.enterBashMode()
      setInputValue(EMPTY_INPUT)
      setHistoryIndex(null)
      setSavedDraft('')
      return
    }
    if (nextEscWillClearInput) {
      setNextEscWillClearInput(false)
      if (clearInputTimeoutRef.current) clearTimeout(clearInputTimeoutRef.current)
    }
    if (historyNavRef.current) {
      historyNavRef.current = false
    } else if (historyIndex != null && value.text !== (history[historyIndex] ?? '')) {
      setHistoryIndex(null)
      setSavedDraft('')
    }
    setInputValue(value)
  }, [env.bashMode, nextEscWillClearInput, historyIndex, history])

  const handlePaste = useCallback(async (eventText?: string) => {
    await handleChatControllerPaste({
      eventText,
      addClipboardImage: addImageAttachment,
      addImageFromFilePath: addImageAttachmentFromFilePath,
      setInputValue,
    })
  }, [addImageAttachment, addImageAttachmentFromFilePath])

  const clearComposer = useCallback(() => {
    setInputValue(EMPTY_INPUT)
    setAttachments([])
    setHistoryIndex(null)
    setSavedDraft('')
  }, [])

  const handleSubmit = useCallback((message: string, visibleMessage?: string, mentionAttachments: Attachment[] = []) => {
    const targetForkId = selectedForkId
    const slashText = visibleMessage ?? message
    if (!env.bashMode && shouldHandleSlashCommandInTab(targetForkId) && services.runSlashCommand(slashText)) {
      clearComposer()
      return
    }
    if (env.bashMode) {
      const trimmed = message.trim()
      if (!trimmed) return
      const result = services.executeBash(trimmed)
      services.appendBashOutput(result)
      setInputValue(EMPTY_INPUT)
      setHistoryIndex(null)
      setSavedDraft('')
      return
    }
    if (!env.modelsConfigured) return

    const historyText = (visibleMessage ?? message).trim()
    if (historyText.length > 0) {
      setHistory(prev => [...prev, historyText].slice(-MAX_HISTORY))
    }
    setHistoryIndex(null)
    setSavedDraft('')

    services.clearSystemBanners()
    const currentAttachments: Attachment[] = [...attachments, ...mentionAttachments]
    clearComposer()

    for (const event of buildSubmitDispatchEvents(targetForkId)) {
      services.submitUserMessageToFork({
        forkId: event.forkId,
        message,
        visibleMessage,
        mentionAttachments,
        attachments: currentAttachments,
      })
    }
  }, [env.bashMode, env.modelsConfigured, selectedForkId, services, attachments, clearComposer])

  const handleInputSubmit = useCallback(() => {
    setHistoryIndex(null)
    setSavedDraft('')
    if (inputValue.text.trim() || attachments.length > 0) {
      const { text, mentions } = reconstituteInputTextWithMentions(inputValue)
      const mentionAttachments: Attachment[] = mentions.map((mention) => ({
        type: 'mention',
        path: mention.path,
        contentType: mention.contentType,
      }))
      handleSubmit(text, inputValue.text, mentionAttachments)
    }
  }, [inputValue, attachments.length, handleSubmit])

  const selectedSubagentAgentId = selectedForkId == null
    ? null
    : (subagentTabs.find((tab) => tab.forkId === selectedForkId)?.agentId ?? selectedForkId)

  const pendingKillTab = pendingKillForkId == null
    ? null
    : (subagentTabs.find((tab) => tab.forkId === pendingKillForkId) ?? null)

  return (
    <>
      <ChatSurfaceKeyboard
        status={env.status}
        hasRunningForks={env.hasRunningForks}
        nextEscWillKillAll={nextEscWillKillAll}
        setNextEscWillKillAll={setNextEscWillKillAll}
        killAllTimeoutRef={killAllTimeoutRef}
        onInterrupt={handleInterrupt}
        onInterruptAll={handleInterruptAll}
        inputText={inputValue.text}
        nextEscWillClearInput={nextEscWillClearInput}
        setNextEscWillClearInput={setNextEscWillClearInput}
        clearInputTimeoutRef={clearInputTimeoutRef}
        onClearInput={clearComposer}
        selectedFileOpen={selectedFileOpen}
        onCloseFilePanel={onCloseFilePanel}
        bashMode={env.bashMode}
        onExitBashMode={() => {
          services.exitBashMode()
          clearComposer()
        }}
        fileMentionOpen={fileMentions.isOpen}
        slashMenuOpen={slashCommands.isSlashMenuOpen}
        onToggleTaskPanel={services.toggleTaskPanel}
        pendingApproval={env.pendingApproval}
        onApprove={onApprove}
        onReject={onReject}
      />

      <box style={{ paddingLeft: 1, paddingRight: 1, flexShrink: 0 }}>
        <box style={{ borderStyle: 'single', border: ['left'], borderColor: env.bashMode ? orange[400] : env.modeColor, customBorderChars: { ...BOX_CHARS, vertical: '┃' } }}>
          <box style={{ backgroundColor: env.theme.inputBg, paddingTop: 0, paddingLeft: 1, paddingRight: 2, flexDirection: 'column', flexGrow: 1 }}>
            <SubagentTabBar
              tabs={subagentTabs}
              selectedForkId={selectedForkId}
              onSelect={onSubagentTabSelect}
              onCloseTab={(forkId, phase) => {
                if (phase === 'idle') {
                  services.dismissIdleSubagentTab(forkId)
                  return
                }
                setPendingKillForkId(forkId)
              }}
            />
            {!env.bashMode && fileMentions.isOpen && (
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
            {!env.bashMode && shouldHandleSlashCommandInTab(selectedForkId) && slashCommands.isSlashMenuOpen && (
              <SlashCommandMenu
                commands={slashCommands.filteredCommands}
                selectedIndex={slashCommands.selectedIndex}
                onSelect={(cmd) => executeSlashCommand(`/${cmd.id}`)}
                onHoverIndex={slashCommands.setSelectedIndex}
              />
            )}
            <box style={{ height: 1, backgroundColor: env.theme.inputBg }} />
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
                    focused={!env.pendingApproval}
                    highlightColor={env.bashMode ? orange[400] : undefined}
                    placeholder={env.pendingApproval ? 'Approve or reject the pending action...' : env.bashMode ? 'Enter a command...' : env.isSubagentView ? `Chat directly with subagent ${selectedSubagentAgentId}...` : env.status === 'streaming' ? 'Type to queue a message...' : 'Chat with the main agent...'}
                    maxHeight={10}
                    minHeight={1}
                  />
                </box>
              </box>
              <box style={{ flexDirection: 'row', justifyContent: 'flex-end' }}>
                {env.bashMode ? (
                  <text style={{ fg: orange[400] }} attributes={TextAttributes.BOLD}>Bash Mode</text>
                ) : env.isSubagentView ? (
                  <>
                    <text style={{ fg: env.theme.muted }}>{env.modelSummary?.provider ?? '—'}</text>
                    <text style={{ fg: env.theme.muted }}> {'\u00b7'} </text>
                    <text style={{ fg: env.theme.foreground }}>{env.modelSummary?.model ?? '—'}</text>
                  </>
                ) : (
                  <>
                    <Button onClick={() => services.openSettings('provider')} onMouseOver={() => setIsProviderHovered(true)} onMouseOut={() => setIsProviderHovered(false)}>
                      <text style={{ fg: isProviderHovered ? env.theme.primary : env.theme.muted }}>{env.modelSummary?.provider ?? 'No provider'}</text>
                    </Button>
                    <text style={{ fg: env.theme.muted }}> {'\u00b7'} </text>
                    <Button onClick={() => services.openSettings('model')} onMouseOver={() => setIsModelHovered(true)} onMouseOut={() => setIsModelHovered(false)}>
                      <text style={{ fg: isModelHovered ? env.theme.primary : env.theme.foreground }}>{env.modelSummary?.model ?? 'No model'}</text>
                    </Button>
                  </>
                )}
              </box>
            </box>
          </box>
        </box>
        <box style={{ height: 1, borderStyle: 'single', border: ['left'], borderColor: env.bashMode ? orange[400] : env.modeColor, customBorderChars: { topLeft: '', bottomLeft: '', topRight: '', bottomRight: '', horizontal: ' ', vertical: '╹', topT: '', bottomT: '', leftT: '', rightT: '', cross: '' } }}>
          <box style={{ height: 1, borderStyle: 'single', border: ['bottom'], borderColor: env.theme.inputBg, customBorderChars: { topLeft: '', bottomLeft: '', topRight: '', bottomRight: '', horizontal: '▀', vertical: ' ', topT: '', bottomT: '', leftT: '', rightT: '', cross: '' } }} />
        </box>
      </box>

      <box style={{ paddingLeft: 2, paddingRight: 2, flexShrink: 0, height: 1, minHeight: 1, maxHeight: 1, flexDirection: 'row', justifyContent: 'space-between' }}>
        <box style={{ flexDirection: 'row', alignItems: 'center', gap: 1 }}>
          {attachments.length > 0 ? (
            <AttachmentsBar attachments={attachments} onRemove={removeAttachment} maxWidth={env.attachmentsMaxWidth} />
          ) : nextEscWillKillAll ? (
            <text style={{ fg: env.theme.secondary }}>Press Esc again to interrupt all subagents</text>
          ) : nextEscWillClearInput ? (
            <text style={{ fg: env.theme.secondary }}>Press Esc again to clear text</text>
          ) : env.nextCtrlCWillExit ? (
            <text style={{ fg: env.theme.secondary }}>Press Ctrl-C again to exit</text>
          ) : env.bashMode ? (
            <text style={{ fg: env.theme.muted }}><span attributes={TextAttributes.BOLD}>Esc</span> to exit Bash mode</text>
          ) : null}
        </box>
        <box style={{ flexDirection: 'row', alignItems: 'center', gap: 1 }}>
          {attachments.length > 0 && (nextEscWillKillAll ? (
            <text style={{ fg: env.theme.secondary }}>Press Esc again to interrupt all subagents</text>
          ) : nextEscWillClearInput ? (
            <text style={{ fg: env.theme.secondary }}>Press Esc again to clear text</text>
          ) : env.nextCtrlCWillExit ? (
            <text style={{ fg: env.theme.secondary }}>Press Ctrl-C again to exit</text>
          ) : env.bashMode ? (
            <text style={{ fg: env.theme.muted }}><span attributes={TextAttributes.BOLD}>Esc</span> to exit Bash mode</text>
          ) : null)}
          {env.tokenEstimate > 0 && (
            <ContextUsageBar
              tokenEstimate={env.tokenEstimate}
              hardCap={env.contextHardCap ?? env.tokenEstimate}
              isCompacting={env.isCompacting}
            />
          )}
        </box>
      </box>

      {pendingKillTab && (
        <box style={{ position: 'absolute', top: 0, left: 0, right: 0, bottom: 0, zIndex: 20, alignItems: 'center', justifyContent: 'center', backgroundColor: RGBA.fromInts(0, 0, 0, 153) }}>
          <box style={{ borderStyle: 'single', border: ['left', 'right', 'top', 'bottom'], borderColor: env.theme.border, backgroundColor: env.theme.surface, customBorderChars: BOX_CHARS, minWidth: 52, maxWidth: 72, paddingLeft: 0, paddingRight: 0, paddingTop: 0, paddingBottom: 0 }}>
            <box style={{ paddingLeft: 2, paddingRight: 2, paddingTop: 1, paddingBottom: 1, flexDirection: 'column' }}>
              <text style={{ fg: env.theme.foreground }}>Kill active subagent {pendingKillTab.agentId}?</text>
              <text style={{ fg: env.theme.muted }}>This removes all subagent progress.</text>
              <box style={{ height: 1 }} />
              <box style={{ flexDirection: 'row', justifyContent: 'flex-end', gap: 1 }}>
                <Button onClick={() => setPendingKillForkId(null)} onMouseOver={() => setIsKillCancelHovered(true)} onMouseOut={() => setIsKillCancelHovered(false)}>
                  <box style={{ borderStyle: 'single', borderColor: isKillCancelHovered ? env.theme.foreground : env.theme.border, customBorderChars: BOX_CHARS, paddingLeft: 1, paddingRight: 1 }}>
                    <text style={{ fg: isKillCancelHovered ? env.theme.foreground : env.theme.muted }}>Cancel</text>
                  </box>
                </Button>
                <Button
                  onClick={() => {
                    services.requestActiveSubagentKill({ forkId: pendingKillTab.forkId, agentId: pendingKillTab.agentId })
                    setPendingKillForkId(null)
                  }}
                  onMouseOver={() => setIsKillConfirmHovered(true)}
                  onMouseOut={() => setIsKillConfirmHovered(false)}
                >
                  <box style={{ borderStyle: 'single', borderColor: isKillConfirmHovered ? env.theme.error : env.theme.border, customBorderChars: BOX_CHARS, paddingLeft: 1, paddingRight: 1 }}>
                    <text style={{ fg: isKillConfirmHovered ? env.theme.error : env.theme.foreground }}>Kill subagent</text>
                  </box>
                </Button>
              </box>
            </box>
          </box>
        </box>
      )}
    </>
  )
}