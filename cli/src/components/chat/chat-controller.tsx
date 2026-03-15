import { TextAttributes, type KeyEvent } from '@opentui/core'
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

export function ChatController({
  env,
  services,
  selectedArtifactOpen,
  onCloseArtifact,
  onApprove,
  onReject,
  onInputHasTextChange,
  restoredQueuedInputText,
  onRestoredQueuedInputHandled,
}: ChatControllerProps) {
  const [inputValue, setInputValue] = useState<InputValue>(EMPTY_INPUT)
  const [attachments, setAttachments] = useState<ImageAttachment[]>([])
  const [nextEscWillKillAll, setNextEscWillKillAll] = useState(false)
  const killAllTimeoutRef = useRef<NodeJS.Timeout | null>(null)
  const [nextEscWillClearInput, setNextEscWillClearInput] = useState(false)
  const clearInputTimeoutRef = useRef<NodeJS.Timeout | null>(null)
  const [isProviderHovered, setIsProviderHovered] = useState(false)
  const [isModelHovered, setIsModelHovered] = useState(false)
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

  const handleInterrupt = useCallback(() => services.interrupt(), [services])
  const handleInterruptAll = useCallback(() => services.interruptAll(), [services])

  const handleKeyIntercept = useCallback((key: KeyEvent): boolean => {
    if (env.pendingApproval) return true
    if (!env.bashMode && fileMentions.handleKeyIntercept(key)) return true
    if (!env.bashMode && slashCommands.handleKeyIntercept(key)) return true
    if (env.widgetNavActive && services.handleWidgetKeyEvent(key)) return true
    return false
  }, [env.pendingApproval, env.bashMode, env.widgetNavActive, fileMentions, slashCommands, services])

  const handleInputChange = useCallback((value: InputValue) => {
    if (!env.bashMode && value.text === '!') {
      services.enterBashMode()
      setInputValue(EMPTY_INPUT)
      return
    }
    if (nextEscWillClearInput) {
      setNextEscWillClearInput(false)
      if (clearInputTimeoutRef.current) clearTimeout(clearInputTimeoutRef.current)
    }
    setInputValue(value)
  }, [env.bashMode, nextEscWillClearInput])

  const handlePaste = useCallback(async (eventText?: string) => {
    const wasClipboardImage = await addImageAttachment()
    if (wasClipboardImage) return
    const pasteText = eventText || readClipboardText()
    if (!pasteText) return
    const wasImagePath = await addImageAttachmentFromFilePath(pasteText)
    if (wasImagePath) return
    setInputValue(prev => {
      if (pasteText.length > INLINE_PASTE_PILL_CHAR_LIMIT) return insertPasteSegment(prev, pasteText, createId())
      return applyTextEditWithPastesAndMentions(prev, prev.cursorPosition, prev.cursorPosition, pasteText)
    })
  }, [addImageAttachment, addImageAttachmentFromFilePath])

  const clearComposer = useCallback(() => {
    setInputValue(EMPTY_INPUT)
    setAttachments([])
  }, [])

  const handleSubmit = useCallback((message: string, visibleMessage?: string, mentionAttachments: Attachment[] = []) => {
    const slashText = visibleMessage ?? message
    if (!env.bashMode && services.runSlashCommand(slashText)) {
      clearComposer()
      return
    }
    if (env.bashMode) {
      const trimmed = message.trim()
      if (!trimmed) return
      const result = services.executeBash(trimmed)
      services.appendBashOutput(result)
      setInputValue(EMPTY_INPUT)
      return
    }
    if (!env.modelsConfigured) return

    services.clearSystemBanners()
    const currentAttachments: Attachment[] = [...attachments, ...mentionAttachments]
    clearComposer()
    services.submitUserMessage({
      message,
      visibleMessage,
      mentionAttachments,
      attachments: currentAttachments,
    })
  }, [env.bashMode, env.modelsConfigured, services, attachments, clearComposer])

  const handleInputSubmit = useCallback(() => {
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
        onClearInput={() => setInputValue(EMPTY_INPUT)}
        selectedArtifactOpen={selectedArtifactOpen}
        onCloseArtifact={onCloseArtifact}
        bashMode={env.bashMode}
        onExitBashMode={() => {
          services.exitBashMode()
          setInputValue(EMPTY_INPUT)
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
          <box style={{ backgroundColor: env.theme.inputBg, paddingTop: 1, paddingLeft: 1, paddingRight: 2, flexDirection: 'column', flexGrow: 1 }}>
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
            {!env.bashMode && slashCommands.isSlashMenuOpen && (
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
                    focused={!env.pendingApproval}
                    highlightColor={env.bashMode ? orange[400] : undefined}
                    placeholder={env.pendingApproval ? 'Approve or reject the pending action...' : env.bashMode ? 'Enter a command...' : env.status === 'streaming' ? 'Type to queue a message...' : 'Type a message...'}
                    maxHeight={10}
                    minHeight={1}
                  />
                </box>
              </box>
              <box style={{ flexDirection: 'row', justifyContent: 'flex-end' }}>
                {env.bashMode ? (
                  <text style={{ fg: orange[400] }} attributes={TextAttributes.BOLD}>Bash Mode</text>
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
    </>
  )
}