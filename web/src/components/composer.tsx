import { Button } from "@/components/ui/button"
import { Textarea } from "@/components/ui/textarea"
import { ActionTooltip } from "@/components/ui/tooltip"

/**
 * Composer — spec §9.6
 *
 * Textarea, submit/stop button, meta row, slash command menu,
 * file mention menu, attachment pills, bash mode.
 */
import { useState, useRef, useCallback, useMemo, type ReactNode } from "react"
import {
  ArrowUp,
  Square,
  FileText,
  Folder,
  X,
  Terminal,
  Sparkles,
} from "lucide-react"
import {
  applyTextEditWithPastesAndMentions,
  insertMentionSegment,
  mentionAttachmentFromSegment,
  mentionOccurrenceFromInputSegment,
  useSlashCommands,
  useFileMentions,
  type InputMentionSegment,
  type InputValue,
  type MentionFileItem,
  type SlashCommandDefinition,
  type MentionSearchClient,
  type SlashCommandOutcome,
  slashCommandUnhandled,
} from "@magnitudedev/client-common"
import {
  useAtomValue,
  useAtomSet,
  useAtomMount,
  Atom,
} from "@effect-atom/atom-react"
import { Effect } from "effect"
import { toGenericKeyEvent, isSendKey, isEscapeKey } from "../utils/keyboard"
import {
  messageHistoryAtom,
  composerTextAtom,
  composerAttachmentsAtom,
  composerUploadsAtom,
  composerHistoryIndexAtom,
} from "@magnitudedev/client-common"
import type { MentionAttachment, RawMessageUpload, RawMentionOccurrence } from "@magnitudedev/sdk"
import { FileCodeIcon, PlusIcon, XIcon } from "@phosphor-icons/react"
import { Spinner } from "@/components/ui/spinner"
import { appendMessageUploads, ingestClientFiles, MESSAGE_UPLOAD_ACCEPT } from "@/lib/message-uploads"
export interface ComposerProps {
  /** Current role label (e.g. "Leader") */
  role?: string
  /** Whether the agent is currently streaming */
  isStreaming?: boolean
  /** Bash mode active */
  bashMode?: boolean
  /** Send a message */
  onSend: (
    text: string,
    mentions?: RawMentionOccurrence[],
    uploads?: RawMessageUpload[],
  ) => void
  /** Surface file-ingestion failures. */
  onAttachmentError?: (message: string) => void
  /** Interrupt the current turn */
  onInterrupt?: () => void
  /** Run a bash command (bash mode) */
  onRunBash?: (command: string) => Promise<boolean>
  /** Execute a slash command */
  onSlashCommand?: (command: string) => SlashCommandOutcome
  /** Toggle bash mode */
  onToggleBashMode?: () => void
  /** File mention confirmation callback */
  onMentionConfirm?: (item: MentionFileItem) => void
  /** Client for file mentions (null if not available) */
  mentionClient?: MentionSearchClient | null
  /** Working directory for file mentions */
  cwd?: string | null
  /** Remove outer margins when the composer is inside the main bottom dock */
  docked?: boolean
  /** Why agent submission is unavailable. Bash commands remain available. */
  disabledReason?: string | null
  /** Navigate to the action that resolves disabled submission. */
  onDisabledAction?: () => void
  /** Runtime controls displayed inside the composer's lower edge. */
  footer?: ReactNode
}

interface PendingClientFile {
  readonly id: string
  readonly filename: string
  readonly image: boolean
}
export function Composer({
  role = "Leader",
  isStreaming = false,
  bashMode = false,
  onSend,
  onAttachmentError,
  onInterrupt,
  onRunBash,
  onSlashCommand,
  onToggleBashMode,
  onMentionConfirm,
  mentionClient,
  cwd = null,
  docked = false,
  disabledReason = null,
  onDisabledAction,
  footer,
}: ComposerProps): ReactNode {
  const text = useAtomValue(composerTextAtom)
  const setText = useAtomSet(composerTextAtom)
  const attachments = useAtomValue(composerAttachmentsAtom)
  const setAttachments = useAtomSet(composerAttachmentsAtom)
  const uploads = useAtomValue(composerUploadsAtom)
  const setUploads = useAtomSet(composerUploadsAtom)
  const historyIndex = useAtomValue(composerHistoryIndexAtom)
  const setHistoryIndex = useAtomSet(composerHistoryIndexAtom)
  const [savedDraft, setSavedDraft] = useState<{
    text: string
    mentions: InputMentionSegment[]
    uploads: RawMessageUpload[]
  }>({
    text: "",
    mentions: [],
    uploads: [],
  })
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const ingestControllersRef = useRef(new Set<AbortController>())
  const [pendingFiles, setPendingFiles] = useState<readonly PendingClientFile[]>([])
  const [dragActive, setDragActive] = useState(false)
  // Track what the user last typed so we can distinguish external restore
  // (queued input / rollback) from normal user input.
  const lastUserTextRef = useRef("")
  const [cursorPosition, setCursorPosition] = useState(0)

  // Cursor/focus restore on external `composerTextAtom` changes (queued input
  // restore, send-failure rollback). useAtomMount — the change originates from
  // the server (display controller), not a user action.
  const restoreFocusAtom = useMemo(
    () =>
      Atom.make(
        Effect.gen(function* () {
          // `text` is captured from useAtomValue; if it differs from what the
          // user last typed, it's an external restore.
          if (text && text !== lastUserTextRef.current && textareaRef.current) {
            if (document.activeElement !== textareaRef.current) {
              textareaRef.current.focus()
              textareaRef.current.setSelectionRange(text.length, text.length)
            }
          }
        })
      ),
    [text]
  )
  useAtomMount(restoreFocusAtom)

  const ingestLifetimeAtom = useMemo(() => Atom.make(Effect.addFinalizer(() => Effect.sync(() => {
      for (const controller of ingestControllersRef.current) controller.abort()
      ingestControllersRef.current.clear()
    }))), [])
  useAtomMount(ingestLifetimeAtom)

  // Message history navigation (spec §14.4: ↑/↓ in composer)
  const messageHistory = useAtomValue(messageHistoryAtom)
  const setMessageHistory = useAtomSet(messageHistoryAtom)

  // Slash commands
  const slashState = useSlashCommands(text, (cmdText: string) => {
    const outcome = onSlashCommand?.(cmdText) ?? slashCommandUnhandled
    if (outcome._tag === "Handled") {
      setText("")
      setAttachments([])
      setUploads([])
      setHistoryIndex(-1)
      setSavedDraft({
        text: "",
        mentions: [],
        uploads: [],
      })
      setCursorPosition(0)
      lastUserTextRef.current = ""
    }
    return outcome
  })

  // File mentions
  const mentionState = useFileMentions({
    inputText: text,
    cursorPosition,
    client: mentionClient ?? null,
    cwd,
    onConfirm: (item: MentionFileItem) => {
      // Insert the mention as @path text
      insertMention(item)
      if (onMentionConfirm) onMentionConfirm(item)
    },
  })
  const insertMention = useCallback(
    (item: MentionFileItem) => {
      const before = text.slice(0, cursorPosition)
      // Replace the @query with @path
      const atIdx = before.lastIndexOf("@")
      if (atIdx === -1) return
      const input: InputValue = {
        text,
        cursorPosition,
        lastEditDueToNav: false,
        pasteSegments: [],
        mentionSegments: attachments,
        selectedPasteSegmentId: null,
        selectedMentionSegmentId: null,
      }
      const next = insertMentionSegment(
        input,
        item,
        crypto.randomUUID(),
        atIdx,
        cursorPosition
      )
      setText(next.text)
      setAttachments(next.mentionSegments)
      lastUserTextRef.current = next.text
      // Move cursor after the inserted text
      const newCursorPos = next.cursorPosition
      setCursorPosition(newCursorPos)
      requestAnimationFrame(() => {
        textareaRef.current?.focus()
        textareaRef.current?.setSelectionRange(newCursorPos, newCursorPos)
      })
    },
    [text, cursorPosition, attachments, setText, setAttachments]
  )
  const handleSubmit = useCallback(async () => {
    const trimmed = text.trim()
    if (!trimmed && attachments.length === 0 && uploads.length === 0) return
    if (bashMode && onRunBash) {
      const didRun = await onRunBash(trimmed)
      if (!didRun) return
      setText("")
      setAttachments([])
      setUploads([])
      lastUserTextRef.current = ""
      return
    }
    if (disabledReason) {
      onDisabledAction?.()
      return
    }
    onSend(
      text,
      attachments.length > 0
        ? attachments.map(mentionOccurrenceFromInputSegment)
        : undefined,
      uploads.length > 0 ? [...uploads] : undefined,
    )
    // Push to message history (most recent first, dedup consecutive)
    setMessageHistory((prev: string[]) =>
      prev[0] === trimmed ? prev : [trimmed, ...prev].slice(0, 100)
    )
    // Reset history navigation
    setHistoryIndex(-1)
    setSavedDraft({
      text: "",
      mentions: [],
      uploads: [],
    })
    setText("")
    setAttachments([])
    setUploads([])
    // Keep lastUserTextRef in sync so the restore-focus Effect doesn't
    // re-focus after submit clears text.
    lastUserTextRef.current = ""
  }, [
    text,
    attachments,
    uploads,
    bashMode,
    onRunBash,
    disabledReason,
    onDisabledAction,
    onSend,
    setMessageHistory,
    setHistoryIndex,
    setText,
    setAttachments,
    setUploads,
  ])
  const hasSendContent = text.trim().length > 0 || attachments.length > 0 || uploads.length > 0
  const canSend = pendingFiles.length === 0 && hasSendContent
  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
      // Update cursor position
      setCursorPosition(e.currentTarget.selectionStart)

      // Slash command menu / file mention menu key handling
      const genericKey = toGenericKeyEvent(e.nativeEvent)
      if (slashState.isSlashMenuOpen) {
        if (slashState.handleKeyIntercept(genericKey)) {
          e.preventDefault()
          return
        }
      }
      if (mentionState.isOpen) {
        if (mentionState.handleKeyIntercept(genericKey)) {
          e.preventDefault()
          return
        }
      }

      // ↑/↓ history navigation (spec §14.4)
      // Up: when at first line or empty, navigate back in history
      // Down: when navigating history, navigate forward; exit at the end
      if (e.key === "ArrowUp" && !e.shiftKey && !e.metaKey && !e.ctrlKey) {
        // Only navigate history when at the first line (cursor at line start)
        const atFirstLine =
          e.currentTarget.selectionStart === 0 ||
          text.slice(0, e.currentTarget.selectionStart).indexOf("\n") === -1
        if (atFirstLine && messageHistory.length > 0) {
          e.preventDefault()
          if (historyIndex === -1) {
            // Entering history mode — save current draft
            setSavedDraft({
              text,
              mentions: attachments,
              uploads: [...uploads],
            })
            setAttachments([])
            setUploads([])
            setHistoryIndex(0)
            const entry = messageHistory[0]
            if (entry !== undefined) {
              setText(entry)
              lastUserTextRef.current = entry
              requestAnimationFrame(() => {
                const ta = textareaRef.current
                if (ta) {
                  ta.setSelectionRange(entry.length, entry.length)
                  resizeTextarea(ta)
                }
              })
            }
          } else if (historyIndex < messageHistory.length - 1) {
            const nextIndex = historyIndex + 1
            setHistoryIndex(nextIndex)
            const entry = messageHistory[nextIndex]
            if (entry !== undefined) {
              setText(entry)
              lastUserTextRef.current = entry
              requestAnimationFrame(() => {
                const ta = textareaRef.current
                if (ta) {
                  ta.setSelectionRange(entry.length, entry.length)
                  resizeTextarea(ta)
                }
              })
            }
          }
          return
        }
      }
      if (e.key === "ArrowDown" && !e.shiftKey && !e.metaKey && !e.ctrlKey) {
        if (historyIndex !== -1) {
          // Only navigate forward when at the last line
          const cursorPos = e.currentTarget.selectionStart
          const afterCursor = text.slice(cursorPos)
          const atLastLine = afterCursor.indexOf("\n") === -1
          if (atLastLine) {
            e.preventDefault()
            if (historyIndex > 0) {
              const nextIndex = historyIndex - 1
              setHistoryIndex(nextIndex)
              const entry = messageHistory[nextIndex]
              if (entry !== undefined) {
                setText(entry)
                lastUserTextRef.current = entry
                requestAnimationFrame(() => {
                  const ta = textareaRef.current
                  if (ta) {
                    ta.setSelectionRange(entry.length, entry.length)
                    resizeTextarea(ta)
                  }
                })
              }
            } else {
              // Exit history mode — restore saved draft
              setHistoryIndex(-1)
              setSavedDraft({
                text: "",
                mentions: [],
                uploads: [],
              })
              setText(savedDraft.text)
              setAttachments(savedDraft.mentions)
              setUploads(savedDraft.uploads)
              lastUserTextRef.current = savedDraft.text
              requestAnimationFrame(() => {
                const ta = textareaRef.current
                if (ta) {
                  ta.setSelectionRange(
                    savedDraft.text.length,
                    savedDraft.text.length
                  )
                }
              })
            }
            return
          }
        }
      }

      // Enter to send
      if (isSendKey(e.nativeEvent)) {
        e.preventDefault()
        if (canSend) {
          handleSubmit()
        } else if (isStreaming && onInterrupt) {
          onInterrupt()
        }
        return
      }

      // Esc to exit bash mode
      if (isEscapeKey(e.nativeEvent) && bashMode && onToggleBashMode) {
        e.preventDefault()
        onToggleBashMode()
        return
      }
    },
    [
      slashState,
      mentionState,
      canSend,
      isStreaming,
      bashMode,
      onInterrupt,
      onToggleBashMode,
      handleSubmit,
      messageHistory,
      historyIndex,
      text,
      attachments,
      savedDraft,
      setText,
      setAttachments,
      setHistoryIndex,
      setSavedDraft,
      uploads,
      setUploads,
    ]
  )
  const handleTextareaChange = useCallback(
    (e: React.ChangeEvent<HTMLTextAreaElement>) => {
      const nextText = e.target.value
      let start = 0
      while (
        start < text.length &&
        start < nextText.length &&
        text[start] === nextText[start]
      )
        start++
      let oldEnd = text.length
      let newEnd = nextText.length
      while (
        oldEnd > start &&
        newEnd > start &&
        text[oldEnd - 1] === nextText[newEnd - 1]
      ) {
        oldEnd--
        newEnd--
      }
      const input: InputValue = {
        text,
        cursorPosition,
        lastEditDueToNav: false,
        pasteSegments: [],
        mentionSegments: attachments,
        selectedPasteSegmentId: null,
        selectedMentionSegmentId: null,
      }
      const next = applyTextEditWithPastesAndMentions(
        input,
        start,
        oldEnd,
        nextText.slice(start, newEnd)
      )
      lastUserTextRef.current = next.text
      setText(next.text)
      setAttachments(next.mentionSegments)
      setCursorPosition(next.cursorPosition)
      resizeTextarea(e.target)
    },
    [text, cursorPosition, attachments, setText, setAttachments]
  )
  const handleTextareaSelect = useCallback(
    (e: React.SyntheticEvent<HTMLTextAreaElement>) => {
      setCursorPosition(e.currentTarget.selectionStart)
    },
    []
  )
  const removeAttachment = useCallback(
    (index: number) => {
      const segment = attachments[index]
      if (!segment) return
      const input: InputValue = {
        text,
        cursorPosition,
        lastEditDueToNav: false,
        pasteSegments: [],
        mentionSegments: attachments,
        selectedPasteSegmentId: null,
        selectedMentionSegmentId: null,
      }
      const next = applyTextEditWithPastesAndMentions(
        input,
        segment.start,
        segment.end,
        ""
      )
      setText(next.text)
      setAttachments(next.mentionSegments)
    },
    [attachments, text, cursorPosition, setText, setAttachments]
  )

  const addClientFiles = useCallback((files: readonly File[]) => {
    if (bashMode || files.length === 0) return
    const pending: PendingClientFile[] = files.map(file => ({
      id: crypto.randomUUID(),
      filename: file.name,
      image: file.type.startsWith("image/"),
    }))
    const pendingIds = new Set<string>(pending.map(item => item.id))
    setPendingFiles(current => [...current, ...pending])

    const controller = new AbortController()
    ingestControllersRef.current.add(controller)
    void Effect.runPromise(ingestClientFiles(files), { signal: controller.signal }).then(results => {
      const accepted = results.flatMap(result => result._tag === "accepted" ? [result.value] : [])
      let capacityRejections: readonly { readonly filename: string; readonly reason: string }[] = []
      if (accepted.length > 0) {
        setUploads(current => {
          const appended = appendMessageUploads(current, accepted)
          capacityRejections = appended.rejected
          return appended.uploads
        })
      }
      for (const result of results) {
        if (result._tag === "rejected") {
          onAttachmentError?.(`${result.error.filename}: ${result.error.reason}`)
        }
      }
      for (const rejection of capacityRejections) {
        onAttachmentError?.(`${rejection.filename}: ${rejection.reason}`)
      }
    }).catch(() => {
      if (!controller.signal.aborted) onAttachmentError?.("The selected files could not be read.")
    }).finally(() => {
      ingestControllersRef.current.delete(controller)
      if (controller.signal.aborted) return
      setPendingFiles(current => current.filter(item => !pendingIds.has(item.id)))
      requestAnimationFrame(() => textareaRef.current?.focus())
    })
  }, [bashMode, onAttachmentError, setUploads])

  const removeUpload = useCallback((index: number) => {
    setUploads(current => current.filter((_, candidate) => candidate !== index))
  }, [setUploads])

  const handleDrop = useCallback((event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault()
    setDragActive(false)
    if (bashMode) return
    addClientFiles(Array.from(event.dataTransfer.files))
  }, [addClientFiles, bashMode])

  const handlePasteFiles = useCallback((event: React.ClipboardEvent<HTMLTextAreaElement>) => {
    const files = Array.from(event.clipboardData.files)
    if (files.length === 0 || bashMode) return
    event.preventDefault()
    addClientFiles(files)
  }, [addClientFiles, bashMode])

  // Placeholder text
  const placeholder = bashMode
    ? "Run a command..."
    : isStreaming
    ? "Type to queue a message..."
    : "Describe a task or ask a question"
  const submitTooltip = pendingFiles.length > 0
    ? "Reading attachments…"
    : !canSend && isStreaming
    ? "Interrupt"
    : disabledReason ?? "Send"
  const submitDisabled = pendingFiles.length > 0 || (!isStreaming && !canSend && !disabledReason)

  return (
    <div
      className={`${docked ? "[margin:0px]" : "[margin:0_12px_4px]"}  composer`}
      data-bash-mode={bashMode}
    >
      <div
        onDragEnter={(event) => {
          event.preventDefault()
          if (!bashMode && event.dataTransfer.types.includes("Files")) setDragActive(true)
        }}
        onDragOver={(event) => {
          if (!bashMode && event.dataTransfer.types.includes("Files")) event.preventDefault()
        }}
        onDragLeave={(event) => {
          if (!event.currentTarget.contains(event.relatedTarget as Node | null)) setDragActive(false)
        }}
        onDrop={handleDrop}
        className={`${
          bashMode
            ? "border-orange-700 dark:border-orange-400"
            : dragActive
            ? "border-blue-500 ring-2 ring-blue-500/20 dark:border-blue-400 dark:ring-blue-400/20"
            : "border-slate-300 dark:border-slate-750"
        } relative rounded-md border bg-white px-3 py-2.5 dark:bg-slate-800 max-[640px]:!p-2`}
      >
        {/* Slash command menu */}
        {slashState.isSlashMenuOpen && (
          <SlashCommandMenu
            commands={slashState.filteredCommands}
            selectedIndex={slashState.selectedIndex}
            onSelectIndex={slashState.setSelectedIndex}
          />
        )}

        {/* File mention menu */}
        {mentionState.isOpen && (
          <FileMentionMenu
            items={mentionState.items}
            recentItems={mentionState.recentItems}
            overflowCount={mentionState.overflowCount}
            selectedIndex={mentionState.selectedIndex}
            onSelectIndex={mentionState.setSelectedIndex}
            loading={mentionState.loading}
          />
        )}

        {(pendingFiles.length > 0 || uploads.length > 0) && (
          <div
            className="mb-2.5 flex h-[72px] gap-2 overflow-x-auto overflow-y-hidden [scrollbar-width:none] [&::-webkit-scrollbar]:hidden"
            aria-label="Attached files"
          >
            {uploads.map((upload, index) => (
              <MessageUploadCard
                key={`${upload.type}:${upload.type === "raw_image_clipboard" ? index : upload.filename}:${index}`}
                upload={upload}
                onRemove={() => removeUpload(index)}
              />
            ))}
            {pendingFiles.map(file => <PendingUploadCard key={file.id} file={file} />)}
          </div>
        )}

        {/* Inline mention pills */}
        {attachments.length > 0 && (
          <div className="flex flex-wrap [gap:4px] [margin-bottom:6px]">
            {attachments.map((att, i) => (
              <AttachmentPill
                key={att.id}
                attachment={mentionAttachmentFromSegment(att)}
                onRemove={() => removeAttachment(i)}
              />
            ))}
          </div>
        )}

        <Textarea
          ref={textareaRef}
          className="composer-textarea field-sizing-fixed box-border min-h-16 max-h-60 w-full resize-none border-0 bg-transparent p-0 pr-[42px] font-sans text-[14px] leading-[1.5] text-slate-900 shadow-none focus-visible:ring-0 dark:bg-transparent dark:text-slate-200"
          value={text}
          aria-label="Message"
          placeholder={placeholder}
          onChange={handleTextareaChange}
          onKeyDown={handleKeyDown}
          onSelect={handleTextareaSelect}
          onPaste={handlePasteFiles}
          onClick={(e) => setCursorPosition(e.currentTarget.selectionStart)}
          rows={3}
        />

        <div className="pl-8">{footer}</div>

        <input
          ref={fileInputRef}
          type="file"
          multiple
          accept={MESSAGE_UPLOAD_ACCEPT}
          className="hidden"
          tabIndex={-1}
          onChange={(event) => {
            addClientFiles(Array.from(event.currentTarget.files ?? []))
            event.currentTarget.value = ""
          }}
        />
        <ActionTooltip
          label="Attach files"
          side="top"
          trigger={(
            <Button
              type="button"
              variant="unstyled"
              size="unstyled"
              disabled={bashMode}
              onClick={() => fileInputRef.current?.click()}
              aria-label="Attach files"
              className="absolute bottom-[10px] left-[10px] flex size-7 items-center justify-center rounded-md text-slate-500 transition-colors hover:bg-slate-150 hover:text-slate-900 focus-visible:bg-slate-150 focus-visible:text-slate-900 disabled:cursor-default disabled:opacity-40 dark:text-slate-400 dark:hover:bg-slate-750 dark:hover:text-slate-100 dark:focus-visible:bg-slate-750 dark:focus-visible:text-slate-100"
            >
              <PlusIcon className="size-[18px]" weight="regular" />
            </Button>
          )}
        />

        {/* Submit / Stop button */}
        <ActionTooltip
          label={submitTooltip}
          side="top"
          trigger={
            <span
              className="absolute bottom-[10px] right-[10px] inline-flex"
              tabIndex={submitDisabled ? 0 : undefined}
              aria-label={submitDisabled ? submitTooltip : undefined}
            >
              <Button variant="unstyled" size="unstyled"
                onClick={() => {
                  if (canSend) handleSubmit()
                  else if (isStreaming && onInterrupt) onInterrupt()
                  else if (disabledReason) onDisabledAction?.()
                }}
                disabled={submitDisabled}
                aria-disabled={!isStreaming && !!disabledReason}
                className={`${
                  isStreaming || canSend || disabledReason
                    ? "cursor-pointer"
                    : "cursor-default"
                } ${
                  isStreaming || canSend ? "opacity-[1]" : "opacity-[0.45]"
                } group flex size-7 items-center justify-center rounded-[4px] border-0 bg-white transition-opacity hover:bg-white dark:bg-slate-850 dark:hover:bg-slate-850`}
                data-can-send={canSend ? "true" : "false"}
                aria-label={
                  !canSend && isStreaming
                    ? "Interrupt"
                    : disabledReason
                    ? `${disabledReason}. Open Settings`
                    : "Send message"
                }
              >
                {!canSend && isStreaming ? (
                  <Square
                    size={16}
                    fill="currentColor"
                    className="text-red-600 dark:text-red-500"
                  />
                ) : (
                  <ArrowUp
                    size={17}
                    strokeWidth={2.4}
                    className={`${
                      canSend && !disabledReason
                        ? "text-blue-700 dark:text-blue-500 group-hover:text-slate-900 dark:group-hover:text-slate-200"
                        : "text-slate-500"
                    } transition-colors duration-100`}
                  />
                )}
              </Button>
            </span>
          }
        />
      </div>
    </div>
  )
}
function resizeTextarea(ta: HTMLTextAreaElement) {
  ta.style.height = "auto"
  ta.style.height = `${Math.min(ta.scrollHeight, 240)}px`
}

function base64ByteSize(data: string): number {
  const padding = data.endsWith("==") ? 2 : data.endsWith("=") ? 1 : 0
  return Math.max(0, Math.floor(data.length * 3 / 4) - padding)
}

function formatFileSize(bytes: number): string {
  if (bytes < 1_000) return `${bytes} B`
  if (bytes < 1_000_000) return `${Math.max(1, Math.round(bytes / 1_000))} KB`
  const megabytes = bytes / 1_000_000
  return `${megabytes >= 10 ? megabytes.toFixed(0) : megabytes.toFixed(1)} MB`
}

function MessageUploadCard({
  upload,
  onRemove,
}: {
  readonly upload: RawMessageUpload
  readonly onRemove: () => void
}): ReactNode {
  const filename = upload.type === "raw_image_clipboard" ? "Clipboard image" : upload.filename
  const image = upload.type !== "raw_text_file"
  const byteSize = base64ByteSize(upload.data)

  return (
    <div
      className="group relative h-[72px] w-36 shrink-0 overflow-hidden rounded-md border border-slate-300 bg-transparent transition-colors hover:border-slate-400 focus-within:border-blue-500 focus-within:ring-2 focus-within:ring-blue-500/20 dark:border-slate-700 dark:hover:border-slate-600 dark:focus-within:border-blue-400 dark:focus-within:ring-blue-400/20"
      aria-label={`${filename}, ${formatFileSize(byteSize)}`}
    >
      {image ? (
        <>
          <img
            src={`data:${upload.mediaType};base64,${upload.data}`}
            alt=""
            className="h-full w-full object-cover"
          />
          <div className="absolute inset-x-0 bottom-0 truncate bg-slate-925/90 px-2 py-1 pr-7 font-sans text-[11px] leading-4 text-slate-100">
            {filename}
          </div>
        </>
      ) : (
        <div className="flex h-full min-w-0 items-center gap-2 px-2.5 pr-8">
          <FileCodeIcon className="size-5 shrink-0 text-slate-500 dark:text-slate-400" aria-hidden="true" />
          <div className="min-w-0 font-sans">
            <div className="truncate text-[12px] font-medium leading-4 text-slate-900 dark:text-slate-100">
              {filename}
            </div>
            <div className="mt-0.5 text-[10px] leading-4 text-slate-500 dark:text-slate-400">
              {formatFileSize(byteSize)}
            </div>
          </div>
        </div>
      )}
      <Button
        type="button"
        variant="unstyled"
        size="unstyled"
        onClick={onRemove}
        aria-label={`Remove ${filename}`}
        className="absolute right-1 top-1 flex size-5 items-center justify-center rounded-full border border-slate-300 bg-white/95 text-slate-600 transition-colors hover:bg-slate-100 hover:text-slate-900 focus-visible:ring-2 focus-visible:ring-blue-500/30 dark:border-slate-600 dark:bg-slate-850/95 dark:text-slate-300 dark:hover:bg-slate-750 dark:hover:text-slate-50"
      >
        <XIcon className="size-3" weight="bold" />
      </Button>
    </div>
  )
}

function PendingUploadCard({ file }: { readonly file: PendingClientFile }): ReactNode {
  return (
    <div
      className="relative flex h-[72px] w-36 shrink-0 items-center gap-2 rounded-md border border-slate-300 bg-transparent px-2.5 dark:border-slate-700"
      aria-label={`Reading ${file.filename}`}
    >
      {file.image ? (
        <div className="flex size-8 shrink-0 items-center justify-center rounded bg-slate-100 dark:bg-slate-750">
          <Spinner className="size-4 text-blue-600 dark:text-blue-400" />
        </div>
      ) : (
        <Spinner className="size-5 shrink-0 text-blue-600 dark:text-blue-400" />
      )}
      <div className="min-w-0 font-sans">
        <div className="truncate text-[12px] font-medium leading-4 text-slate-700 dark:text-slate-200">
          {file.filename}
        </div>
        <div className="mt-0.5 text-[10px] leading-4 text-slate-500 dark:text-slate-400">Reading…</div>
      </div>
    </div>
  )
}

// ── Slash Command Menu ──

function SlashCommandMenu({
  commands,
  selectedIndex,
  onSelectIndex,
}: {
  commands: SlashCommandDefinition[]
  selectedIndex: number
  onSelectIndex: (index: number) => void
}): ReactNode {
  // Find divider between built-ins and skills
  const firstSkillIdx = commands.findIndex((c) => c.source === "skill")
  return (
    <div className="slash-command-menu absolute z-30 rounded-md border border-slate-300 dark:border-slate-750 bg-slate-100 dark:bg-slate-800 shadow-[0_4px_24px_rgba(0,0,0,.4)] absolute [bottom:100%] [left:0px] [right:0px] [max-height:240px] overflow-y-auto [margin-bottom:4px]">
      {commands.map((cmd, i) => (
        <div key={cmd.id}>
          {firstSkillIdx === i && i > 0 && (
            <div className="[height:1px] bg-slate-200 dark:bg-slate-800 [margin:2px_0]" />
          )}
          <div
            className="flex h-8 items-center gap-2 bg-transparent px-2.5 cursor-pointer hover:bg-slate-150 data-[selected=true]:bg-slate-200 dark:hover:bg-slate-750 dark:data-[selected=true]:bg-slate-700"
            data-selected={i === selectedIndex}
            onMouseEnter={() => onSelectIndex(i)}
          >
            {cmd.source === "skill" ? (
              <Sparkles
                size={14}
                className="text-slate-600 dark:text-slate-400 shrink-0"
              />
            ) : (
              <Terminal
                size={14}
                className="text-slate-600 dark:text-slate-400 shrink-0"
              />
            )}
            <span className="text-slate-900 dark:text-slate-200 text-[13px] shrink-0">
              /{cmd.id}
            </span>
            <span className="text-slate-500 text-[11px] [margin-left:auto] min-w-0 [flex:1_1_0] overflow-hidden text-ellipsis whitespace-nowrap">
              {cmd.description}
            </span>
          </div>
        </div>
      ))}
    </div>
  )
}

// ── File Mention Menu ──

function FileMentionMenu({
  items,
  recentItems,
  overflowCount,
  selectedIndex,
  onSelectIndex,
  loading,
}: {
  items: MentionFileItem[]
  recentItems: MentionFileItem[]
  overflowCount: number
  selectedIndex: number
  onSelectIndex: (index: number) => void
  loading: boolean
}): ReactNode {
  const allItems = [...items]
  const recentSlice = recentItems.slice(0, 5)
  const flatItems = [...recentSlice, ...allItems]
  const hasRecent = recentSlice.length > 0
  const recentCount = recentSlice.length
  return (
    <div className="file-mention-menu absolute z-30 rounded-md border border-slate-300 dark:border-slate-750 bg-slate-100 dark:bg-slate-800 shadow-[0_4px_24px_rgba(0,0,0,.4)] absolute [bottom:100%] [left:0px] [right:0px] [max-height:240px] overflow-y-auto [margin-bottom:4px]">
      {loading && flatItems.length === 0 && (
        <div className="[padding:8px_10px] text-slate-500 text-[12px]">
          Loading...
        </div>
      )}

      {hasRecent && (
        <div className="[padding:4px_10px_2px] text-slate-500 text-[11px] font-medium">
          Recent files
        </div>
      )}
      {recentSlice.map((item, i) => (
        <MentionMenuItem
          key={`recent-${item.path}`}
          item={item}
          selected={i === selectedIndex}
          onHover={() => onSelectIndex(i)}
        />
      ))}

      {allItems.length > 0 && (
        <div className="[padding:4px_10px_2px] text-slate-500 text-[11px] font-medium">
          Project files
        </div>
      )}
      {allItems.map((item, i) => (
        <MentionMenuItem
          key={`proj-${item.path}`}
          item={item}
          selected={i + recentCount === selectedIndex}
          onHover={() => onSelectIndex(i + recentCount)}
        />
      ))}

      {overflowCount > 0 && (
        <div className="[padding:4px_10px] text-slate-500 text-[11px]">
          +{overflowCount} more
        </div>
      )}

      {!loading && flatItems.length === 0 && (
        <div className="[padding:8px_10px] text-slate-500 text-[12px]">
          No files found
        </div>
      )}
    </div>
  )
}
function MentionMenuItem({
  item,
  selected,
  onHover,
}: {
  item: MentionFileItem
  selected: boolean
  onHover: () => void
}): ReactNode {
  const Icon = item.kind === "directory" ? Folder : FileText
  return (
    <div
      className="flex h-8 items-center gap-2 bg-transparent px-2.5 cursor-pointer hover:bg-slate-150 data-[selected=true]:bg-slate-200 dark:hover:bg-slate-750 dark:data-[selected=true]:bg-slate-700"
      data-selected={selected}
      onMouseEnter={onHover}
    >
      <Icon size={14} className="text-slate-600 dark:text-slate-400 shrink-0" />
      <span className="text-slate-900 dark:text-slate-200 text-[13px] overflow-hidden text-ellipsis whitespace-nowrap">
        {item.path}
      </span>
      {item.contentType && (
        <span className="text-slate-500 text-[11px] [margin-left:auto] shrink-0">
          {item.contentType}
        </span>
      )}
    </div>
  )
}

// ── Attachment Pill ──

function AttachmentPill({
  attachment,
  onRemove,
}: {
  attachment: MentionAttachment
  onRemove: () => void
}): ReactNode {
  const Icon = attachment.type === "mention_directory" ? Folder : FileText
  const rangeSuffix =
    attachment.type === "mention_file_range"
      ? `:${attachment.startLine}-${attachment.endLine}`
      : ""
  return (
    <span className="inline-flex items-center [gap:4px] bg-white dark:bg-slate-850 border border-slate-300 dark:border-slate-750 rounded-[4px] [padding:2px_6px] text-[11px]">
      <Icon size={14} className="text-slate-600 dark:text-slate-400" />
      <span className="text-slate-900 dark:text-slate-200">
        {attachment.path}
      </span>
      {rangeSuffix && <span className="text-slate-500">{rangeSuffix}</span>}
      <Button variant="unstyled" size="unstyled"
        onClick={onRemove}
        aria-label="Remove attachment"
        className="[background:transparent] border-0 text-slate-500 cursor-pointer [padding:0px] flex"
      >
        <X size={14} />
      </Button>
    </span>
  )
}
