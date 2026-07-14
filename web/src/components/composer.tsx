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
import { useSlashCommands, useFileMentions, type MentionFileItem, type SlashCommandDefinition, type MentionSearchClient, mentionAttachmentFromSegment } from "@magnitudedev/client-common"
import { useAtomValue, useAtomSet, useAtomMount, Atom } from "@effect-atom/atom-react"
import { Effect } from "effect"
import { toGenericKeyEvent, isSendKey, isEscapeKey } from "../utils/keyboard"
import {
  messageHistoryAtom,
  composerTextAtom,
  composerAttachmentsAtom,
  composerHistoryIndexAtom,
} from "@magnitudedev/client-common"
import type { MentionAttachment } from "@magnitudedev/sdk"

export interface ComposerProps {
  /** Current role label (e.g. "Leader") */
  role?: string
  /** Whether the agent is currently streaming */
  isStreaming?: boolean
  /** Bash mode active */
  bashMode?: boolean
  /** Send a message */
  onSend: (text: string, attachments?: MentionAttachment[]) => void
  /** Interrupt the current turn */
  onInterrupt?: () => void
  /** Run a bash command (bash mode) */
  onRunBash?: (command: string) => void
  /** Execute a slash command */
  onSlashCommand?: (command: string) => void
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
}

export function Composer({
  role = "Leader",
  isStreaming = false,
  bashMode = false,
  onSend,
  onInterrupt,
  onRunBash,
  onSlashCommand,
  onToggleBashMode,
  onMentionConfirm,
  mentionClient,
  cwd = null,
  docked = false,
}: ComposerProps): ReactNode {
  const text = useAtomValue(composerTextAtom)
  const setText = useAtomSet(composerTextAtom)
  const attachments = useAtomValue(composerAttachmentsAtom)
  const setAttachments = useAtomSet(composerAttachmentsAtom)
  const historyIndex = useAtomValue(composerHistoryIndexAtom)
  const setHistoryIndex = useAtomSet(composerHistoryIndexAtom)
  const [savedDraft, setSavedDraft] = useState("")
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  // Track what the user last typed so we can distinguish external restore
  // (queued input / rollback) from normal user input.
  const lastUserTextRef = useRef("")

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
        }),
      ),
    [text],
  )
  useAtomMount(restoreFocusAtom)

  // Message history navigation (spec §14.4: ↑/↓ in composer)
  const messageHistory = useAtomValue(messageHistoryAtom)
  const setMessageHistory = useAtomSet(messageHistoryAtom)

  // Slash commands
  const slashState = useSlashCommands(text, (cmdText: string) => {
    if (onSlashCommand) {
      onSlashCommand(cmdText)
    }
    setText("")
    lastUserTextRef.current = ""
  })

  // File mentions
  const [cursorPosition, setCursorPosition] = useState(0)

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
      const after = text.slice(cursorPosition)
      // Replace the @query with @path
      const atIdx = before.lastIndexOf("@")
      if (atIdx === -1) return
      const replacement = `@${item.path}`
      const newText = text.slice(0, atIdx) + replacement + after
      setText(newText)
      lastUserTextRef.current = newText
      // Add as attachment mention
      const attachment = mentionAttachmentFromSegment(item)
      setAttachments((prev) => [...prev, attachment])
      // Move cursor after the inserted text
      const newCursorPos = atIdx + replacement.length
      setCursorPosition(newCursorPos)
      requestAnimationFrame(() => {
        textareaRef.current?.focus()
        textareaRef.current?.setSelectionRange(newCursorPos, newCursorPos)
      })
    },
    [text, cursorPosition, setText, setAttachments],
  )

  const handleSubmit = useCallback(() => {
    const trimmed = text.trim()
    if (!trimmed && attachments.length === 0) return

    if (bashMode && onRunBash) {
      onRunBash(trimmed)
      setText("")
      setAttachments([])
      lastUserTextRef.current = ""
      return
    }

    onSend(trimmed, attachments.length > 0 ? attachments : undefined)
    // Push to message history (most recent first, dedup consecutive)
    setMessageHistory((prev: string[]) =>
      prev[0] === trimmed ? prev : [trimmed, ...prev].slice(0, 100),
    )
    // Reset history navigation
    setHistoryIndex(-1)
    setSavedDraft("")
    setText("")
    setAttachments([])
    // Keep lastUserTextRef in sync so the restore-focus Effect doesn't
    // re-focus after submit clears text.
    lastUserTextRef.current = ""
  }, [text, attachments, bashMode, onRunBash, onSend, setMessageHistory, setHistoryIndex, setText, setAttachments])

  const canSend = text.trim().length > 0 || attachments.length > 0

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
        const atFirstLine = e.currentTarget.selectionStart === 0
          || text.slice(0, e.currentTarget.selectionStart).indexOf("\n") === -1
        if (atFirstLine && messageHistory.length > 0) {
          e.preventDefault()
          if (historyIndex === -1) {
            // Entering history mode — save current draft
            setSavedDraft(text)
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
              setSavedDraft("")
              setText(savedDraft)
              lastUserTextRef.current = savedDraft
              requestAnimationFrame(() => {
                const ta = textareaRef.current
                if (ta) {
                  ta.setSelectionRange(savedDraft.length, savedDraft.length)
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
    [slashState, mentionState, canSend, isStreaming, bashMode, onInterrupt, onToggleBashMode, handleSubmit, messageHistory, historyIndex, text, savedDraft, setText, setHistoryIndex, setSavedDraft],
  )

  const handleTextareaChange = useCallback(
    (e: React.ChangeEvent<HTMLTextAreaElement>) => {
      lastUserTextRef.current = e.target.value
      setText(e.target.value)
      setCursorPosition(e.target.selectionStart)
      resizeTextarea(e.target)
    },
    [setText],
  )

  const handleTextareaSelect = useCallback(
    (e: React.SyntheticEvent<HTMLTextAreaElement>) => {
      setCursorPosition(e.currentTarget.selectionStart)
    },
    [],
  )

  const removeAttachment = useCallback((index: number) => {
    setAttachments((prev) => prev.filter((_, i) => i !== index))
  }, [setAttachments])

  // Placeholder text
  const placeholder = bashMode
    ? "Run a command..."
    : isStreaming
      ? "Type to queue a message..."
      : "Describe a task or ask a question"

  // Left accent color
  const accentColor = bashMode
    ? "var(--line-bash)"
    : "var(--accent-primary)"

  return (
    <div className="composer" data-bash-mode={bashMode} style={{ margin: docked ? 0 : "0 12px 4px" }}>
      <div
        className="composer-box"
        style={{
          position: "relative",
          background: "var(--bg-input)",
          border: `1px solid ${bashMode ? accentColor : "var(--border-default)"}`,
          borderRadius: 6,
          padding: "10px 12px",
        }}
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

        {/* Attachment pills */}
        {attachments.length > 0 && (
          <div style={{ display: "flex", flexWrap: "wrap", gap: 4, marginBottom: 6 }}>
            {attachments.map((att, i) => (
              <AttachmentPill key={i} attachment={att} onRemove={() => removeAttachment(i)} />
            ))}
          </div>
        )}

        <textarea
          ref={textareaRef}
          className="composer-textarea"
          value={text}
          placeholder={placeholder}
          onChange={handleTextareaChange}
          onKeyDown={handleKeyDown}
          onSelect={handleTextareaSelect}
          onClick={(e) => setCursorPosition(e.currentTarget.selectionStart)}
          style={{
            width: "100%",
            boxSizing: "border-box",
            paddingRight: 42,
            fontFamily: "var(--font-sans)",
            fontSize: 14,
            lineHeight: 1.5,
            color: "var(--fg-primary)",
            background: "transparent",
            border: "none",
            outline: "none",
            resize: "none",
            minHeight: 64,
            maxHeight: 240,
          }}
          data-placeholder-color="var(--fg-placeholder)"
          rows={3}
        />

        {/* Submit / Stop button */}
        <button
          onClick={() => {
            if (canSend) handleSubmit()
            else if (isStreaming && onInterrupt) onInterrupt()
          }}
          disabled={!isStreaming && !canSend}
          className="composer-send-button"
          data-can-send={canSend ? "true" : "false"}
          style={{
            position: "absolute",
            right: 10,
            bottom: 10,
            width: 28,
            height: 28,
            borderRadius: 4,
            border: "none",
            cursor: isStreaming || canSend ? "pointer" : "default",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            opacity: isStreaming || canSend ? 1 : 0.45,
            transition: "opacity 100ms",
          }}
          title={!canSend && isStreaming ? "Interrupt" : "Send"}
          aria-label={!canSend && isStreaming ? "Interrupt" : "Send message"}
        >
          {!canSend && isStreaming ? (
            <Square size={16} fill="currentColor" style={{ color: "var(--accent-error)" }} />
          ) : (
            <ArrowUp
              size={17}
              strokeWidth={2.4}
              className="composer-send-arrow"
              style={{ color: canSend ? "var(--accent-primary)" : "var(--fg-tertiary)" }}
            />
          )}
        </button>
      </div>
    </div>
  )
}

function resizeTextarea(ta: HTMLTextAreaElement) {
  ta.style.height = "auto"
  ta.style.height = `${Math.min(ta.scrollHeight, 240)}px`
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
    <div
      className="slash-command-menu popover"
      style={{
        position: "absolute",
        bottom: "100%",
        left: 0,
        right: 0,
        maxHeight: 240,
        overflowY: "auto",
        marginBottom: 4,
      }}
    >
      {commands.map((cmd, i) => (
        <div key={cmd.id}>
          {firstSkillIdx === i && i > 0 && (
            <div style={{ height: 1, background: "var(--border-subtle)", margin: "2px 0" }} />
          )}
          <div
            className="menu-item"
            data-selected={i === selectedIndex}
            onMouseEnter={() => onSelectIndex(i)}
          >
            {cmd.source === "skill" ? (
              <Sparkles size={14} style={{ color: "var(--fg-secondary)", flexShrink: 0 }} />
            ) : (
              <Terminal size={14} style={{ color: "var(--fg-secondary)", flexShrink: 0 }} />
            )}
            <span style={{ color: "var(--fg-primary)", fontSize: 13, flexShrink: 0 }}>
              /{cmd.id}
            </span>
            <span
              style={{
                color: "var(--fg-tertiary)",
                fontSize: 11,
                marginLeft: "auto",
                overflow: "hidden",
                textOverflow: "ellipsis",
                whiteSpace: "nowrap",
              }}
            >
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
    <div
      className="file-mention-menu popover"
      style={{
        position: "absolute",
        bottom: "100%",
        left: 0,
        right: 0,
        maxHeight: 240,
        overflowY: "auto",
        marginBottom: 4,
      }}
    >
      {loading && flatItems.length === 0 && (
        <div style={{ padding: "8px 10px", color: "var(--fg-tertiary)", fontSize: 12 }}>
          Loading...
        </div>
      )}

      {hasRecent && (
        <div style={{ padding: "4px 10px 2px", color: "var(--fg-tertiary)", fontSize: 11, fontWeight: 500 }}>
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
        <div style={{ padding: "4px 10px 2px", color: "var(--fg-tertiary)", fontSize: 11, fontWeight: 500 }}>
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
        <div style={{ padding: "4px 10px", color: "var(--fg-tertiary)", fontSize: 11 }}>
          +{overflowCount} more
        </div>
      )}

      {!loading && flatItems.length === 0 && (
        <div style={{ padding: "8px 10px", color: "var(--fg-tertiary)", fontSize: 12 }}>
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
      className="menu-item"
      data-selected={selected}
      onMouseEnter={onHover}
    >
      <Icon size={14} style={{ color: "var(--fg-secondary)", flexShrink: 0 }} />
      <span
        style={{
          color: "var(--fg-primary)",
          fontSize: 13,
          overflow: "hidden",
          textOverflow: "ellipsis",
          whiteSpace: "nowrap",
        }}
      >
        {item.path}
      </span>
      {item.contentType && (
        <span
          style={{
            color: "var(--fg-tertiary)",
            fontSize: 11,
            marginLeft: "auto",
            flexShrink: 0,
          }}
        >
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
    attachment.type === "mention_file_range" ? `:${attachment.startLine}-${attachment.endLine}` : ""

  return (
    <span
      style={{
        display: "inline-flex",
        alignItems: "center",
        gap: 4,
        background: "var(--bg-surface)",
        border: "1px solid var(--border-default)",
        borderRadius: 4,
        padding: "2px 6px",
        fontSize: 11,
      }}
    >
      <Icon size={14} style={{ color: "var(--fg-secondary)" }} />
      <span style={{ color: "var(--fg-primary)" }}>{attachment.path}</span>
      {rangeSuffix && <span style={{ color: "var(--fg-tertiary)" }}>{rangeSuffix}</span>}
      <button
        onClick={onRemove}
        aria-label="Remove attachment"
        style={{
          background: "transparent",
          border: "none",
          color: "var(--fg-tertiary)",
          cursor: "pointer",
          padding: 0,
          display: "flex",
        }}
      >
        <X size={14} />
      </button>
    </span>
  )
}
