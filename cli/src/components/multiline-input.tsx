import { TextAttributes } from '@opentui/core'
import { logger } from '@magnitudedev/logger'
import { useKeyboard, useRenderer } from '@opentui/react'
import {
  forwardRef,
  useCallback,
  useEffect,
  useImperativeHandle,
  useRef,
  useState,
} from 'react'

import {
  CONTROL_CHAR_REGEX,
  locateLineEnd,
  locateLineStart,
  findWordStartBefore,
  findWordEndAfter,
  hasAltStyleModifier,
  isLikelyPrintableKey,


  TAB_WIDTH,
} from './multiline-input.helpers'
import { useTheme } from '../hooks/use-theme'

import { terminalSupportsRgb24 } from '../utils/theme'
import { stepCursorVertical } from './multiline-input.helpers'

import type { InputValue } from '../types/store'
import type {
  KeyEvent,
  MouseEvent,
  ScrollBoxRenderable,
  TextBufferView,
  TextRenderable,
} from '@opentui/core'

type KeyWithPreventDefault =
  | {
      preventDefault?: () => void
    }
  | null
  | undefined

function suppressKeyDefault(key: KeyWithPreventDefault) {
  key?.preventDefault?.()
}

function renderIndexToSourceIndex(text: string, renderPos: number): number {
  let sourcePos = 0
  let currentRenderPos = 0

  while (sourcePos < text.length && currentRenderPos < renderPos) {
    currentRenderPos += text[sourcePos] === '\t' ? TAB_WIDTH : 1
    sourcePos++
  }

  return Math.min(sourcePos, text.length)
}



export { INPUT_CURSOR_CHAR } from './multiline-input.helpers'

interface CursorIndicatorProps {
  visible: boolean
  focused: boolean
  shouldBlink?: boolean
  char?: string
  color?: string
  blinkDelay?: number
  blinkInterval?: number
  bold?: boolean
}

export function InputCursor({
  visible,
  focused,
  shouldBlink = true,
  char = '▍',
  color,
  blinkDelay = 500,
  blinkInterval = 500,
  bold = true,
}: CursorIndicatorProps) {
  // false = normal/visible, true = invisible
  const [isBlinkHidden, setIsInvisible] = useState(false)
  const blinkIntervalTimerRef = useRef<NodeJS.Timeout | null>(null)

  // Handle blinking (toggle visible/invisible) when idle
  useEffect(() => {
    // Clear any existing interval
    if (blinkIntervalTimerRef.current) {
      clearInterval(blinkIntervalTimerRef.current)
      blinkIntervalTimerRef.current = null
    }

    // Reset cursor to visible
    setIsInvisible(false)

    // Only blink if shouldBlink is enabled, focused, and visible
    if (!shouldBlink || !focused || !visible) return

    // Set up idle detection
    const blinkStartTimer = setTimeout(() => {
      // Start blinking interval (toggle between visible and invisible)
      blinkIntervalTimerRef.current = setInterval(() => {
        setIsInvisible((prev) => !prev)
      }, blinkInterval)
    }, blinkDelay)

    return () => {
      clearTimeout(blinkStartTimer)
      if (blinkIntervalTimerRef.current) {
        clearInterval(blinkIntervalTimerRef.current)
        blinkIntervalTimerRef.current = null
      }
    }
  }, [visible, focused, shouldBlink, blinkDelay, blinkInterval])

  if (!visible || !focused) {
    return null
  }

  // When invisible, return a space to maintain layout
  if (isBlinkHidden) {
    return <span> </span>
  }

  return (
    <span
      {...(color ? { fg: color } : undefined)}
      {...(bold ? { attributes: TextAttributes.BOLD } : undefined)}
    >
      {char}
    </span>
  )
}

// Helper type for scrollbox with focus/blur methods (not exposed in OpenTUI types but available at runtime)
interface FocusableScrollBox {
  focus?: () => void
  blur?: () => void
}

interface MultilineInputProps {
  value: string
  onChange: (value: InputValue) => void
  onSubmit: () => void
  onKeyIntercept?: (key: KeyEvent) => boolean
  onPaste: (fallbackText?: string) => void
  placeholder?: string
  focused?: boolean
  shouldBlinkCursor?: boolean
  maxHeight?: number
  minHeight?: number
  cursorPosition: number
  showScrollbar?: boolean
  highlightColor?: string
}

export type MultilineInputHandle = {
  focus: () => void
  blur: () => void
}

export const MultilineInput = forwardRef<
  MultilineInputHandle,
  MultilineInputProps
>(function MultilineInput(
  {
    value,
    onChange,
    onSubmit,
    onPaste,
    placeholder = '',
    focused = true,
    shouldBlinkCursor,
    maxHeight = 5,
    minHeight = 1,
    onKeyIntercept,
    cursorPosition,
    showScrollbar = false,
    highlightColor,
  }: MultilineInputProps,
  forwardedRef,
) {
  const theme = useTheme()
  const renderer = useRenderer()
  const effectiveShouldBlinkCursor = shouldBlinkCursor ?? true

  const scrollBoxRef = useRef<ScrollBoxRenderable | null>(null)
  const [lastActivity, setLastActivity] = useState(Date.now())

  const stickyColumnRef = useRef<number | null>(null)

  // Refs to track latest value and cursor position synchronously for IME input handling.
  // When IME sends multiple character events rapidly (e.g., Chinese input), React batches
  // state updates, causing subsequent events to see stale closure values. These refs are
  // updated synchronously to ensure each keystroke builds on the previous one.
  const valueRef = useRef(value)
  const cursorPositionRef = useRef(cursorPosition)

  // Keep refs current on every render (synchronous assignment avoids useEffect timing issues)
  valueRef.current = value
  cursorPositionRef.current = cursorPosition

  // Helper to get or set the sticky column for vertical navigation.
  // When stickyColumnRef.current is set, we return it (preserving column across
  // multiple up/down presses). When null, we calculate from current cursor position.
  const resolveStickyColumn = useCallback(
    (lineStarts: number[], cursorIsChar: boolean): number => {
      if (stickyColumnRef.current != null) {
        return stickyColumnRef.current
      }
      const lineIndex = lineStarts.findLastIndex(
        (lineStart) => lineStart <= cursorPosition,
      )
      const column =
        lineIndex === -1
          ? 0
          : cursorPosition - lineStarts[lineIndex] + (cursorIsChar ? -1 : 0)
      stickyColumnRef.current = Math.max(0, column)
      return stickyColumnRef.current
    },
    [cursorPosition],
  )

  // Update last activity on value or cursor changes
  useEffect(() => {
    setLastActivity(Date.now())
  }, [value, cursorPosition])

  const textRef = useRef<TextRenderable | null>(null)

  const lineInfo = textRef.current
    ? (
        (textRef.current satisfies TextRenderable as any)
          .textBufferView as TextBufferView
      ).lineInfo
    : null

  // Focus/blur scrollbox when focused prop changes
  const prevFocusedRef = useRef(false)
  useEffect(() => {
    if (focused && !prevFocusedRef.current) {
      (scrollBoxRef.current as FocusableScrollBox | null)?.focus?.()
    } else if (!focused && prevFocusedRef.current) {
      (scrollBoxRef.current as FocusableScrollBox | null)?.blur?.()
    }
    prevFocusedRef.current = focused
  }, [focused])

  // Expose focus/blur for imperative use cases
  useImperativeHandle(
    forwardedRef,
    () => ({
      focus: () => {
        (scrollBoxRef.current as FocusableScrollBox | null)?.focus?.()
      },
      blur: () => {
        (scrollBoxRef.current as FocusableScrollBox | null)?.blur?.()
      },
    }),
    [],
  )

  const cursorRow = lineInfo
    ? Math.max(
        0,
        lineInfo.lineStarts.findLastIndex(
          (lineStart) => lineStart <= cursorPosition,
        ),
      )
    : 0

  // Auto-scroll to cursor when content changes
  useEffect(() => {
    const scrollBox = scrollBoxRef.current
    if (scrollBox && focused) {
      const scrollPosition = Math.min(
        Math.max(
          scrollBox.verticalScrollBar.scrollPosition,
          Math.max(0, cursorRow - scrollBox.viewport.height + 1),
        ),
        Math.min(scrollBox.scrollHeight - scrollBox.viewport.height, cursorRow),
      )

      scrollBox.verticalScrollBar.scrollPosition = scrollPosition
    }
  }, [scrollBoxRef.current, cursorPosition, focused, cursorRow])

  // Helper to get current selection in original text coordinates
  const readSelectedRange = useCallback((): { start: number; end: number } | null => {
    const textBufferView = (textRef.current as any)?.textBufferView
    if (!textBufferView?.hasSelection?.() || !textBufferView?.getSelection) {
      return null
    }
    const selection = textBufferView.getSelection()
    if (!selection) return null

    // Convert from render positions to original text positions
    const start = renderIndexToSourceIndex(value, Math.min(selection.start, selection.end))
    const end = renderIndexToSourceIndex(value, Math.max(selection.start, selection.end))

    if (start === end) return null
    return { start, end }
  }, [value])

  // Helper to clear the current selection
  const dismissSelection = useCallback(() => {
    // Use renderer's clearSelection for proper visual clearing
    ;(renderer as any)?.clearSelection?.()
  }, [renderer])

  // Helper to delete selected text and return new value and cursor position
  const cutSelectionRange = useCallback((): { newValue: string; newCursor: number } | null => {
    const selection = readSelectedRange()
    if (!selection) return null

    const newValue = value.slice(0, selection.start) + value.slice(selection.end)
    dismissSelection()
    return { newValue, newCursor: selection.start }
  }, [value, readSelectedRange, dismissSelection])

  // Helper to handle selection deletion and call onChange if selection existed
  // Returns true if selection was deleted, false otherwise
  const removeSelectionIfPresent = useCallback((): boolean => {
    const deleted = cutSelectionRange()
    if (deleted) {
      onChange({
        text: deleted.newValue,
        cursorPosition: deleted.newCursor,
        lastEditDueToNav: false,
      })
      return true
    }
    return false
  }, [cutSelectionRange, onChange])

  const insertAtCaret = useCallback(
    (textToInsert: string) => {
      if (!textToInsert) return

      // Check if there's a selection to replace
      const selection = readSelectedRange()
      if (selection) {
        // Replace selected text with the new text
        dismissSelection()
        // Read from refs which have the latest values (updated synchronously below)
        const currentValue = valueRef.current
        const newValue =
          currentValue.slice(0, selection.start) +
          textToInsert +
          currentValue.slice(selection.end)
        const newCursor = selection.start + textToInsert.length

        // Update refs synchronously BEFORE calling onChange - critical for IME input
        // where multiple characters may arrive before React processes state updates
        valueRef.current = newValue
        cursorPositionRef.current = newCursor

        onChange({
          text: newValue,
          cursorPosition: newCursor,
          lastEditDueToNav: false,
        })
        return
      }

      // No selection, insert at cursor
      // Read from refs to get latest state (handles rapid IME input)
      const currentValue = valueRef.current
      const currentCursor = cursorPositionRef.current
      const newValue =
        currentValue.slice(0, currentCursor) +
        textToInsert +
        currentValue.slice(currentCursor)
      const newCursor = currentCursor + textToInsert.length

      // Update refs synchronously BEFORE calling onChange - critical for IME input
      // where multiple characters may arrive before React processes state updates
      valueRef.current = newValue
      cursorPositionRef.current = newCursor

      onChange({
        text: newValue,
        cursorPosition: newCursor,
        lastEditDueToNav: false,
      })
    },
    [onChange, readSelectedRange, dismissSelection],
  )

  const moveCursorTo = useCallback(
    (nextPosition: number) => {
      const clamped = Math.max(0, Math.min(value.length, nextPosition))
      if (clamped === cursorPosition) return
      onChange({
        text: value,
        cursorPosition: clamped,
        lastEditDueToNav: false,
      })
    },
    [cursorPosition, onChange, value],
  )

  // Handle mouse clicks to position cursor
  const handleMouseDown = useCallback(
    (event: MouseEvent) => {
      if (!focused) return

      // Clear sticky column since this is not up/down navigation
      stickyColumnRef.current = null

      const scrollBox = scrollBoxRef.current
      if (!scrollBox) return

      const lineStarts = lineInfo?.lineStarts ?? [0]

      const viewport = (scrollBox as any).viewport
      const viewportTop = Number(viewport?.y ?? 0)
      const viewportLeft = Number(viewport?.x ?? 0)

      // Get click position, accounting for scroll
      const scrollPosition = scrollBox.verticalScrollBar?.scrollPosition ?? 0
      const clickRowInViewport = Math.floor(event.y - viewportTop)
      const clickRow = clickRowInViewport + scrollPosition

      // Find which visual line was clicked
      const lineIndex = Math.min(
        Math.max(0, clickRow),
        lineStarts.length - 1,
      )

      // Get the character range for this line
      const lineStartChar = lineStarts[lineIndex]
      const lineEndChar = lineStarts[lineIndex + 1] ?? value.length

      // Convert click x to character position, accounting for tabs
      const clickCol = Math.max(0, Math.floor(event.x - viewportLeft))

      let visualCol = 0
      let charIndex = lineStartChar

      while (charIndex < lineEndChar && visualCol < clickCol) {
        const char = value[charIndex]
        if (char === '\t') {
          visualCol += TAB_WIDTH
        } else if (char === '\n') {
          break
        } else {
          visualCol += 1
        }
        charIndex++
      }

      // Clamp to valid range
      const newCursorPosition = Math.min(charIndex, value.length)

      // Update cursor position if changed
      if (newCursorPosition !== cursorPosition) {
        onChange({
          text: value,
          cursorPosition: newCursorPosition,
          lastEditDueToNav: false,
        })
      }
    },
    [focused, lineInfo, value, cursorPosition, onChange],
  )

  const isPlaceholder = value.length === 0 && placeholder.length > 0
  const displayValue = isPlaceholder ? placeholder : value
  const showCursor = focused

  // Replace tabs with spaces for proper rendering
  const displayValueForRendering = displayValue.replace(
    /\t/g,
    ' '.repeat(TAB_WIDTH),
  )

  // Calculate cursor position in the expanded string (accounting for tabs)
  let renderCursorPosition = 0
  for (let i = 0; i < cursorPosition && i < displayValue.length; i++) {
    renderCursorPosition += displayValue[i] === '\t' ? TAB_WIDTH : 1
  }

  const { beforeCursor, afterCursor, activeChar, shouldHighlight } = (() => {
    if (!showCursor) {
      return {
        beforeCursor: '',
        afterCursor: '',
        activeChar: ' ',
        shouldHighlight: false,
      }
    }

    const beforeCursor = displayValueForRendering.slice(0, renderCursorPosition)
    const afterCursor = displayValueForRendering.slice(renderCursorPosition)
    const activeChar = afterCursor.charAt(0) || ' '
    const shouldHighlight =
      !isPlaceholder &&
      renderCursorPosition < displayValueForRendering.length &&
      displayValue[cursorPosition] !== '\n' &&
      displayValue[cursorPosition] !== '\t'

    return {
      beforeCursor,
      afterCursor,
      activeChar,
      shouldHighlight,
    }
  })()

  // --- Keyboard Handler Helpers ---

  // Handle enter/newline keys
  const processEnterKey = useCallback(
    (key: KeyEvent): boolean => {
      const lowerKeyName = (key.name ?? '').toLowerCase()
      const isEnterKey = key.name === 'return' || key.name === 'enter'
      // Ctrl+J is translated by the terminal to a linefeed character (0x0a)
      // So we detect it by checking for name === 'linefeed' rather than ctrl + j
      const isCtrlJ =
        lowerKeyName === 'linefeed' ||
        (key.ctrl &&
          !key.meta &&
          !key.option &&
          lowerKeyName === 'j')

      // Only handle Enter and Ctrl+J here
      if (!isEnterKey && !isCtrlJ) return false

      const hasAltLikeModifier = hasAltStyleModifier(key)
      const hasEscapePrefix =
        typeof key.sequence === 'string' &&
        key.sequence.length > 0 &&
        key.sequence.charCodeAt(0) === 0x1b
      const hasBackslashBeforeCursor =
        cursorPosition > 0 && value[cursorPosition - 1] === '\\'

      // Plain Enter: no modifiers, sequence is '\r' (macOS) or '\n' (Linux)
      const isPlainEnter =
        isEnterKey &&
        !key.shift &&
        !key.ctrl &&
        !key.meta &&
        !key.option &&
        !hasAltLikeModifier &&
        !hasEscapePrefix &&
        (key.sequence === '\r' || key.sequence === '\n') &&
        !hasBackslashBeforeCursor
      const isShiftEnter = isEnterKey && Boolean(key.shift)
      const isOptionEnter =
        isEnterKey && (hasAltLikeModifier || hasEscapePrefix)
      const isBackslashEnter = isEnterKey && hasBackslashBeforeCursor

      const shouldInsertNewline =
        isCtrlJ || isShiftEnter || isOptionEnter || isBackslashEnter

      if (shouldInsertNewline) {
        suppressKeyDefault(key)

        // For backslash+Enter, remove the backslash and insert newline
        if (isBackslashEnter) {
          const newValue =
            value.slice(0, cursorPosition - 1) +
            '\n' +
            value.slice(cursorPosition)
          onChange({
            text: newValue,
            cursorPosition,
            lastEditDueToNav: false,
          })
          return true
        }

        // For other newline shortcuts (Shift+Enter, Option+Enter, Ctrl+J), just insert newline
        const newValue =
          value.slice(0, cursorPosition) + '\n' + value.slice(cursorPosition)
        onChange({
          text: newValue,
          cursorPosition: cursorPosition + 1,
          lastEditDueToNav: false,
        })
        return true
      }

      if (isPlainEnter) {
        suppressKeyDefault(key)
        onSubmit()
        return true
      }

      return false
    },
    [value, cursorPosition, onChange, onSubmit],
  )

  // Handle deletion keys (backspace, delete, word/line deletion)
  const processDeletionKey = useCallback(
    (key: KeyEvent): boolean => {
      const hasAltLikeModifier = hasAltStyleModifier(key)
      const lineStart = locateLineStart(value, cursorPosition)
      const wordStart = findWordStartBefore(value, cursorPosition)
      const wordEnd = findWordEndAfter(value, cursorPosition)

      // Alt+Backspace: Delete word backward
      if (key.name === 'backspace' && hasAltLikeModifier) {
        suppressKeyDefault(key)
        if (removeSelectionIfPresent()) return true
        const newValue =
          value.slice(0, wordStart) + value.slice(cursorPosition)
        onChange({
          text: newValue,
          cursorPosition: wordStart,
          lastEditDueToNav: false,
        })
        return true
      }

      // Cmd+Delete: Delete to line start
      if (key.name === 'delete' && key.meta && !hasAltLikeModifier) {
        suppressKeyDefault(key)
        if (removeSelectionIfPresent()) return true
        const originalValue = value
        let newValue = originalValue
        let nextCursor = cursorPosition

        if (cursorPosition > 0) {
          if (
            cursorPosition === lineStart &&
            value[cursorPosition - 1] === '\n'
          ) {
            newValue =
              value.slice(0, cursorPosition - 1) + value.slice(cursorPosition)
            nextCursor = cursorPosition - 1
          } else {
            newValue = value.slice(0, lineStart) + value.slice(cursorPosition)
            nextCursor = lineStart
          }
        }

        if (newValue === originalValue && cursorPosition > 0) {
          newValue =
            value.slice(0, cursorPosition - 1) + value.slice(cursorPosition)
          nextCursor = cursorPosition - 1
        }

        if (newValue !== originalValue) {
          onChange({
            text: newValue,
            cursorPosition: nextCursor,
            lastEditDueToNav: false,
          })
        }
        return true
      }

      // Alt+Delete: Delete word forward
      if (key.name === 'delete' && hasAltLikeModifier) {
        suppressKeyDefault(key)
        if (removeSelectionIfPresent()) return true
        const newValue = value.slice(0, cursorPosition) + value.slice(wordEnd)
        onChange({
          text: newValue,
          cursorPosition,
          lastEditDueToNav: false,
        })
        return true
      }

      // Basic Backspace (no modifiers)
      if (key.name === 'backspace' && !key.ctrl && !key.meta && !key.option) {
        suppressKeyDefault(key)
        if (removeSelectionIfPresent()) return true
        if (cursorPosition > 0) {
          const newValue =
            value.slice(0, cursorPosition - 1) + value.slice(cursorPosition)
          onChange({
            text: newValue,
            cursorPosition: cursorPosition - 1,
            lastEditDueToNav: false,
          })
        }
        return true
      }

      // Basic Delete (no modifiers)
      if (key.name === 'delete' && !key.ctrl && !key.meta && !key.option) {
        suppressKeyDefault(key)
        if (removeSelectionIfPresent()) return true
        if (cursorPosition < value.length) {
          const newValue =
            value.slice(0, cursorPosition) + value.slice(cursorPosition + 1)
          onChange({
            text: newValue,
            cursorPosition,
            lastEditDueToNav: false,
          })
        }
        return true
      }

      return false
    },
    [value, cursorPosition, onChange, removeSelectionIfPresent],
  )

  // Handle navigation keys (arrows, home, end, word navigation)
  const processNavigationKey = useCallback(
    (key: KeyEvent): boolean => {
      const lowerKeyName = (key.name ?? '').toLowerCase()
      const hasAltLikeModifier = hasAltStyleModifier(key)
      const logicalLineStart = locateLineStart(value, cursorPosition)
      const logicalLineEnd = locateLineEnd(value, cursorPosition)
      const wordStart = findWordStartBefore(value, cursorPosition)
      const wordEnd = findWordEndAfter(value, cursorPosition)

      // Read lineInfo inside the callback to get current value (not stale from closure)
      const currentLineInfo = textRef.current
        ? ((textRef.current as any).textBufferView as TextBufferView)?.lineInfo
        : null

      // Calculate visual line boundaries from lineInfo (accounts for word wrap)
      // Fall back to logical line boundaries if visual info is unavailable
      const lineStarts = currentLineInfo?.lineStarts ?? []
      const visualLineIndex = lineStarts.findLastIndex(
        (start) => start <= cursorPosition,
      )
      const visualLineStart = visualLineIndex >= 0
        ? lineStarts[visualLineIndex]
        : logicalLineStart
      const visualLineEnd = lineStarts[visualLineIndex + 1] !== undefined
        ? lineStarts[visualLineIndex + 1] - 1
        : logicalLineEnd

      // Alt+Left/B: Word left
      if (
        hasAltLikeModifier &&
        (key.name === 'left' || lowerKeyName === 'b')
      ) {
        suppressKeyDefault(key)
        onChange({
          text: value,
          cursorPosition: wordStart,
          lastEditDueToNav: false,
        })
        return true
      }

      // Alt+Right/F: Word right
      if (
        hasAltLikeModifier &&
        (key.name === 'right' || lowerKeyName === 'f')
      ) {
        suppressKeyDefault(key)
        onChange({
          text: value,
          cursorPosition: wordEnd,
          lastEditDueToNav: false,
        })
        return true
      }

      // Cmd+Left or Home: Line start
      if (
        (key.meta && key.name === 'left' && !hasAltLikeModifier) ||
        (key.name === 'home' && !key.ctrl && !key.meta)
      ) {
        suppressKeyDefault(key)
        onChange({
          text: value,
          cursorPosition: visualLineStart,
          lastEditDueToNav: false,
        })
        return true
      }

      // Cmd+Right or End: Line end
      if (
        (key.meta && key.name === 'right' && !hasAltLikeModifier) ||
        (key.name === 'end' && !key.ctrl && !key.meta)
      ) {
        suppressKeyDefault(key)
        onChange({
          text: value,
          cursorPosition: visualLineEnd,
          lastEditDueToNav: false,
        })
        return true
      }

      // Cmd+Up or Ctrl+Home: Document start
      if (
        (key.meta && key.name === 'up') ||
        (key.ctrl && key.name === 'home')
      ) {
        suppressKeyDefault(key)
        onChange({ text: value, cursorPosition: 0, lastEditDueToNav: false })
        return true
      }

      // Cmd+Down or Ctrl+End: Document end
      if (
        (key.meta && key.name === 'down') ||
        (key.ctrl && key.name === 'end')
      ) {
        suppressKeyDefault(key)
        onChange({
          text: value,
          cursorPosition: value.length,
          lastEditDueToNav: false,
        })
        return true
      }

      // Left arrow (no modifiers)
      if (key.name === 'left' && !key.ctrl && !key.meta && !key.option) {
        suppressKeyDefault(key)
        moveCursorTo(cursorPosition - 1)
        return true
      }

      // Right arrow (no modifiers)
      if (key.name === 'right' && !key.ctrl && !key.meta && !key.option) {
        suppressKeyDefault(key)
        moveCursorTo(cursorPosition + 1)
        return true
      }

      // Up arrow (no modifiers)
      if (key.name === 'up' && !key.ctrl && !key.meta && !key.option) {
        suppressKeyDefault(key)
        const targetColumn = resolveStickyColumn(lineStarts, !shouldHighlight)
        onChange({
          text: value,
          cursorPosition: stepCursorVertical({
            cursorPosition,
            lineStarts,
            cursorIsChar: !shouldHighlight,
            direction: 'up',
            targetColumn,
          }),
          lastEditDueToNav: false,
        })
        return true
      }

      // Down arrow (no modifiers)
      if (key.name === 'down' && !key.ctrl && !key.meta && !key.option) {
        suppressKeyDefault(key)
        const targetColumn = resolveStickyColumn(lineStarts, !shouldHighlight)
        onChange({
          text: value,
          cursorPosition: stepCursorVertical({
            cursorPosition,
            lineStarts,
            cursorIsChar: !shouldHighlight,
            direction: 'down',
            targetColumn,
          }),
          lastEditDueToNav: false,
        })
        return true
      }

      return false
    },
    [value, cursorPosition, onChange, moveCursorTo, shouldHighlight, resolveStickyColumn],
  )

  // Handle character input (regular chars, tab, and IME/multi-byte input)
  const processCharacterKey = useCallback(
    (key: KeyEvent): boolean => {
      // Tab: let higher-level keyboard handlers (like chat keyboard shortcuts) handle it
      if (
        key.name === 'tab' &&
        key.sequence &&
        !key.shift &&
        !key.ctrl &&
        !key.meta &&
        !key.option
      ) {
        // Don't insert a literal tab character here; allow global keyboard handlers to process it
        return false
      }

      // Character input (including multi-byte characters from IME like Chinese, Japanese, Korean)
      // Check for printable input: has a sequence, no modifier keys, and not a control character
      if (
        key.sequence &&
        key.sequence.length >= 1 &&
        !key.ctrl &&
        !key.meta &&
        !key.option &&
        !CONTROL_CHAR_REGEX.test(key.sequence) &&
        isLikelyPrintableKey(key)
      ) {
        suppressKeyDefault(key)
        insertAtCaret(key.sequence)
        return true
      }

      return false
    },
    [insertAtCaret],
  )

  // Main keyboard handler - delegates to specialized handlers
  useKeyboard(
    useCallback(
      (key: KeyEvent) => {
        if (!focused) return

        if (onKeyIntercept) {
          const handled = onKeyIntercept(key)
          if (handled) return
        }

        // Clear sticky column for non-vertical navigation
        const isVerticalNavKey = key.name === 'up' || key.name === 'down'
        if (!isVerticalNavKey) {
          stickyColumnRef.current = null
        }

        // Delegate to specialized handlers
        if (processEnterKey(key)) return
        if (processDeletionKey(key)) return
        if (processNavigationKey(key)) return
        if (processCharacterKey(key)) return
      },
      [
        focused,
        onKeyIntercept,
        processEnterKey,
        processDeletionKey,
        processNavigationKey,
        processCharacterKey,
      ],
    ),
  )

  const layoutMetrics = (() => {
    const safeMaxHeight = Math.max(1, maxHeight)
    const effectiveMinHeight = Math.max(1, Math.min(minHeight, safeMaxHeight))

    const totalLines =
      lineInfo === null ? 0 : lineInfo.lineStarts.length

    // Add bottom gutter when cursor is on line 2 of exactly 2 lines
    const gutterEnabled =
      totalLines === 2 && cursorRow === 1 && totalLines + 1 <= safeMaxHeight

    const rawHeight = Math.min(
      totalLines + (gutterEnabled ? 1 : 0),
      safeMaxHeight,
    )

    const heightLines = Math.max(effectiveMinHeight, rawHeight)

    // Content is scrollable when total lines exceed max height
    const isScrollable = totalLines > safeMaxHeight

    return {
      heightLines,
      gutterEnabled,
      isScrollable,
    }
  })()

  const inputColor = isPlaceholder
    ? theme.muted
    : focused
      ? theme.inputFocusedFg
      : theme.inputFg

  // Use theme's info color for selection highlight background (or custom override)
  const highlightBg = highlightColor ?? theme.info

  return (
    <scrollbox
      ref={scrollBoxRef}
      scrollX={false}
      stickyScroll={true}
      stickyStart="bottom"
      scrollbarOptions={{ visible: false }}
      verticalScrollbarOptions={{
        visible: showScrollbar && layoutMetrics.isScrollable,
        trackOptions: { width: 1 },
      }}
      onPaste={(event) => {
        logger.debug({ text: event.text?.substring(0, 50) }, 'SCROLLBOX PASTE EVENT')
        onPaste(event.text)
      }}
      onMouseDown={handleMouseDown}
      style={{
        flexGrow: 0,
        flexShrink: 0,
        rootOptions: {
          width: '100%',
          height: layoutMetrics.heightLines,
          backgroundColor: 'transparent',
          flexGrow: 0,
          flexShrink: 0,
        },
        wrapperOptions: {
          paddingLeft: 1,
          paddingRight: 1,
          border: false,
        },
        contentOptions: {
          justifyContent: 'flex-start',
        },
      }}
    >
      <text
        ref={textRef}
        style={{ bg: 'transparent', fg: inputColor, wrapMode: 'word' }}
      >
        {showCursor ? (
          <>
            {beforeCursor}
            {shouldHighlight ? (
              <span
                bg={highlightBg}
                fg={theme.background}
                attributes={TextAttributes.BOLD}
              >
                {activeChar === ' ' ? '\u00a0' : activeChar}
              </span>
            ) : (
              <InputCursor
                visible={true}
                focused={focused}
                shouldBlink={effectiveShouldBlinkCursor}
                color={terminalSupportsRgb24() ? (highlightColor ?? theme.info) : 'cyan'}
                key={lastActivity}
              />
            )}
            {shouldHighlight
              ? afterCursor.length > 0
                ? afterCursor.slice(1)
                : ''
              : afterCursor}
            {layoutMetrics.gutterEnabled ? '\n' : ''}
          </>
        ) : (
          <>
            {displayValueForRendering}
            {layoutMetrics.gutterEnabled ? '\n' : ''}
          </>
        )}
      </text>
    </scrollbox>
  )
})
