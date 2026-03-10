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

import type { InputPasteSegment, InputValue } from '../types/store'
import { applyTextEditWithSegments } from '../utils/strings'
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

function sortSegments(segments: InputPasteSegment[]): InputPasteSegment[] {
  return [...segments].sort((a, b) => a.start - b.start)
}

function findSegmentById(
  segments: InputPasteSegment[],
  id: string | null | undefined,
): InputPasteSegment | undefined {
  if (!id) return undefined
  return segments.find((s) => s.id === id)
}

function segmentAtLeftEdge(
  segments: InputPasteSegment[],
  pos: number,
): InputPasteSegment | undefined {
  return segments.find((s) => s.end === pos)
}

function segmentAtRightEdge(
  segments: InputPasteSegment[],
  pos: number,
): InputPasteSegment | undefined {
  return segments.find((s) => s.start === pos)
}

function segmentContainingInterior(
  segments: InputPasteSegment[],
  pos: number,
): InputPasteSegment | undefined {
  return segments.find((s) => pos > s.start && pos < s.end)
}

function normalizeCursorPosition(
  segments: InputPasteSegment[],
  raw: number,
  textLength: number,
): number {
  let pos = Math.max(0, Math.min(textLength, raw))

  const interior = segmentContainingInterior(segments, pos)
  if (interior) {
    pos = (pos - interior.start <= interior.end - pos)
      ? (interior.start === 0 ? 0 : interior.start - 1)
      : interior.end
  }

  const atStart = segmentAtRightEdge(segments, pos)
  if (atStart && atStart.start > 0) {
    pos = atStart.start - 1
  }

  return pos
}

export { INPUT_CURSOR_CHAR } from './multiline-input.helpers'

interface CursorIndicatorProps {
  visible: boolean
  focused: boolean
  shouldBlink?: boolean
  char?: string
  color?: string
  backgroundColor?: string
  activeChar?: string
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
  backgroundColor,
  activeChar,
  blinkDelay = 500,
  blinkInterval = 500,
  bold = false,
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

  // When blink-off: show underlying character only for block cursor mode.
  // For thin cursor mode, render nothing (never a space).
  if (isBlinkHidden) {
    if (activeChar !== undefined) {
      return <span>{activeChar}</span>
    }
    return null
  }

  return (
    <span
      {...(color ? { fg: color } : undefined)}
      {...(backgroundColor ? { bg: backgroundColor } : undefined)}
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
  pasteSegments?: InputPasteSegment[]
  selectedPasteSegmentId?: string | null
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
    pasteSegments = [],
    selectedPasteSegmentId = null,
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

  const sortedPasteSegments = sortSegments(pasteSegments)

  const commitInput = useCallback(
    (
      next: Partial<InputValue> & Pick<InputValue, 'text' | 'cursorPosition'>,
    ) => {
      const segments = next.pasteSegments ?? pasteSegments
      onChange({
        text: next.text,
        cursorPosition: normalizeCursorPosition(
          segments,
          next.cursorPosition,
          next.text.length,
        ),
        lastEditDueToNav: next.lastEditDueToNav ?? false,
        pasteSegments: segments,
        selectedPasteSegmentId:
          next.selectedPasteSegmentId !== undefined
            ? next.selectedPasteSegmentId
            : selectedPasteSegmentId,
      })
    },
    [onChange, pasteSegments, selectedPasteSegmentId],
  )

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
  // Helper to handle selection deletion and call onChange if selection existed
  // Returns true if selection was deleted, false otherwise
  const removeSelectionIfPresent = useCallback((): boolean => {
    const selection = readSelectedRange()
    if (!selection) return false
    dismissSelection()
    const next = applyTextEditWithSegments(
      {
        text: valueRef.current,
        cursorPosition: cursorPositionRef.current,
        lastEditDueToNav: false,
        pasteSegments: sortedPasteSegments,
        selectedPasteSegmentId,
      },
      selection.start,
      selection.end,
      '',
    )
    commitInput(next)
    valueRef.current = next.text
    cursorPositionRef.current = next.cursorPosition
    return true
  }, [readSelectedRange, dismissSelection, commitInput, sortedPasteSegments, selectedPasteSegmentId])

  const insertAtCaret = useCallback(
    (textToInsert: string) => {
      if (!textToInsert) return

      // Check if there's a selection to replace
      const selection = readSelectedRange()
      if (selection) {
        dismissSelection()
        const next = applyTextEditWithSegments(
          {
            text: valueRef.current,
            cursorPosition: cursorPositionRef.current,
            lastEditDueToNav: false,
            pasteSegments: sortedPasteSegments,
            selectedPasteSegmentId,
          },
          selection.start,
          selection.end,
          textToInsert,
        )

        valueRef.current = next.text
        cursorPositionRef.current = next.cursorPosition
        commitInput(next)
        return
      }

      // No selection, insert at cursor
      // Read from refs to get latest state (handles rapid IME input)
      const currentValue = valueRef.current
      const currentCursor = normalizeCursorPosition(
        sortedPasteSegments,
        cursorPositionRef.current,
        currentValue.length,
      )
      const next = applyTextEditWithSegments(
        {
          text: currentValue,
          cursorPosition: currentCursor,
          lastEditDueToNav: false,
          pasteSegments: sortedPasteSegments,
          selectedPasteSegmentId,
        },
        currentCursor,
        currentCursor,
        textToInsert,
      )

      valueRef.current = next.text
      cursorPositionRef.current = next.cursorPosition

      commitInput(next)
    },
    [readSelectedRange, dismissSelection, sortedPasteSegments, selectedPasteSegmentId, commitInput],
  )

  const moveCursorTo = useCallback(
    (nextPosition: number) => {
      const snapped = normalizeCursorPosition(
        sortedPasteSegments,
        nextPosition,
        value.length,
      )
      if (snapped === cursorPosition && !selectedPasteSegmentId) return
      commitInput({
        text: value,
        cursorPosition: snapped,
        selectedPasteSegmentId: null,
        lastEditDueToNav: false,
      })
    },
    [value, cursorPosition, selectedPasteSegmentId, sortedPasteSegments, commitInput],
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

      const rawClickPosition = Math.min(charIndex, value.length)

      const clickedSegment =
        segmentContainingInterior(sortedPasteSegments, rawClickPosition) ??
        segmentAtRightEdge(sortedPasteSegments, rawClickPosition) ??
        segmentAtLeftEdge(sortedPasteSegments, rawClickPosition)

      if (clickedSegment) {
        if (
          cursorPosition !== clickedSegment.end ||
          selectedPasteSegmentId !== clickedSegment.id
        ) {
          commitInput({
            text: value,
            cursorPosition: clickedSegment.end,
            selectedPasteSegmentId: clickedSegment.id,
            lastEditDueToNav: false,
          })
        }
        return
      }

      const newCursorPosition = normalizeCursorPosition(
        sortedPasteSegments,
        rawClickPosition,
        value.length,
      )

      if (newCursorPosition !== cursorPosition || selectedPasteSegmentId) {
        commitInput({
          text: value,
          cursorPosition: newCursorPosition,
          lastEditDueToNav: false,
          selectedPasteSegmentId: null,
        })
      }
    },
    [
      focused,
      lineInfo,
      value,
      cursorPosition,
      selectedPasteSegmentId,
      commitInput,
      sortedPasteSegments,
    ],
  )

  const isPlaceholder = value.length === 0 && placeholder.length > 0
  const displayValue = isPlaceholder ? placeholder : value
  const showCursor = focused
  const showRenderableCursor = showCursor && !selectedPasteSegmentId

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
          const next = applyTextEditWithSegments(
            {
              text: value,
              cursorPosition,
              lastEditDueToNav: false,
              pasteSegments: sortedPasteSegments,
              selectedPasteSegmentId,
            },
            cursorPosition - 1,
            cursorPosition,
            '\n',
          )
          valueRef.current = next.text
          cursorPositionRef.current = next.cursorPosition
          commitInput(next)
          return true
        }

        // For other newline shortcuts (Shift+Enter, Option+Enter, Ctrl+J), just insert newline
        const next = applyTextEditWithSegments(
          {
            text: value,
            cursorPosition,
            lastEditDueToNav: false,
            pasteSegments: sortedPasteSegments,
            selectedPasteSegmentId,
          },
          cursorPosition,
          cursorPosition,
          '\n',
        )
        valueRef.current = next.text
        cursorPositionRef.current = next.cursorPosition
        commitInput(next)
        return true
      }

      if (isPlainEnter) {
        suppressKeyDefault(key)
        onSubmit()
        return true
      }

      return false
    },
    [value, cursorPosition, onSubmit, commitInput, sortedPasteSegments, selectedPasteSegmentId],
  )

  const deleteSegmentById = useCallback((segmentId: string): boolean => {
    const segment = sortedPasteSegments.find((s) => s.id === segmentId)
    if (!segment) return false
    const next = applyTextEditWithSegments(
      {
        text: value,
        cursorPosition,
        lastEditDueToNav: false,
        pasteSegments: sortedPasteSegments,
        selectedPasteSegmentId,
      },
      segment.start,
      segment.end,
      '',
    )
    valueRef.current = next.text
    cursorPositionRef.current = next.cursorPosition
    commitInput(next)
    return true
  }, [sortedPasteSegments, value, cursorPosition, selectedPasteSegmentId, commitInput])

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
        const next = applyTextEditWithSegments(
          {
            text: value,
            cursorPosition,
            lastEditDueToNav: false,
            pasteSegments: sortedPasteSegments,
            selectedPasteSegmentId,
          },
          wordStart,
          cursorPosition,
          '',
        )
        valueRef.current = next.text
        cursorPositionRef.current = next.cursorPosition
        commitInput(next)
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
          const deleteStart = Math.min(cursorPosition, nextCursor)
          const deleteEnd = Math.max(cursorPosition, nextCursor)
          const next = applyTextEditWithSegments(
            {
              text: value,
              cursorPosition,
              lastEditDueToNav: false,
              pasteSegments: sortedPasteSegments,
              selectedPasteSegmentId,
            },
            deleteStart,
            deleteEnd,
            '',
          )
          valueRef.current = next.text
          cursorPositionRef.current = next.cursorPosition
          commitInput(next)
        }
        return true
      }

      // Alt+Delete: Delete word forward
      if (key.name === 'delete' && hasAltLikeModifier) {
        suppressKeyDefault(key)
        if (removeSelectionIfPresent()) return true
        const next = applyTextEditWithSegments(
          {
            text: value,
            cursorPosition,
            lastEditDueToNav: false,
            pasteSegments: sortedPasteSegments,
            selectedPasteSegmentId,
          },
          cursorPosition,
          wordEnd,
          '',
        )
        valueRef.current = next.text
        cursorPositionRef.current = next.cursorPosition
        commitInput(next)
        return true
      }

      // Basic Backspace (no modifiers)
      if (key.name === 'backspace' && !key.ctrl && !key.meta && !key.option) {
        suppressKeyDefault(key)
        if (removeSelectionIfPresent()) return true

        const selectedSegment = findSegmentById(
          sortedPasteSegments,
          selectedPasteSegmentId,
        )
        if (selectedSegment) {
          return deleteSegmentById(selectedSegment.id)
        }

        const leftSeg = segmentAtLeftEdge(sortedPasteSegments, cursorPosition)
        if (leftSeg) {
          commitInput({
            text: value,
            cursorPosition: leftSeg.end,
            selectedPasteSegmentId: leftSeg.id,
          })
          return true
        }

        if (cursorPosition > 0) {
          const next = applyTextEditWithSegments(
            {
              text: value,
              cursorPosition,
              lastEditDueToNav: false,
              pasteSegments: sortedPasteSegments,
              selectedPasteSegmentId,
            },
            cursorPosition - 1,
            cursorPosition,
            '',
          )
          valueRef.current = next.text
          cursorPositionRef.current = next.cursorPosition
          commitInput(next)
        }
        return true
      }

      // Basic Delete (no modifiers)
      if (key.name === 'delete' && !key.ctrl && !key.meta && !key.option) {
        suppressKeyDefault(key)
        if (removeSelectionIfPresent()) return true

        const selectedSegment = findSegmentById(
          sortedPasteSegments,
          selectedPasteSegmentId,
        )
        if (selectedSegment) {
          return deleteSegmentById(selectedSegment.id)
        }

        const forwardSeg =
          segmentAtRightEdge(sortedPasteSegments, cursorPosition + 1) ??
          segmentAtRightEdge(sortedPasteSegments, cursorPosition)

        if (forwardSeg) {
          commitInput({
            text: value,
            cursorPosition: forwardSeg.end,
            selectedPasteSegmentId: forwardSeg.id,
          })
          return true
        }

        if (cursorPosition < value.length) {
          const next = applyTextEditWithSegments(
            {
              text: value,
              cursorPosition,
              lastEditDueToNav: false,
              pasteSegments: sortedPasteSegments,
              selectedPasteSegmentId,
            },
            cursorPosition,
            cursorPosition + 1,
            '',
          )
          valueRef.current = next.text
          cursorPositionRef.current = next.cursorPosition
          commitInput(next)
        }
        return true
      }

      return false
    },
    [value, cursorPosition, commitInput, removeSelectionIfPresent, sortedPasteSegments, selectedPasteSegmentId, deleteSegmentById],
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
        commitInput({
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
        commitInput({
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
        commitInput({
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
        commitInput({
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
        commitInput({ text: value, cursorPosition: 0, lastEditDueToNav: false })
        return true
      }

      // Cmd+Down or Ctrl+End: Document end
      if (
        (key.meta && key.name === 'down') ||
        (key.ctrl && key.name === 'end')
      ) {
        suppressKeyDefault(key)
        commitInput({
          text: value,
          cursorPosition: value.length,
          lastEditDueToNav: false,
        })
        return true
      }

      const selectedSegment = findSegmentById(
        sortedPasteSegments,
        selectedPasteSegmentId,
      )

      if (selectedPasteSegmentId && !selectedSegment) {
        commitInput({ text: value, cursorPosition, selectedPasteSegmentId: null })
        return true
      }

      // Left arrow (no modifiers)
      if (key.name === 'left' && !key.ctrl && !key.meta && !key.option) {
        suppressKeyDefault(key)

        if (selectedSegment) {
          const adjLeft = segmentAtLeftEdge(sortedPasteSegments, selectedSegment.start)
          if (adjLeft) {
            commitInput({
              text: value,
              cursorPosition: adjLeft.end,
              selectedPasteSegmentId: adjLeft.id,
            })
          } else {
            moveCursorTo(selectedSegment.start - 1)
          }
          return true
        }

        const leftSeg = segmentAtLeftEdge(sortedPasteSegments, cursorPosition)
        if (leftSeg) {
          commitInput({
            text: value,
            cursorPosition: leftSeg.end,
            selectedPasteSegmentId: leftSeg.id,
          })
          return true
        }

        moveCursorTo(cursorPosition - 1)
        return true
      }

      // Right arrow (no modifiers)
      if (key.name === 'right' && !key.ctrl && !key.meta && !key.option) {
        suppressKeyDefault(key)

        if (selectedSegment) {
          const adjRight = segmentAtRightEdge(sortedPasteSegments, selectedSegment.end)
          if (adjRight) {
            commitInput({
              text: value,
              cursorPosition: adjRight.end,
              selectedPasteSegmentId: adjRight.id,
            })
          } else {
            moveCursorTo(selectedSegment.end + 1)
          }
          return true
        }

        const rightSeg = segmentAtRightEdge(sortedPasteSegments, cursorPosition + 1)
        if (rightSeg) {
          commitInput({
            text: value,
            cursorPosition: rightSeg.end,
            selectedPasteSegmentId: rightSeg.id,
          })
          return true
        }

        moveCursorTo(cursorPosition + 1)
        return true
      }

      // Up arrow (no modifiers)
      if (key.name === 'up' && !key.ctrl && !key.meta && !key.option) {
        suppressKeyDefault(key)
        const targetColumn = resolveStickyColumn(lineStarts, !shouldHighlight)
        commitInput({
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
        commitInput({
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
    [value, cursorPosition, commitInput, moveCursorTo, shouldHighlight, resolveStickyColumn, sortedPasteSegments, selectedPasteSegmentId],
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

        const selectedSegment = findSegmentById(
          sortedPasteSegments,
          selectedPasteSegmentId,
        )
        if (selectedPasteSegmentId && !selectedSegment) {
          commitInput({
            text: value,
            cursorPosition,
            selectedPasteSegmentId: null,
          })
          return
        }

        const isPillActionKey =
          key.name === 'left' ||
          key.name === 'right' ||
          key.name === 'backspace' ||
          key.name === 'delete'

        if (selectedSegment && !isPillActionKey) {
          commitInput({
            text: value,
            cursorPosition: selectedSegment.end,
            selectedPasteSegmentId: null,
          })
        }

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
        selectedPasteSegmentId,
        sortedPasteSegments,
        cursorPosition,
        commitInput,
        value,
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
        {isPlaceholder ? (
          <>
            {showRenderableCursor && (
              <InputCursor
                visible={true}
                focused={focused}
                shouldBlink={effectiveShouldBlinkCursor}
                char={'▍'}
                color={terminalSupportsRgb24() ? (highlightColor ?? theme.info) : 'cyan'}
                activeChar={' '}
                key={`placeholder-cursor-${lastActivity}`}
              />
            )}
            {displayValueForRendering}
            {layoutMetrics.gutterEnabled ? '\n' : ''}
          </>
        ) : (
          <>
            {(() => {
              const out: any[] = []
              let pos = 0
              let cursorRendered = false
              let cursorRenderCount = 0

              const pushCursor = (activeChar?: string) => {
                if (!showRenderableCursor || cursorRendered) return
                const cursorColor = terminalSupportsRgb24() ? (highlightColor ?? theme.info) : 'cyan'
                const isBlockCursor =
        activeChar !== undefined &&
        activeChar !== ' ' &&
        activeChar !== '\t'
                out.push(
                  <InputCursor
                    key={`cursor-${cursorPosition}-${lastActivity}-${cursorRenderCount++}`}
                    visible={true}
                    focused={focused}
                    shouldBlink={effectiveShouldBlinkCursor}
                    char={isBlockCursor ? activeChar : '▍'}
                    color={isBlockCursor ? undefined : cursorColor}
                    backgroundColor={isBlockCursor ? cursorColor : undefined}
                    activeChar={activeChar}
                  />,
                )
                cursorRendered = true
              }

              const pushTextChunk = (text: string, key: string) => {
                if (!text) return
                if (
                  showRenderableCursor &&
                  !cursorRendered &&
                  cursorPosition >= pos &&
                  cursorPosition < pos + text.length
                ) {
                  const rel = cursorPosition - pos
                  const activeChar = text.charAt(rel)
                  out.push(<span key={`${key}-pre`}>{text.slice(0, rel)}</span>)
                  pushCursor(activeChar)
                  out.push(<span key={`${key}-post`}>{text.slice(rel + 1)}</span>)
                } else {
                  out.push(<span key={key}>{text}</span>)
                }
                pos += text.length
              }

              for (const segment of sortedPasteSegments) {
                const preText = value.slice(pos, segment.start)

                if (segment.start > pos) {
                  pushTextChunk(preText, `t-${pos}`)
                }

                const segmentText = value.slice(segment.start, segment.end)
                out.push(
                  <span
                    key={`p-${segment.id}`}
                    fg={selectedPasteSegmentId === segment.id ? theme.link : theme.primary}
                    attributes={
                      selectedPasteSegmentId === segment.id
                        ? TextAttributes.BOLD
                        : undefined
                    }
                  >
                    {segmentText}
                  </span>,
                )

                pos = segment.end

              }

              if (pos < value.length) {
                pushTextChunk(value.slice(pos), 'tail')
              }

              if (showRenderableCursor && !cursorRendered && cursorPosition === value.length) {
                pushCursor()
              }

              return out
            })()}
            {layoutMetrics.gutterEnabled ? '\n' : ''}
          </>
        )}
      </text>
    </scrollbox>
  )
})
