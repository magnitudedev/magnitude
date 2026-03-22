import { TextAttributes } from '@opentui/core'

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
import { useMountedRef } from '../hooks/use-mounted-ref'
import { usePasteHandler } from '../hooks/use-paste-handler'
import { useSafeEvent } from '../hooks/use-safe-event'
import { useSafeInterval } from '../hooks/use-safe-interval'
import { useSafeTimeout } from '../hooks/use-safe-timeout'
import { useTheme } from '../hooks/use-theme'

import { safeRenderableAccess, safeRenderableCall } from '../utils/safe-renderable-access'
import { terminalSupportsRgb24 } from '../utils/theme'
import { stepCursorVertical } from './multiline-input.helpers'

import type {
  InputMentionSegment,
  InputPasteSegment,
  InputValue,
} from '../types/store'
import { applyTextEditWithPastesAndMentions } from '../utils/strings'
import { decodeNativePasteText } from './paste-events'
import type {
  KeyEvent,
  LineInfo,
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

function computeLogicalLineStarts(text: string): number[] {
  const starts = [0]
  for (let i = 0; i < text.length; i++) {
    if (text[i] === '\n') {
      starts.push(i + 1)
    }
  }
  return starts
}

function columnToCharOffsetInLine(
  text: string,
  lineStartOffset: number,
  lineEndOffsetExclusive: number,
  targetCol: number,
): number {
  let charOffset = lineStartOffset
  let currentCol = 0
  const clampedTargetCol = Math.max(0, targetCol)

  while (charOffset < lineEndOffsetExclusive && currentCol < clampedTargetCol) {
    currentCol += text[charOffset] === '\t' ? TAB_WIDTH : 1
    charOffset++
  }

  return Math.min(charOffset, lineEndOffsetExclusive)
}

export function deriveVisualLineStarts(text: string, lineInfo: LineInfo | null): number[] {
  if (
    !lineInfo ||
    !Array.isArray(lineInfo.lineSources) ||
    !Array.isArray(lineInfo.lineStartCols) ||
    lineInfo.lineSources.length === 0 ||
    lineInfo.lineStartCols.length === 0
  ) {
    return computeLogicalLineStarts(text)
  }

  const logicalLineStarts = computeLogicalLineStarts(text)
  const visualLineCount = Math.min(lineInfo.lineSources.length, lineInfo.lineStartCols.length)
  const visualLineStarts: number[] = []

  for (let i = 0; i < visualLineCount; i++) {
    const rawSourceLineIndex = lineInfo.lineSources[i]
    const sourceLineIndex = Number.isFinite(rawSourceLineIndex)
      ? Math.max(0, Math.min(logicalLineStarts.length - 1, rawSourceLineIndex))
      : 0
    const rawStartCol = lineInfo.lineStartCols[i]
    const startCol = Number.isFinite(rawStartCol) ? Math.max(0, rawStartCol) : 0

    const lineStart = logicalLineStarts[sourceLineIndex] ?? 0
    const nextLogicalLineStart = logicalLineStarts[sourceLineIndex + 1] ?? text.length + 1
    const lineEndExclusive = text[nextLogicalLineStart - 1] === '\n'
      ? nextLogicalLineStart - 1
      : Math.min(nextLogicalLineStart, text.length)

    const derivedStart = columnToCharOffsetInLine(
      text,
      lineStart,
      lineEndExclusive,
      startCol,
    )

    const prev = visualLineStarts[i - 1]
    const nonDecreasingStart = prev === undefined ? derivedStart : Math.max(prev, derivedStart)
    visualLineStarts.push(Math.max(0, Math.min(text.length, nonDecreasingStart)))
  }

  return visualLineStarts.length > 0 ? visualLineStarts : [0]
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

function sortMentionSegments(segments: InputMentionSegment[]): InputMentionSegment[] {
  return [...segments].sort((a, b) => a.start - b.start)
}

function findMentionById(
  segments: InputMentionSegment[],
  id: string | null | undefined,
): InputMentionSegment | undefined {
  if (!id) return undefined
  return segments.find((s) => s.id === id)
}

function mentionAtLeftEdge(
  segments: InputMentionSegment[],
  pos: number,
): InputMentionSegment | undefined {
  return segments.find((s) => s.end === pos)
}

function mentionAtRightEdge(
  segments: InputMentionSegment[],
  pos: number,
): InputMentionSegment | undefined {
  return segments.find((s) => s.start === pos)
}

function mentionContainingInterior(
  segments: InputMentionSegment[],
  pos: number,
): InputMentionSegment | undefined {
  return segments.find((s) => pos > s.start && pos < s.end)
}

function normalizeCursorPosition(
  pasteSegments: InputPasteSegment[],
  mentionSegments: InputMentionSegment[],
  raw: number,
  textLength: number,
): number {
  let pos = Math.max(0, Math.min(textLength, raw))

  const pasteInterior = segmentContainingInterior(pasteSegments, pos)
  if (pasteInterior) {
    pos = (pos - pasteInterior.start <= pasteInterior.end - pos)
      ? (pasteInterior.start === 0 ? 0 : pasteInterior.start - 1)
      : pasteInterior.end
  }

  const mentionInterior = mentionContainingInterior(mentionSegments, pos)
  if (mentionInterior) {
    pos = (pos - mentionInterior.start <= mentionInterior.end - pos)
      ? (mentionInterior.start === 0 ? 0 : mentionInterior.start - 1)
      : mentionInterior.end
  }

  const atPasteStart = segmentAtRightEdge(pasteSegments, pos)
  if (atPasteStart && atPasteStart.start > 0) {
    pos = atPasteStart.start - 1
  }

  const atMentionStart = mentionAtRightEdge(mentionSegments, pos)
  if (atMentionStart && atMentionStart.start > 0) {
    pos = atMentionStart.start - 1
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
  const blinkIntervalTimerRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const safeTimeout = useSafeTimeout()
  const safeInterval = useSafeInterval()

  // Handle blinking (toggle visible/invisible) when idle
  useEffect(() => {
    safeInterval.clear(blinkIntervalTimerRef.current)
    blinkIntervalTimerRef.current = null

    // Reset cursor to visible
    setIsInvisible(false)

    // Only blink if shouldBlink is enabled, focused, and visible
    if (!shouldBlink || !focused || !visible) return

    const blinkStartTimer = safeTimeout.set(() => {
      blinkIntervalTimerRef.current = safeInterval.set(() => {
        setIsInvisible((prev) => !prev)
      }, blinkInterval)
    }, blinkDelay)

    return () => {
      safeTimeout.clear(blinkStartTimer)
      safeInterval.clear(blinkIntervalTimerRef.current)
      blinkIntervalTimerRef.current = null
    }
  }, [visible, focused, shouldBlink, blinkDelay, blinkInterval, safeTimeout, safeInterval])

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
  mentionSegments?: InputMentionSegment[]
  selectedPasteSegmentId?: string | null
  selectedMentionSegmentId?: string | null
  bulkInsertEpoch?: number
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
    mentionSegments = [],
    selectedPasteSegmentId = null,
    selectedMentionSegmentId = null,
    bulkInsertEpoch = 0,
  }: MultilineInputProps,
  forwardedRef,
) {
  const theme = useTheme()
  const renderer = useRenderer()
  const effectiveShouldBlinkCursor = shouldBlinkCursor ?? true

  const scrollBoxRef = useRef<ScrollBoxRenderable | null>(null)
  const mountedRef = useMountedRef()
  const [lastActivity, setLastActivity] = useState(Date.now())

  const stickyColumnRef = useRef<number | null>(null)
  const prevBulkInsertEpochRef = useRef(bulkInsertEpoch)
  const [suppressBottomFollowAutoScroll, setSuppressBottomFollowAutoScroll] = useState(false)

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

  useEffect(() => {
    if (bulkInsertEpoch === prevBulkInsertEpochRef.current) return
    prevBulkInsertEpochRef.current = bulkInsertEpoch
    setSuppressBottomFollowAutoScroll(true)
  }, [bulkInsertEpoch])

  const textRef = useRef<TextRenderable | null>(null)

  const lineInfo = safeRenderableAccess(
    textRef.current,
    (el) => ((el satisfies TextRenderable as any).textBufferView as TextBufferView).lineInfo,
    {
      mountedRef,
      fallback: null,
    },
  )

  const visualLineStarts = deriveVisualLineStarts(value, lineInfo)

  // Focus/blur scrollbox when focused prop changes
  const prevFocusedRef = useRef(false)
  useEffect(() => {
    if (focused && !prevFocusedRef.current) {
      safeRenderableCall(
        scrollBoxRef.current as FocusableScrollBox | null,
        (sb) => sb.focus?.(),
        { mountedRef },
      )
    } else if (!focused && prevFocusedRef.current) {
      safeRenderableCall(
        scrollBoxRef.current as FocusableScrollBox | null,
        (sb) => sb.blur?.(),
        { mountedRef },
      )
    }
    prevFocusedRef.current = focused
  }, [focused, mountedRef])

  // Expose focus/blur for imperative use cases
  useImperativeHandle(
    forwardedRef,
    () => ({
      focus: () => {
        safeRenderableCall(
          scrollBoxRef.current as FocusableScrollBox | null,
          (sb) => sb.focus?.(),
          { mountedRef },
        )
      },
      blur: () => {
        safeRenderableCall(
          scrollBoxRef.current as FocusableScrollBox | null,
          (sb) => sb.blur?.(),
          { mountedRef },
        )
      },
    }),
    [mountedRef],
  )

  const cursorRow = Math.max(
    0,
    visualLineStarts.findLastIndex(
      (lineStart) => lineStart <= cursorPosition,
    ),
  )

  // Auto-scroll to cursor when content changes
  useEffect(() => {
    if (suppressBottomFollowAutoScroll) {
      setSuppressBottomFollowAutoScroll(false)
      return
    }

    const scrollMetrics = focused
      ? safeRenderableAccess(
          scrollBoxRef.current,
          (scrollBox) => ({
            current: scrollBox.verticalScrollBar.scrollPosition,
            viewportHeight: scrollBox.viewport.height,
            scrollHeight: scrollBox.scrollHeight,
          }),
          {
            mountedRef,
            fallback: null,
          },
        )
      : null

    if (!scrollMetrics) return

    const scrollPosition = Math.min(
      Math.max(
        scrollMetrics.current,
        Math.max(0, cursorRow - scrollMetrics.viewportHeight + 1),
      ),
      Math.min(scrollMetrics.scrollHeight - scrollMetrics.viewportHeight, cursorRow),
    )

    safeRenderableCall(
      scrollBoxRef.current,
      (scrollBox) => {
        scrollBox.verticalScrollBar.scrollPosition = scrollPosition
      },
      { mountedRef },
    )
  }, [cursorPosition, focused, cursorRow, mountedRef, suppressBottomFollowAutoScroll])

  const sortedPasteSegments = sortSegments(pasteSegments)
  const sortedMentionSegments = sortMentionSegments(mentionSegments)

  const commitInput = useCallback(
    (
      next: Partial<InputValue> & Pick<InputValue, 'text' | 'cursorPosition'>,
    ) => {
      const nextPasteSegments = next.pasteSegments ?? pasteSegments
      const nextMentionSegments = next.mentionSegments ?? mentionSegments
      onChange({
        text: next.text,
        cursorPosition: normalizeCursorPosition(
          nextPasteSegments,
          nextMentionSegments,
          next.cursorPosition,
          next.text.length,
        ),
        lastEditDueToNav: next.lastEditDueToNav ?? false,
        pasteSegments: nextPasteSegments,
        mentionSegments: nextMentionSegments,
        selectedPasteSegmentId:
          next.selectedPasteSegmentId !== undefined
            ? next.selectedPasteSegmentId
            : selectedPasteSegmentId,
        selectedMentionSegmentId:
          next.selectedMentionSegmentId !== undefined
            ? next.selectedMentionSegmentId
            : selectedMentionSegmentId,
      })
    },
    [
      onChange,
      pasteSegments,
      mentionSegments,
      selectedPasteSegmentId,
      selectedMentionSegmentId,
    ],
  )

  // Helper to get current selection in original text coordinates
  const readSelectedRange = useCallback((): { start: number; end: number } | null => {
    const selection = safeRenderableAccess(
      textRef.current,
      (el) => {
        const textBufferView = (el as any)?.textBufferView
        if (!textBufferView?.hasSelection?.() || !textBufferView?.getSelection) {
          return null
        }
        return textBufferView.getSelection()
      },
      {
        mountedRef,
        fallback: null,
      },
    )
    if (!selection) return null

    // Convert from render positions to original text positions
    const start = renderIndexToSourceIndex(value, Math.min(selection.start, selection.end))
    const end = renderIndexToSourceIndex(value, Math.max(selection.start, selection.end))

    if (start === end) return null
    return { start, end }
  }, [value, mountedRef])

  // Helper to clear the current selection
  const dismissSelection = useCallback(() => {
    safeRenderableCall(
      renderer as any,
      (r) => r.clearSelection?.(),
      { mountedRef },
    )
  }, [renderer, mountedRef])

  // Helper to delete selected text and return new value and cursor position
  // Helper to handle selection deletion and call onChange if selection existed
  // Returns true if selection was deleted, false otherwise
  const removeSelectionIfPresent = useCallback((): boolean => {
    const selection = readSelectedRange()
    if (!selection) return false
    dismissSelection()
    const next = applyTextEditWithPastesAndMentions(
      {
        text: valueRef.current,
        cursorPosition: cursorPositionRef.current,
        lastEditDueToNav: false,
        pasteSegments: sortedPasteSegments,
        mentionSegments: sortedMentionSegments,
        selectedPasteSegmentId,
        selectedMentionSegmentId,
      },
      selection.start,
      selection.end,
      '',
    )
    commitInput(next)
    valueRef.current = next.text
    cursorPositionRef.current = next.cursorPosition
    return true
  }, [
    readSelectedRange,
    dismissSelection,
    commitInput,
    sortedPasteSegments,
    sortedMentionSegments,
    selectedPasteSegmentId,
    selectedMentionSegmentId,
  ])

  const insertAtCaret = useCallback(
    (textToInsert: string) => {
      if (!textToInsert) return

      // Check if there's a selection to replace
      const selection = readSelectedRange()
      if (selection) {
        dismissSelection()
        const next = applyTextEditWithPastesAndMentions(
          {
            text: valueRef.current,
            cursorPosition: cursorPositionRef.current,
            lastEditDueToNav: false,
            pasteSegments: sortedPasteSegments,
            mentionSegments: sortedMentionSegments,
            selectedPasteSegmentId,
            selectedMentionSegmentId,
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
        sortedMentionSegments,
        cursorPositionRef.current,
        currentValue.length,
      )
      const next = applyTextEditWithPastesAndMentions(
        {
          text: currentValue,
          cursorPosition: currentCursor,
          lastEditDueToNav: false,
          pasteSegments: sortedPasteSegments,
          mentionSegments: sortedMentionSegments,
          selectedPasteSegmentId,
          selectedMentionSegmentId,
        },
        currentCursor,
        currentCursor,
        textToInsert,
      )

      valueRef.current = next.text
      cursorPositionRef.current = next.cursorPosition

      commitInput(next)
    },
    [
      readSelectedRange,
      dismissSelection,
      sortedPasteSegments,
      sortedMentionSegments,
      selectedPasteSegmentId,
      selectedMentionSegmentId,
      commitInput,
    ],
  )

  const moveCursorTo = useCallback(
    (nextPosition: number) => {
      const snapped = normalizeCursorPosition(
        sortedPasteSegments,
        sortedMentionSegments,
        nextPosition,
        value.length,
      )
      if (
        snapped === cursorPosition &&
        !selectedPasteSegmentId &&
        !selectedMentionSegmentId
      ) return
      commitInput({
        text: value,
        cursorPosition: snapped,
        selectedPasteSegmentId: null,
        selectedMentionSegmentId: null,
        lastEditDueToNav: false,
      })
    },
    [
      value,
      cursorPosition,
      selectedPasteSegmentId,
      selectedMentionSegmentId,
      sortedPasteSegments,
      sortedMentionSegments,
      commitInput,
    ],
  )

  // Handle mouse clicks to position cursor
  const handleMouseDown = useSafeEvent(
    useCallback(
      (event: MouseEvent) => {
        if (!focused) return

        // Clear sticky column since this is not up/down navigation
        stickyColumnRef.current = null

        const renderableData = safeRenderableAccess(
          scrollBoxRef.current,
          (scrollBox) => ({
            viewportTop: Number((scrollBox as any).viewport?.y ?? 0),
            viewportLeft: Number((scrollBox as any).viewport?.x ?? 0),
            scrollPosition: scrollBox.verticalScrollBar?.scrollPosition ?? 0,
          }),
          {
            mountedRef,
            fallback: null,
          },
        )
        if (!renderableData) return

        const lineStarts = visualLineStarts

        // Get click position, accounting for scroll
        const clickRowInViewport = Math.floor(event.y - renderableData.viewportTop)
        const clickRow = clickRowInViewport + renderableData.scrollPosition

        // Find which visual line was clicked
        const lineIndex = Math.min(
          Math.max(0, clickRow),
          lineStarts.length - 1,
        )

        // Get the character range for this line
        const lineStartChar = lineStarts[lineIndex]
        const lineEndChar = lineStarts[lineIndex + 1] ?? value.length

        // Convert click x to character position, accounting for tabs
        const clickCol = Math.max(0, Math.floor(event.x - renderableData.viewportLeft))

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

        const clickedMention =
          mentionContainingInterior(sortedMentionSegments, rawClickPosition) ??
          mentionAtRightEdge(sortedMentionSegments, rawClickPosition) ??
          mentionAtLeftEdge(sortedMentionSegments, rawClickPosition)

        if (clickedSegment) {
          if (
            cursorPosition !== clickedSegment.end ||
            selectedPasteSegmentId !== clickedSegment.id ||
            selectedMentionSegmentId
          ) {
            commitInput({
              text: value,
              cursorPosition: clickedSegment.end,
              selectedPasteSegmentId: clickedSegment.id,
              selectedMentionSegmentId: null,
              lastEditDueToNav: false,
            })
          }
          return
        }

        if (clickedMention) {
          if (
            cursorPosition !== clickedMention.end ||
            selectedMentionSegmentId !== clickedMention.id ||
            selectedPasteSegmentId
          ) {
            commitInput({
              text: value,
              cursorPosition: clickedMention.end,
              selectedMentionSegmentId: clickedMention.id,
              selectedPasteSegmentId: null,
              lastEditDueToNav: false,
            })
          }
          return
        }

        const newCursorPosition = normalizeCursorPosition(
          sortedPasteSegments,
          sortedMentionSegments,
          rawClickPosition,
          value.length,
        )

        if (
          newCursorPosition !== cursorPosition ||
          selectedPasteSegmentId ||
          selectedMentionSegmentId
        ) {
          commitInput({
            text: value,
            cursorPosition: newCursorPosition,
            lastEditDueToNav: false,
            selectedPasteSegmentId: null,
            selectedMentionSegmentId: null,
          })
        }
      },
      [
        focused,
        lineInfo,
        value,
        cursorPosition,
        selectedPasteSegmentId,
        selectedMentionSegmentId,
        commitInput,
        sortedPasteSegments,
        sortedMentionSegments,
        mountedRef,
      ],
    ),
  )

  const isPlaceholder = value.length === 0 && placeholder.length > 0
  const displayValue = isPlaceholder ? placeholder : value
  const showCursor = focused
  const showRenderableCursor =
    showCursor && !selectedPasteSegmentId && !selectedMentionSegmentId

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
          const next = applyTextEditWithPastesAndMentions(
            {
              text: value,
              cursorPosition,
              lastEditDueToNav: false,
              pasteSegments: sortedPasteSegments,
              selectedPasteSegmentId,
              mentionSegments: sortedMentionSegments,
              selectedMentionSegmentId: null,
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
        const next = applyTextEditWithPastesAndMentions(
          {
            text: value,
            cursorPosition,
            lastEditDueToNav: false,
            pasteSegments: sortedPasteSegments,
            selectedPasteSegmentId,
            mentionSegments: sortedMentionSegments,
            selectedMentionSegmentId: null,
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
    [
      value,
      cursorPosition,
      onSubmit,
      commitInput,
      sortedPasteSegments,
      sortedMentionSegments,
      selectedPasteSegmentId,
      selectedMentionSegmentId,
    ],
  )

  const deleteSegmentById = useCallback((segmentId: string): boolean => {
    const segment = sortedPasteSegments.find((s) => s.id === segmentId)
    if (!segment) return false
    const next = applyTextEditWithPastesAndMentions(
      {
        text: value,
        cursorPosition,
        lastEditDueToNav: false,
        pasteSegments: sortedPasteSegments,
        mentionSegments: sortedMentionSegments,
        selectedPasteSegmentId,
        selectedMentionSegmentId,
      },
      segment.start,
      segment.end,
      '',
    )
    valueRef.current = next.text
    cursorPositionRef.current = next.cursorPosition
    commitInput(next)
    return true
  }, [
    sortedPasteSegments,
    sortedMentionSegments,
    value,
    cursorPosition,
    selectedPasteSegmentId,
    selectedMentionSegmentId,
    commitInput,
  ])

  const deleteMentionById = useCallback((mentionId: string): boolean => {
    const mention = sortedMentionSegments.find((s) => s.id === mentionId)
    if (!mention) return false
    const next = applyTextEditWithPastesAndMentions(
      {
        text: value,
        cursorPosition,
        lastEditDueToNav: false,
        pasteSegments: sortedPasteSegments,
        mentionSegments: sortedMentionSegments,
        selectedPasteSegmentId,
        selectedMentionSegmentId,
      },
      mention.start,
      mention.end,
      '',
    )
    valueRef.current = next.text
    cursorPositionRef.current = next.cursorPosition
    commitInput(next)
    return true
  }, [
    sortedPasteSegments,
    sortedMentionSegments,
    value,
    cursorPosition,
    selectedPasteSegmentId,
    selectedMentionSegmentId,
    commitInput,
  ])

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
        const next = applyTextEditWithPastesAndMentions(
          {
            text: value,
            cursorPosition,
            lastEditDueToNav: false,
            pasteSegments: sortedPasteSegments,
            selectedPasteSegmentId,
            mentionSegments: sortedMentionSegments,
            selectedMentionSegmentId: null,
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
          const next = applyTextEditWithPastesAndMentions(
            {
              text: value,
              cursorPosition,
              lastEditDueToNav: false,
              pasteSegments: sortedPasteSegments,
              selectedPasteSegmentId,
              mentionSegments: sortedMentionSegments,
              selectedMentionSegmentId: null,
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
        const next = applyTextEditWithPastesAndMentions(
          {
            text: value,
            cursorPosition,
            lastEditDueToNav: false,
            pasteSegments: sortedPasteSegments,
            selectedPasteSegmentId,
            mentionSegments: sortedMentionSegments,
            selectedMentionSegmentId: null,
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

        const selectedMention = findMentionById(
          sortedMentionSegments,
          selectedMentionSegmentId,
        )
        if (selectedMention) {
          return deleteMentionById(selectedMention.id)
        }

        const leftSeg = segmentAtLeftEdge(sortedPasteSegments, cursorPosition)
        if (leftSeg) {
          commitInput({
            text: value,
            cursorPosition: leftSeg.end,
            selectedPasteSegmentId: leftSeg.id,
            selectedMentionSegmentId: null,
          })
          return true
        }

        const leftMention = mentionAtLeftEdge(sortedMentionSegments, cursorPosition)
        if (leftMention) {
          commitInput({
            text: value,
            cursorPosition: leftMention.end,
            selectedMentionSegmentId: leftMention.id,
            selectedPasteSegmentId: null,
          })
          return true
        }

        if (cursorPosition > 0) {
          const next = applyTextEditWithPastesAndMentions(
            {
              text: value,
              cursorPosition,
              lastEditDueToNav: false,
              pasteSegments: sortedPasteSegments,
              selectedPasteSegmentId,
              mentionSegments: sortedMentionSegments,
              selectedMentionSegmentId: null,
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

        const selectedMention = findMentionById(
          sortedMentionSegments,
          selectedMentionSegmentId,
        )
        if (selectedMention) {
          return deleteMentionById(selectedMention.id)
        }

        const forwardSeg =
          segmentAtRightEdge(sortedPasteSegments, cursorPosition + 1) ??
          segmentAtRightEdge(sortedPasteSegments, cursorPosition)

        if (forwardSeg) {
          commitInput({
            text: value,
            cursorPosition: forwardSeg.end,
            selectedPasteSegmentId: forwardSeg.id,
            selectedMentionSegmentId: null,
          })
          return true
        }

        const forwardMention =
          mentionAtRightEdge(sortedMentionSegments, cursorPosition + 1) ??
          mentionAtRightEdge(sortedMentionSegments, cursorPosition)

        if (forwardMention) {
          commitInput({
            text: value,
            cursorPosition: forwardMention.end,
            selectedMentionSegmentId: forwardMention.id,
            selectedPasteSegmentId: null,
          })
          return true
        }

        if (cursorPosition < value.length) {
          const next = applyTextEditWithPastesAndMentions(
            {
              text: value,
              cursorPosition,
              lastEditDueToNav: false,
              pasteSegments: sortedPasteSegments,
              selectedPasteSegmentId,
              mentionSegments: sortedMentionSegments,
              selectedMentionSegmentId: null,
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
    [
      value,
      cursorPosition,
      commitInput,
      removeSelectionIfPresent,
      sortedPasteSegments,
      sortedMentionSegments,
      selectedPasteSegmentId,
      selectedMentionSegmentId,
      deleteSegmentById,
      deleteMentionById,
    ],
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
      const currentLineInfo = safeRenderableAccess(
        textRef.current,
        (el) => ((el as any).textBufferView as TextBufferView)?.lineInfo,
        {
          mountedRef,
          fallback: null,
        },
      )

      // Calculate visual line boundaries from lineInfo (accounts for word wrap)
      // Fall back to logical line boundaries if visual info is unavailable
      const lineStarts = deriveVisualLineStarts(value, currentLineInfo)
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
      const selectedMention = findMentionById(
        sortedMentionSegments,
        selectedMentionSegmentId,
      )

      if (selectedPasteSegmentId && !selectedSegment) {
        commitInput({
          text: value,
          cursorPosition,
          selectedPasteSegmentId: null,
          selectedMentionSegmentId: null,
        })
        return true
      }

      if (selectedMentionSegmentId && !selectedMention) {
        commitInput({
          text: value,
          cursorPosition,
          selectedPasteSegmentId: null,
          selectedMentionSegmentId: null,
        })
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
              selectedMentionSegmentId: null,
            })
          } else {
            moveCursorTo(selectedSegment.start - 1)
          }
          return true
        }

        if (selectedMention) {
          const adjLeft = mentionAtLeftEdge(sortedMentionSegments, selectedMention.start)
          if (adjLeft) {
            commitInput({
              text: value,
              cursorPosition: adjLeft.end,
              selectedMentionSegmentId: adjLeft.id,
              selectedPasteSegmentId: null,
            })
          } else {
            moveCursorTo(selectedMention.start - 1)
          }
          return true
        }

        const leftSeg = segmentAtLeftEdge(sortedPasteSegments, cursorPosition)
        if (leftSeg) {
          commitInput({
            text: value,
            cursorPosition: leftSeg.end,
            selectedPasteSegmentId: leftSeg.id,
            selectedMentionSegmentId: null,
          })
          return true
        }

        const leftMention = mentionAtLeftEdge(sortedMentionSegments, cursorPosition)
        if (leftMention) {
          commitInput({
            text: value,
            cursorPosition: leftMention.end,
            selectedMentionSegmentId: leftMention.id,
            selectedPasteSegmentId: null,
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
              selectedMentionSegmentId: null,
            })
          } else {
            moveCursorTo(selectedSegment.end + 1)
          }
          return true
        }

        if (selectedMention) {
          const adjRight = mentionAtRightEdge(sortedMentionSegments, selectedMention.end)
          if (adjRight) {
            commitInput({
              text: value,
              cursorPosition: adjRight.end,
              selectedMentionSegmentId: adjRight.id,
              selectedPasteSegmentId: null,
            })
          } else {
            moveCursorTo(selectedMention.end + 1)
          }
          return true
        }

        const rightSeg = segmentAtRightEdge(sortedPasteSegments, cursorPosition + 1)
        if (rightSeg) {
          commitInput({
            text: value,
            cursorPosition: rightSeg.end,
            selectedPasteSegmentId: rightSeg.id,
            selectedMentionSegmentId: null,
          })
          return true
        }

        const rightMention = mentionAtRightEdge(sortedMentionSegments, cursorPosition + 1)
        if (rightMention) {
          commitInput({
            text: value,
            cursorPosition: rightMention.end,
            selectedMentionSegmentId: rightMention.id,
            selectedPasteSegmentId: null,
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
    [
      value,
      cursorPosition,
      commitInput,
      moveCursorTo,
      shouldHighlight,
      resolveStickyColumn,
      sortedPasteSegments,
      sortedMentionSegments,
      selectedPasteSegmentId,
      selectedMentionSegmentId,
      mountedRef,
    ],
  )

  // Handle character input (regular chars, tab, and IME/multi-byte input)
  const { handlePasteKey, handlePasteEvent } = usePasteHandler({
    enabled: focused,
    onPaste,
  })

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
    useSafeEvent(
      useCallback(
        (key: KeyEvent) => {
          if (!focused) return

          const selectedSegment = findSegmentById(
            sortedPasteSegments,
            selectedPasteSegmentId,
          )
          const selectedMention = findMentionById(
            sortedMentionSegments,
            selectedMentionSegmentId,
          )
          if (selectedPasteSegmentId && !selectedSegment) {
            commitInput({
              text: value,
              cursorPosition,
              selectedPasteSegmentId: null,
              selectedMentionSegmentId: null,
            })
            return
          }

          if (selectedMentionSegmentId && !selectedMention) {
            commitInput({
              text: value,
              cursorPosition,
              selectedPasteSegmentId: null,
              selectedMentionSegmentId: null,
            })
            return
          }

          const isPillActionKey =
            key.name === 'left' ||
            key.name === 'right' ||
            key.name === 'backspace' ||
            key.name === 'delete'

          if ((selectedSegment || selectedMention) && !isPillActionKey) {
            commitInput({
              text: value,
              cursorPosition: (selectedSegment ?? selectedMention)!.end,
              selectedPasteSegmentId: null,
              selectedMentionSegmentId: null,
            })
          }

          if (onKeyIntercept) {
            const handled = onKeyIntercept(key)
            if (handled) return
          }

          if (handlePasteKey(key)) return

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
          selectedMentionSegmentId,
          sortedPasteSegments,
          sortedMentionSegments,
          cursorPosition,
          commitInput,
          value,
          handlePasteKey,
        ],
      ),
    ),
  )

  const layoutMetrics = (() => {
    const safeMaxHeight = Math.max(1, maxHeight)
    const effectiveMinHeight = Math.max(1, Math.min(minHeight, safeMaxHeight))

    const totalLines = visualLineStarts.length

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
      stickyScroll={!suppressBottomFollowAutoScroll}
      stickyStart="bottom"
      scrollbarOptions={{ visible: false }}
      verticalScrollbarOptions={{
        visible: showScrollbar && layoutMetrics.isScrollable,
        trackOptions: { width: 1 },
      }}
      onPaste={(event) => {
        handlePasteEvent({ text: decodeNativePasteText(event) })
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

              const orderedSegments = [
                ...sortedPasteSegments.map((segment) => ({
                  kind: 'paste' as const,
                  segment,
                })),
                ...sortedMentionSegments.map((segment) => ({
                  kind: 'mention' as const,
                  segment,
                })),
              ].sort((a, b) => a.segment.start - b.segment.start)

              for (const item of orderedSegments) {
                const preText = value.slice(pos, item.segment.start)

                if (item.segment.start > pos) {
                  pushTextChunk(preText, `t-${pos}`)
                }

                const segmentText = value.slice(item.segment.start, item.segment.end)

                if (item.kind === 'paste') {
                  out.push(
                    <span
                      key={`p-${item.segment.id}`}
                      fg={selectedPasteSegmentId === item.segment.id ? theme.link : theme.primary}
                      attributes={
                        selectedPasteSegmentId === item.segment.id
                          ? TextAttributes.BOLD
                          : undefined
                      }
                    >
                      {segmentText}
                    </span>,
                  )
                } else {
                  out.push(
                    <span
                      key={`m-${item.segment.id}`}
                      fg={selectedMentionSegmentId === item.segment.id ? theme.link : theme.info}
                      attributes={
                        selectedMentionSegmentId === item.segment.id
                          ? TextAttributes.BOLD
                          : undefined
                      }
                    >
                      {segmentText}
                    </span>,
                  )
                }

                pos = item.segment.end
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
