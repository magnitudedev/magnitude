import { useRef } from 'react'
import { useAnimationTime } from './use-animation-time'

export interface StreamingRevealResult {
  displayedContent: string
  isCatchingUp: boolean
  showCursor: boolean
}

/**
 * Streaming reveal animation — clock-driven, no useEffect.
 *
 * Uses the shared animation clock for the reveal animation.
 * Refs hold mutable state (previous content/streaming, animation frame, displayed length).
 * The clock subscription via useSyncExternalStore drives re-renders,
 * during which we read refs and compute the new displayed length.
 * No useState, no useEffect — refs are updated during render (safe because
 * they don't trigger re-renders; the animation clock does).
 */
export function useStreamingReveal(
  content: string,
  isStreaming: boolean,
  isInterrupted?: boolean,
  initialDisplayedLength?: number,
): StreamingRevealResult {
  const stateRef = useRef({
    lastRevealFrame: null as number | null,
    prevContent: '',
    prevIsStreaming: false,
    // Completed content mounts fully revealed; a mid-stream mount starts from
    // the requested prefix (the fresh-stream-start transition below re-applies
    // this and records the initial animation frame).
    displayedLength: isStreaming
      ? Math.max(0, Math.min(initialDisplayedLength ?? 0, content.length))
      : content.length,
  })

  const s = stateRef.current

  // Active streams reveal on every shared animation frame. Completion and
  // interruption snap synchronously and therefore never need the clock.
  const animationTime = useAnimationTime(isStreaming && !isInterrupted)

  // Handle state transitions — update refs during render (safe, no re-render trigger)
  if (s.prevContent !== content || s.prevIsStreaming !== isStreaming) {
    if (!s.prevIsStreaming && isStreaming) {
      // Fresh stream start
      s.displayedLength = Math.max(0, Math.min(initialDisplayedLength ?? 0, content.length))
      s.lastRevealFrame = animationTime
    } else if (!isStreaming) {
      // A completed response is authoritative and appears in full immediately.
      s.displayedLength = content.length
    } else if (content.length < s.prevContent.length) {
      // Content shrunk
      s.displayedLength = Math.min(s.displayedLength, content.length)
    }

    s.prevContent = content
    s.prevIsStreaming = isStreaming
  }

  // Interrupt snap
  if (isInterrupted && s.displayedLength < content.length) {
    s.displayedLength = content.length
  }

  // Catch up proportionally once per shared animation frame while streaming.
  if (isStreaming && !isInterrupted) {
    const target = content.length
    if (s.displayedLength < target && s.lastRevealFrame !== animationTime) {
      s.lastRevealFrame = animationTime
      const remaining = target - s.displayedLength
      const speed = Math.max(1, Math.floor(remaining * 0.15))
      s.displayedLength = Math.min(target, s.displayedLength + speed)
    }
  }

  const safeDisplayedLength = Math.min(s.displayedLength, content.length)
  const displayedContent = content.slice(0, safeDisplayedLength)
  const isCatchingUp = safeDisplayedLength < content.length
  const showCursor = isStreaming || isCatchingUp

  return { displayedContent, isCatchingUp, showCursor }
}
