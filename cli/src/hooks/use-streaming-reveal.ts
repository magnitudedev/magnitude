import { useRef } from 'react'
import { useAnimationStep } from './use-animation-time'

export interface StreamingRevealResult {
  displayedContent: string
  isCatchingUp: boolean
  showCursor: boolean
}

/**
 * Streaming reveal animation — clock-driven, no useEffect.
 *
 * Uses the shared animation clock for the reveal animation.
 * Refs hold mutable state (previous content/streaming, animation step, displayed length).
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
    lastRevealStep: null as number | null,
    prevContent: '',
    prevIsStreaming: false,
    // Completed content mounts fully revealed; a mid-stream mount starts from
    // the requested prefix (the fresh-stream-start transition below re-applies
    // this and records the initial reveal step).
    displayedLength: isStreaming
      ? Math.max(0, Math.min(initialDisplayedLength ?? 0, content.length))
      : content.length,
  })

  const s = stateRef.current

  // Subscribe to the shared clock only while animating. Completed messages pay
  // nothing per frame, and the interval stops when no component is animating.
  // Entering animation is prop-driven (isStreaming flips / content grows), so
  // no clock frame is needed to start; the render where the reveal catches up
  // computes needsTicks false and unsubscribes on commit.
  const needsClock = !isInterrupted && (isStreaming || s.displayedLength < content.length)

  const revealStep = useAnimationStep(needsClock, 80)

  // Handle state transitions — update refs during render (safe, no re-render trigger)
  if (s.prevContent !== content || s.prevIsStreaming !== isStreaming) {
    if (!s.prevIsStreaming && isStreaming) {
      // Fresh stream start
      s.displayedLength = Math.max(0, Math.min(initialDisplayedLength ?? 0, content.length))
      s.lastRevealStep = revealStep
    } else if (content.length < s.prevContent.length) {
      // Content shrunk
      s.displayedLength = Math.min(s.displayedLength, content.length)
    } else if (
      !isStreaming &&
      s.prevContent &&
      !content.startsWith(s.prevContent.slice(0, Math.min(s.prevContent.length, content.length)))
    ) {
      // Non-monotonic content change outside streaming
      s.displayedLength = content.length
    } else if (!isStreaming && content.length < s.displayedLength) {
      s.displayedLength = content.length
    }

    s.prevContent = content
    s.prevIsStreaming = isStreaming
  }

  // Interrupt snap
  if (isInterrupted && s.displayedLength < content.length) {
    s.displayedLength = content.length
  }

  // Preserve the reveal's 80ms cadence while the shared clock renders at 30 FPS.
  if ((isStreaming || s.displayedLength < content.length) && !isInterrupted) {
    const target = content.length
    if (s.displayedLength < target && s.lastRevealStep !== revealStep) {
      s.lastRevealStep = revealStep
      if (isStreaming) {
        const remaining = target - s.displayedLength
        const speed = Math.max(1, Math.floor(remaining * 0.15))
        s.displayedLength = Math.min(target, s.displayedLength + speed)
      } else {
        // Linear drain when not streaming — must not require a prior stream
        // start, or a never-streamed component whose content grew would keep
        // needsTicks true forever without ever revealing.
        s.displayedLength = Math.min(target, s.displayedLength + 8)
      }
    }
  }

  const safeDisplayedLength = Math.min(s.displayedLength, content.length)
  const displayedContent = content.slice(0, safeDisplayedLength)
  const isCatchingUp = safeDisplayedLength < content.length
  const showCursor = isStreaming || isCatchingUp

  return { displayedContent, isCatchingUp, showCursor }
}
