import { useEffect, useRef, useState } from 'react'

export function useStreamingReveal(
  content: string,
  isStreaming: boolean,
  isInterrupted?: boolean,
  initialDisplayedLength?: number,
): { displayedContent: string; isCatchingUp: boolean; showCursor: boolean } {
  const [displayedLength, setDisplayedLength] = useState(() => {
    if (!isStreaming) return content.length
    return Math.max(0, Math.min(initialDisplayedLength ?? 0, content.length))
  })
  const isLinearDrainRef = useRef(!isStreaming)
  const previousContentRef = useRef(content)
  const previousIsStreamingRef = useRef(isStreaming)

  useEffect(() => {
    const previousContent = previousContentRef.current
    const previousIsStreaming = previousIsStreamingRef.current

    if (!previousIsStreaming && isStreaming) {
      // Fresh stream start — initialize to requested prefix for reveal effect
      setDisplayedLength(Math.max(0, Math.min(initialDisplayedLength ?? 0, content.length)))
    } else if (content.length < previousContent.length) {
      // Content shrunk (e.g. edit replacement is shorter than original)
      // Don't reset to 0 — just clamp to new length
      setDisplayedLength(prev => Math.min(prev, content.length))
    } else if (
      !isStreaming &&
      previousContent &&
      !content.startsWith(previousContent.slice(0, Math.min(previousContent.length, content.length)))
    ) {
      // Content changed non-monotonically outside streaming (e.g. a new edit session on the same artifact)
      // Snap to the full content immediately instead of revealing from stale state.
      setDisplayedLength(content.length)
    } else if (!isStreaming && content.length < displayedLength) {
      setDisplayedLength(content.length)
    }

    previousContentRef.current = content
    previousIsStreamingRef.current = isStreaming
  }, [content, isStreaming, displayedLength, initialDisplayedLength])

  useEffect(() => {
    if (isStreaming) {
      isLinearDrainRef.current = false
    } else {
      isLinearDrainRef.current = true
    }
  }, [isStreaming])

  useEffect(() => {
    if (isInterrupted) setDisplayedLength(content.length)
  }, [isInterrupted, content.length])

  useEffect(() => {
    if (!isStreaming && displayedLength >= content.length) return

    const interval = setInterval(() => {
      setDisplayedLength((prev) => {
        const target = content.length
        if (prev >= target) return prev

        if (isLinearDrainRef.current) {
          return Math.min(target, prev + 8)
        }

        const remaining = target - prev
        const speed = Math.max(1, Math.floor(remaining * 0.15))
        return Math.min(target, prev + speed)
      })
    }, 33)

    return () => clearInterval(interval)
  }, [content.length, displayedLength, isStreaming])

  const safeDisplayedLength = Math.min(displayedLength, content.length)
  const displayedContent = content.slice(0, safeDisplayedLength)
  const isCatchingUp = safeDisplayedLength < content.length
  const showCursor = isStreaming || isCatchingUp

  return { displayedContent, isCatchingUp, showCursor }
}