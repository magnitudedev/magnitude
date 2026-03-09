import { useState, useMemo, useCallback } from 'react'
import type { DisplayMessage } from '@magnitudedev/agent'
import type { BashResult } from '../utils/bash-executor'
import type { TimelineItem } from '../types/timeline'

const TIMELINE_BATCH_SIZE = 20

interface UsePaginatedTimelineReturn {
  visibleItems: readonly TimelineItem[]
  hiddenCount: number
  loadMore: () => void
  hasMore: boolean
}

export function usePaginatedTimeline(
  messages: readonly DisplayMessage[],
  bashOutputs: readonly BashResult[],
  systemMessages: readonly { id: string; text: string; timestamp: number }[]
): UsePaginatedTimelineReturn {
  const [limit, setLimit] = useState(TIMELINE_BATCH_SIZE)

  const allItems: TimelineItem[] = useMemo(() => {
    const chatItems: TimelineItem[] = messages.map(m => ({
      kind: 'chat' as const,
      id: m.id,
      timestamp: m.timestamp,
      message: m,
    }))

    const bashItems: TimelineItem[] = bashOutputs.map(r => ({
      kind: 'bash' as const,
      id: r.id,
      timestamp: r.timestamp,
      result: r,
    }))

    const systemItems: TimelineItem[] = systemMessages.map(s => ({
      kind: 'system' as const,
      id: s.id,
      text: s.text,
      timestamp: s.timestamp,
    }))

    return [...chatItems, ...bashItems, ...systemItems]
      .sort((a, b) => a.timestamp - b.timestamp)
  }, [messages, bashOutputs, systemMessages])

  const visibleItems = useMemo(
    () => allItems.slice(Math.max(0, allItems.length - limit)),
    [allItems, limit]
  )

  const hiddenItems = allItems.slice(0, Math.max(0, allItems.length - limit))
  const hiddenCount = hiddenItems.filter(
    item => item.kind === 'chat' && item.message.type === 'user_message'
  ).length
  const hasMore = hiddenCount > 0

  const loadMore = useCallback(() => {
    setLimit(prev => prev + TIMELINE_BATCH_SIZE)
  }, [])

  return { visibleItems, hiddenCount, loadMore, hasMore }
}
