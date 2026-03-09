import React, { useMemo } from 'react'

import { type RecentChat } from '../data/recent-chats'
import { useTheme } from '../hooks/use-theme'

interface SessionLoadingViewProps {
  sessionSelection: string | null | undefined
  recentChats: RecentChat[] | null
}

const formatTimestamp = (timestamp?: number): string | null => {
  if (!timestamp) return null
  const date = new Date(timestamp)
  if (Number.isNaN(date.getTime())) return null

  return date.toLocaleString(undefined, {
    month: 'short',
    day: 'numeric',
    year: 'numeric',
    hour: 'numeric',
    minute: '2-digit',
  })
}

export const SessionLoadingView = ({
  sessionSelection,
  recentChats,
}: SessionLoadingViewProps) => {
  const theme = useTheme()

  const isResuming = sessionSelection !== null
  const loadingLabel = isResuming ? 'Restoring session...' : 'Starting new session...'

  const selectedSession = useMemo(() => {
    if (!sessionSelection || !recentChats) return null
    return recentChats.find((chat) => chat.id === sessionSelection) ?? null
  }, [sessionSelection, recentChats])

  const formattedTimestamp = useMemo(
    () => formatTimestamp(selectedSession?.timestamp),
    [selectedSession?.timestamp],
  )

  return (
    <box
      style={{
        flexDirection: 'column',
        flexGrow: 1,
        justifyContent: 'center',
        alignItems: 'center',
      }}
    >
      <box style={{ flexDirection: 'row', alignItems: 'center' }}>
        <text style={{ fg: theme.muted }}>{loadingLabel}</text>
      </box>

      {selectedSession?.title ? (
        <box style={{ marginTop: 1 }}>
          <text style={{ fg: theme.muted }}>{selectedSession.title}</text>
        </box>
      ) : null}

      {formattedTimestamp ? (
        <box>
          <text style={{ fg: theme.muted }}>{formattedTimestamp}</text>
        </box>
      ) : null}
    </box>
  )
}