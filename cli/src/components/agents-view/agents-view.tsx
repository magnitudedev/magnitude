import { memo } from 'react'
import type { AgentsViewItem, AgentsViewMessageItem, AgentsViewActivityItem, AgentsViewArtifactItem } from '@magnitudedev/agent'
import { AgentMessageBubble } from './agent-message-bubble'
import { AgentActivityLine } from './agent-activity-line'
import { AgentArtifactEvent } from './agent-artifact-event'

interface AgentsViewProps {
  items: readonly AgentsViewItem[]
  onForkExpand: (forkId: string) => void
  onArtifactClick: (name: string, section?: string) => void
  onViewAgents?: () => void
}

export const AgentsView = memo(function AgentsView({
  items,
  onForkExpand,
  onArtifactClick,
}: AgentsViewProps) {
  return (
    <scrollbox
      stickyScroll
      stickyStart="bottom"
      scrollX={false}
      focusable={false}
      scrollbarOptions={{ visible: false }}
      verticalScrollbarOptions={{
        visible: true,
        trackOptions: { width: 1 },
      }}
      style={{
        flexGrow: 1,
        rootOptions: {
          flexGrow: 1,
          backgroundColor: 'transparent',
        },
        wrapperOptions: {
          border: false,
          backgroundColor: 'transparent',
        },
        contentOptions: {
          paddingLeft: 2,
          paddingRight: 1,
          paddingTop: 1,
          paddingBottom: 1,
          justifyContent: 'flex-end',
        },
      }}
    >
      {items.map((item) => {
        if (item.type === 'agents_view_message') {
          return (
            <AgentMessageBubble
              key={item.id}
              item={item as AgentsViewMessageItem}
              onArtifactClick={onArtifactClick}
            />
          )
        }
        if (item.type === 'agents_view_activity') {
          return (
            <AgentActivityLine
              key={item.id}
              item={item as AgentsViewActivityItem}
              onForkExpand={onForkExpand}
            />
          )
        }
        if (item.type === 'agents_view_artifact') {
          return (
            <AgentArtifactEvent
              key={item.id}
              item={item as AgentsViewArtifactItem}
              onArtifactClick={onArtifactClick}
            />
          )
        }
        return null
      })}
    </scrollbox>
  )
})