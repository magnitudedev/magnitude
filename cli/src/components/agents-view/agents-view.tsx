import React, { memo, Fragment } from 'react'
import type { AgentsViewItem, AgentsViewMessageItem, AgentsViewActivityStartItem, AgentsViewActivityEndItem, AgentsViewArtifactItem } from '@magnitudedev/agent'
import { AgentMessageBubble } from './agent-message-bubble'
import { AgentActivityLine } from './agent-activity-line'
import { AgentActivityEndLine } from './agent-activity-end-line'
import { AgentArtifactEvent } from './agent-artifact-event'
import { LaneGutter, type LaneEntry } from './lane-gutter'

interface AgentsViewProps {
  items: readonly AgentsViewItem[]
  onForkExpand: (forkId: string) => void
  onArtifactClick: (name: string, section?: string) => void
  onViewAgents?: () => void
}

function computeLanes(items: readonly AgentsViewItem[]): LaneEntry[][] {
  const activeForks = new Map<string, { role: string }>()
  const result: LaneEntry[][] = []

  const orchestratorLane: LaneEntry = { role: 'orchestrator' }

  for (const item of items) {
    if (item.type === 'agents_view_activity_start') {
      activeForks.set(item.forkId, { role: item.agentRole })
    }

    result.push([orchestratorLane, ...Array.from(activeForks.values()).map(f => ({ role: f.role }))])

    if (item.type === 'agents_view_activity_end') {
      activeForks.delete(item.forkId)
    }
  }

  return result
}

export const AgentsView = memo(function AgentsView({
  items,
  onForkExpand,
  onArtifactClick,
}: AgentsViewProps) {
  const lanes = computeLanes(items)
  const finishedForkIds = new Set(
    items.filter(i => i.type === 'agents_view_activity_end').map(i => (i as AgentsViewActivityEndItem).forkId)
  )

  return (
    <scrollbox
      stickyScroll
      stickyStart="top"
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

        },
      }}
    >
      {items.map((item, index) => {
        const itemLanes = lanes[index] ?? []
        const spacerLanes = lanes[index + 1] ?? []
        let rendered: React.ReactNode = null
        if (item.type === 'agents_view_message') {
          rendered = (
            <AgentMessageBubble
              key={item.id}
              item={item as AgentsViewMessageItem}
              onArtifactClick={onArtifactClick}
              lanes={itemLanes}
            />
          )
        } else if (item.type === 'agents_view_activity_start') {
          rendered = (
            <AgentActivityLine
              key={item.id}
              item={item as AgentsViewActivityStartItem}
              onForkExpand={onForkExpand}
              lanes={itemLanes}
              isFinished={finishedForkIds.has((item as AgentsViewActivityStartItem).forkId)}
            />
          )
        } else if (item.type === 'agents_view_activity_end') {
          rendered = (
            <AgentActivityEndLine
              key={item.id}
              item={item as AgentsViewActivityEndItem}
              onForkExpand={onForkExpand}
              lanes={itemLanes}
            />
          )
        } else if (item.type === 'agents_view_artifact') {
          rendered = (
            <AgentArtifactEvent
              key={item.id}
              item={item as AgentsViewArtifactItem}
              onArtifactClick={onArtifactClick}
              lanes={itemLanes}
            />
          )
        }
        if (rendered === null) return null
        return (
          <Fragment key={item.id}>
            {rendered}
            {index < items.length - 1 && (
              <box style={{ flexDirection: 'row', alignItems: 'stretch', height: 1 }}>
                <LaneGutter lanes={spacerLanes} />
              </box>
            )}
          </Fragment>
        )
      })}
    </scrollbox>
  )
})