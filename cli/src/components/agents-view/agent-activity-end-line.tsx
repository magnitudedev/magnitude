import { memo, useState } from 'react'
import type { AgentsViewActivityEndItem } from '@magnitudedev/agent'
import type { ForkActivityToolCounts } from '@magnitudedev/agent'
import { Button } from '../button'
import { useTheme } from '../../hooks/use-theme'
import { getAgentColorByRole } from '../../utils/agent-colors'
import { green } from '../../utils/palette'
import { TextAttributes } from '@opentui/core'
import { LaneGutter, type LaneEntry } from './lane-gutter'

interface AgentActivityEndLineProps {
  item: AgentsViewActivityEndItem
  onForkExpand: (forkId: string) => void
  lanes?: LaneEntry[]
}

function formatDuration(seconds: number): string {
  if (seconds < 60) return `${seconds}s`
  const m = Math.floor(seconds / 60)
  const s = seconds % 60
  return s > 0 ? `${m}m ${s}s` : `${m}m`
}

function getTotalToolCount(counts: ForkActivityToolCounts): number {
  return counts.reads + counts.writes + counts.edits + counts.commands
    + counts.webSearches + counts.webFetches + counts.artifactWrites + counts.artifactUpdates
    + counts.searches + counts.clicks + counts.navigations + counts.inputs
    + counts.evaluations + counts.other
}


export const AgentActivityEndLine = memo(function AgentActivityEndLine({
  item,
  onForkExpand,
  lanes = [],
}: AgentActivityEndLineProps) {
  const theme = useTheme()
  const [nameHovered, setNameHovered] = useState(false)

  const palette = getAgentColorByRole(item.agentRole)
  const elapsed = Math.floor((item.completedAt - item.startedAt) / 1000)
  const totalTools = getTotalToolCount(item.toolCounts)
  const timeStr = formatDuration(elapsed)

  return (
    <box
      style={{
        flexDirection: 'row',
        alignItems: 'stretch',
        minHeight: 1,
      }}
    >
      <LaneGutter lanes={lanes} />
      <text style={{ wrapMode: 'none' }}>
        <span fg={green[400]}>{'✔ '}</span>
      </text>
      <Button
        onClick={() => onForkExpand(item.forkId)}
        onMouseOver={() => setNameHovered(true)}
        onMouseOut={() => setNameHovered(false)}
      >
        <text style={{ wrapMode: 'none' }}>
          <span fg={nameHovered ? palette.pulse[0] : palette.border}>{item.agentName}</span>
        </text>
      </Button>
      <text style={{ wrapMode: 'none' }}>
        <span fg={theme.muted}>{' finished ('}{timeStr}</span>
        {totalTools > 0 ? (
          <span fg={theme.muted}>{' · '}{String(totalTools)}{totalTools === 1 ? ' tool' : ' tools'}</span>
        ) : null}
        <span fg={theme.muted}>{')'}</span>
      </text>
    </box>
  )
})