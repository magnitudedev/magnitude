import { getAgentColorByRole } from '../../utils/agent-colors'
import type { BorderCharacters } from '@opentui/core'

export interface LaneEntry {
  role: string
}

const LANE_BORDER_CHARS: BorderCharacters = {
  topLeft: ' ', topRight: ' ',
  bottomLeft: ' ', bottomRight: ' ',
  horizontal: ' ', vertical: '┃',
  leftT: ' ', rightT: ' ',
  topT: ' ', bottomT: ' ',
  cross: ' ',
}

export function LaneGutter({ lanes }: { lanes: LaneEntry[] }) {
  if (lanes.length === 0) {
    return <text style={{ wrapMode: 'none' }}>{'  '}</text>
  }

  return (
    <box style={{ flexDirection: 'row', alignSelf: 'stretch' }}>
      {lanes.map((lane, i) => (
        <box
          key={i}
          style={{
            width: 1,
            marginRight: 1,
            alignSelf: 'stretch',
            borderStyle: 'single',
            border: ['left'],
            borderColor: getAgentColorByRole(lane.role).border,
            customBorderChars: LANE_BORDER_CHARS,
          }}
        />
      ))}
      <text style={{ wrapMode: 'none' }}>{' '}</text>
    </box>
  )
}