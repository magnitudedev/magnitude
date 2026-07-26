import type { ReactNode } from 'react'

const ACTIVITY_HEADER_PREFIX = '┏━ '

/** Renders live activity as a terminal-background header attached directly to the composer rail. */
export function ActivityRailSlot({
  width,
  color,
  children,
}: {
  readonly width: number
  readonly color: string
  readonly children: ReactNode
}): ReactNode {
  return (
    <box id="root-activity-rail" style={{
      height: 1,
      flexShrink: 0,
      flexDirection: 'column',
      paddingLeft: 1,
      paddingRight: 2,
    }}>
      <box style={{ width, minWidth: 0, height: 1, flexShrink: 0, flexDirection: 'row', overflow: 'hidden' }}>
        <text style={{ fg: color }}>{ACTIVITY_HEADER_PREFIX}</text>
        <box style={{ flexGrow: 1, minWidth: 0, height: 1, overflow: 'hidden' }}>
          {children}
        </box>
      </box>
    </box>
  )
}
