import type { ReactNode } from 'react'

/** Aligns the rail with assistant output; the preceding entry supplies the top gap. */
export function ActivityRailSlot({
  width,
  children,
}: {
  readonly width: number
  readonly children: ReactNode
}): ReactNode {
  return (
    <box id="root-activity-rail" style={{
      height: 2,
      flexShrink: 0,
      flexDirection: 'column',
      paddingLeft: 1,
      paddingRight: 2,
    }}>
      <box style={{ width, minWidth: 0, height: 1, flexShrink: 0, overflow: 'hidden' }}>
        {children}
      </box>
      <box id="activity-spacing-below" style={{ height: 1, flexShrink: 0 }} />
    </box>
  )
}
