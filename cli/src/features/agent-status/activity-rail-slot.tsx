import type { ReactNode } from 'react'

/** Aligns the live rail with assistant output; composer chrome supplies its lower separation. */
export function ActivityRailSlot({
  width,
  children,
}: {
  readonly width: number
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
      <box style={{ width, minWidth: 0, height: 1, flexShrink: 0, overflow: 'hidden' }}>
        {children}
      </box>
    </box>
  )
}
