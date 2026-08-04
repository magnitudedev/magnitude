import type { ReactNode } from 'react'
import { TextAttributes } from '@opentui/core'
import { truncateToDisplayWidth } from '@magnitudedev/client-common'
import { useTheme } from '../../hooks/use-theme'
import {
  CompactMagnitudeLogo,
  COMPACT_MAGNITUDE_LOGO_WIDTH,
} from '../../components/compact-magnitude-logo'

const SIDE_BY_SIDE_MIN_WIDTH = 72

export function StartupHeader({
  width,
  workingDirectory,
  recentChats,
}: {
  readonly width: number
  readonly workingDirectory: string
  readonly recentChats: ReactNode
}): ReactNode {
  const theme = useTheme()
  const sideBySide = width >= SIDE_BY_SIDE_MIN_WIDTH
  const detailsWidth = sideBySide
    ? Math.max(1, width - COMPACT_MAGNITUDE_LOGO_WIDTH - 5)
    : Math.max(1, width - 2)
  const directoryWidth = Math.max(8, detailsWidth - 'Current directory: '.length)
  const displayedDirectory = truncateToDisplayWidth(workingDirectory, directoryWidth)

  const details = (
    <box style={{ flexDirection: 'column', minWidth: 0, flexShrink: 1 }}>
      <text style={{ fg: theme.foreground }} attributes={TextAttributes.BOLD}>Magnitude</text>
      <text style={{ fg: theme.muted }}>The best way to code with local models</text>
      <box style={{ height: 1, flexShrink: 0 }} />
      <text wrapMode="none">
        <span style={{ fg: theme.muted }}>Current directory: </span>
        <span style={{ fg: theme.muted }} attributes={TextAttributes.BOLD}>{displayedDirectory}</span>
      </text>
      <text wrapMode="none">
        <span style={{ fg: theme.foreground }} attributes={TextAttributes.BOLD}>Tip: </span>
        <span style={{ fg: theme.muted }}>Use </span>
        <span style={{ fg: theme.foreground }}>/settings</span>
        <span style={{ fg: theme.muted }}> to manage models.</span>
      </text>
    </box>
  )

  return (
    <>
      <box style={{
        paddingLeft: 1,
        paddingBottom: recentChats ? 0 : 1,
        flexDirection: sideBySide ? 'row' : 'column',
      }}>
        <CompactMagnitudeLogo />
        <box style={{
          width: sideBySide ? 2 : 0,
          height: sideBySide ? 0 : 1,
          flexShrink: 0,
        }} />
        {details}
      </box>
      {recentChats && <box style={{ paddingLeft: 1, paddingTop: 1 }}>{recentChats}</box>}
    </>
  )
}
