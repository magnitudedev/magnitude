import { memo } from 'react'
import { TextAttributes } from '@opentui/core'
import { useTheme } from '../../../hooks/use-theme'
import type { SystemMessage } from '@magnitudedev/client-common'

/**
 * Renders a CLI-local system banner (slash-command feedback, status banners).
 * These are not part of the server-projected timeline — the CLI container
 * merges them into the scrollback as `kind: 'system'` rows.
 */
export const SystemMessageRow = memo(function SystemMessageRow({ message }: { message: SystemMessage }) {
  const theme = useTheme()
  return (
    <box style={{ marginBottom: 1 }}>
      <text attributes={TextAttributes.DIM}>
        <span style={{ fg: theme.muted }}>{message.text}</span>
      </text>
    </box>
  )
})
