import type { DisplayMessage } from '@magnitudedev/agent'
import { TextAttributes } from '@opentui/core'
import { useTheme } from '../hooks/use-theme'

type BackgroundProcessMessage = Extract<DisplayMessage, { type: 'background_process' }>

function truncate(text: string, max: number): string {
  const normalized = text.replace(/\s+/g, ' ').trim()
  if (!normalized) return ''
  return normalized.length > max ? normalized.slice(0, max - 3) + '...' : normalized
}

function previewOutput(message: BackgroundProcessMessage): string {
  return truncate(message.stderr || message.stdout, 100)
}

function statusText(message: BackgroundProcessMessage): string {
  if (message.status === 'running') return 'running'
  if (message.signal) return `signal ${message.signal}`
  if (message.exitCode !== null) return `exit ${message.exitCode}`
  return message.status
}

export function BackgroundProcessCard({ message }: { message: BackgroundProcessMessage }) {
  const theme = useTheme()
  const running = message.status === 'running'
  const cmd = truncate(message.command, 60)
  const preview = previewOutput(message)
  const fullOutput = [message.stdout, message.stderr].filter(Boolean).join('\n').trim()

  return (
    <box style={{ flexDirection: 'column', marginBottom: 1 }}>
      <text>
        <span style={{ fg: running ? theme.warning : theme.secondary }}>
          {running ? '◌' : '✓'}
        </span>
        <span style={{ fg: theme.muted }}>{` PID ${message.pid} `}</span>
        <span style={{ fg: theme.foreground }}>{cmd}</span>
        <span style={{ fg: theme.muted }} attributes={TextAttributes.DIM}>{` · ${statusText(message)}`}</span>
      </text>

      {preview && (
        <text style={{ fg: running ? theme.muted : theme.secondary }} attributes={TextAttributes.DIM}>
          {preview}
        </text>
      )}

      {!running && fullOutput && (
        <text style={{ fg: theme.muted }} attributes={TextAttributes.DIM}>
          {fullOutput.length > 500 ? fullOutput.slice(-500) + '\n...(truncated)' : fullOutput}
        </text>
      )}
    </box>
  )
}