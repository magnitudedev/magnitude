import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it, vi } from 'vitest'
import type { DisplayMessage } from '@magnitudedev/sdk'
import { MessageView } from './message-view'

vi.mock('../../hooks/use-theme', () => ({
  useTheme: () => ({
    foreground: 'white',
    muted: 'gray',
  }),
}))

const summary = (durationMs: number): DisplayMessage => ({
  id: 'work_summary:chain-1',
  type: 'work_summary',
  chainId: 'chain-1',
  durationMs,
  phase: 'worked',
  timestamp: 1,
})

describe('work summary message', () => {
  it('renders in the assistant column with a blank row below', () => {
    const html = renderToStaticMarkup(
      <MessageView message={summary(5_000)} isStreaming={false} />,
    )

    expect(html).toContain('Worked for 5 seconds')
    expect(html).toContain('padding-left:1px')
    expect(html).toContain('margin-bottom:1px')
  })

  it('uses compact minute formatting', () => {
    const html = renderToStaticMarkup(
      <MessageView message={summary(65_000)} isStreaming={false} />,
    )

    expect(html).toContain('Worked for 1:05')
  })
})
