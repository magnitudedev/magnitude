import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it, vi } from 'vitest'
import type { DisplayMessage } from '@magnitudedev/sdk'
import { Option } from 'effect'
import { MessageView } from './message-view'

vi.mock('../../hooks/use-theme', () => ({
  useTheme: () => ({
    foreground: 'white',
    muted: 'gray',
  }),
}))

const summary = (durationMs: number): Extract<DisplayMessage, { type: 'work_summary' }> => ({
  id: 'work_summary:chain-1',
  type: 'work_summary',
  chainId: 'chain-1',
  durationMs,
  phase: 'worked',
  performance: Option.none(),
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

  it('renders native model performance with the standard separators', () => {
    const message: DisplayMessage = {
      ...summary(6_000),
      performance: Option.some({
        modelDisplayName: 'Qwen3 Coder',
        timeToFirstTokenMs: 6.2,
        decodeTokensPerSecond: 20.46,
      }),
    }
    const html = renderToStaticMarkup(<MessageView message={message} isStreaming={false} />)

    expect(html).toContain('Qwen3 Coder worked for 6 seconds · 6ms ttft · 20.5 tok/s')
  })

  it('keeps long first-request TTFT values compact and visible', () => {
    const message: Extract<DisplayMessage, { type: 'work_summary' }> = {
      ...summary(20_000),
      performance: Option.some({
        modelDisplayName: 'Qwen3 Coder',
        timeToFirstTokenMs: 15_000,
        decodeTokensPerSecond: 20.5,
      }),
    }
    const html = renderToStaticMarkup(<MessageView message={message} isStreaming={false} />)

    expect(html).toContain('· 15.0s ttft ·')
  })
})
