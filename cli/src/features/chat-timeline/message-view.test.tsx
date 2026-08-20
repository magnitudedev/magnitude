import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it, vi } from 'vitest'
import type { DisplayMessage } from '@magnitudedev/sdk'
import { Option } from 'effect'
import { MessageView } from './message-view'
import { defaultCliThemes } from '../../utils/theme'

vi.mock('../../hooks/use-theme', () => ({
  useTheme: () => defaultCliThemes.dark,
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

  it('avoids showing zero seconds for sub-second work', () => {
    const html = renderToStaticMarkup(
      <MessageView message={summary(999)} isStreaming={false} />,
    )

    expect(html).toContain('Worked for &lt;1 second')
    expect(html).not.toContain('Worked for 0 seconds')
  })

  it('renders native model throughput with the standard separator', () => {
    const message: DisplayMessage = {
      ...summary(6_000),
      performance: Option.some({
        modelDisplayName: 'Qwen3 Coder',
        decodeTokensPerSecond: Option.some(20.46),
      }),
    }
    const html = renderToStaticMarkup(<MessageView message={message} isStreaming={false} />)

    expect(html).toContain('Qwen3 Coder worked for 6 seconds · 20.5 tok/s')
    expect(html).not.toContain('ttft')
  })

  it('renders the cloud model name without unavailable throughput', () => {
    const message: DisplayMessage = {
      ...summary(1_000),
      performance: Option.some({
        modelDisplayName: 'DeepSeek V4 Flash',
        decodeTokensPerSecond: Option.none(),
      }),
    }
    const html = renderToStaticMarkup(<MessageView message={message} isStreaming={false} />)

    expect(html).toContain('DeepSeek V4 Flash worked for 1 second')
    expect(html).not.toContain('tok/s')
  })
})

describe('thinking message visibility', () => {
  const thinking: DisplayMessage = {
    id: 'thinking:turn-1:1',
    type: 'thinking',
    content: 'Internal reasoning',
    label: Option.none(),
    phase: 'completed',
    completedAt: Option.some(2_000),
    timestamp: 1_000,
  }

  it('keeps thinking hidden in normal CLI mode', () => {
    const html = renderToStaticMarkup(
      <MessageView message={thinking} isStreaming={false} mode="default" />,
    )
    expect(html).not.toContain('Internal reasoning')
  })

  it('renders thinking in CLI transcript mode', () => {
    const html = renderToStaticMarkup(
      <MessageView message={thinking} isStreaming={false} mode="transcript" />,
    )
    expect(html).toContain('Internal reasoning')
  })
})
