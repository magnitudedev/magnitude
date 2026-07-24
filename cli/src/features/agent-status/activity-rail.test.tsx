import { renderToStaticMarkup } from 'react-dom/server'
import type { ReactNode } from 'react'
import { describe, expect, it, vi } from 'vitest'
import type {
  DisplayActorWork,
  DisplayModelRequestActivity,
} from '@magnitudedev/sdk'
import { ActivityRail } from './activity-rail'

vi.mock('../../hooks/use-theme', () => ({
  useTheme: () => ({
    primary: 'cyan',
    foreground: 'white',
    muted: 'gray',
  }),
}))

const htmlToText = (html: string): string =>
  html.replace(/<[^>]*>/g, '').replace(/\s+/g, ' ').trim()

const text = (node: ReactNode): string => htmlToText(renderToStaticMarkup(<>{node}</>))

const work = (overrides: Partial<DisplayActorWork> = {}): DisplayActorWork => ({
  phase: 'working',
  activeSince: 1_000,
  lastWorkMs: 0,
  accumulatedMs: 0,
  resumeCount: 0,
  activity: null,
  activeChildCount: 0,
  ...overrides,
})

const request = (
  overrides: Partial<DisplayModelRequestActivity> = {},
): DisplayModelRequestActivity => ({
  requestId: 'request-1',
  turnId: 'turn-1',
  forkId: null,
  startedAt: 1_000,
  phase: 'prefill',
  completedTokens: 0,
  totalTokens: 1_100,
  cachedTokens: 0,
  ...overrides,
})

describe('activity rail', () => {
  it('shows model loading immediately instead of Working', () => {
    expect(text(
      <ActivityRail
        work={work()}
        width={100}
        waitsForGenerationProgress
        modelLoadActivity={{ percentage: 42, text: 'Loading model · 42%' }}
        modelRequestActivity={request()}
      />,
    )).toBe('⠋ Loading model · 42%')
  })

  it('keeps the reserved row blank during the request anti-flicker delay', () => {
    vi.spyOn(Date, 'now').mockReturnValue(1_400)
    const html = renderToStaticMarkup(
      <ActivityRail
        work={work()}
        width={100}
        waitsForGenerationProgress
        modelLoadActivity={null}
        modelRequestActivity={request()}
      />,
    )
    expect(htmlToText(html)).toBe('')
    expect(html).toContain('height:1')
    vi.restoreAllMocks()
  })

  it('shows cached prefill as one line after the delay', () => {
    vi.spyOn(Date, 'now').mockReturnValue(2_000)
    expect(text(
      <ActivityRail
        work={work()}
        width={100}
        waitsForGenerationProgress
        modelLoadActivity={null}
        modelRequestActivity={request({
          completedTokens: 14_020,
          totalTokens: 14_300,
          cachedTokens: 13_200,
        })}
      />,
    )).toBe('⠋ Adding new messages · 820 / 1.1k tokens · Reusing 13.2k tokens')
    vi.restoreAllMocks()
  })

  it('drops trailing cached detail on narrow terminals', () => {
    vi.spyOn(Date, 'now').mockReturnValue(2_000)
    expect(text(
      <ActivityRail
        work={work()}
        width={60}
        waitsForGenerationProgress
        modelLoadActivity={null}
        modelRequestActivity={request({
          completedTokens: 14_020,
          totalTokens: 14_300,
          cachedTokens: 13_200,
        })}
      />,
    )).toBe('⠋ Adding new messages · 820 / 1.1k tokens')
    vi.restoreAllMocks()
  })

  it('starts Working at response generation and retains thinking details', () => {
    vi.spyOn(Date, 'now').mockReturnValue(16_000)
    expect(text(
      <ActivityRail
        work={work({
          respondingSince: 11_000,
          activity: { kind: 'thinking', message: 'Thinking' },
        })}
        width={100}
        waitsForGenerationProgress
        modelLoadActivity={null}
        modelRequestActivity={null}
      />,
    )).toBe('● Working... 0:05 · Thinking')
    vi.restoreAllMocks()
  })

  it('keeps the actor timer fallback for providers without generation progress', () => {
    vi.spyOn(Date, 'now').mockReturnValue(6_000)
    expect(text(
      <ActivityRail
        work={work({ activeSince: 1_000 })}
        width={100}
        waitsForGenerationProgress={false}
        modelLoadActivity={null}
        modelRequestActivity={null}
      />,
    )).toBe('● Working... 0:05')
    vi.restoreAllMocks()
  })
})
