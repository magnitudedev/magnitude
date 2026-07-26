import { renderToStaticMarkup } from 'react-dom/server'
import type { ReactNode } from 'react'
import { describe, expect, it, vi } from 'vitest'
import {
  ModelSlotBlocked,
  ModelSlotLoadingLocalModel,
  PRIMARY_SLOT_ID,
  ProviderIdSchema,
  ProviderModelIdSchema,
  ReasoningEffortSchema,
  type DisplayActorWork,
  type DisplayModelRequestActivity,
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
const selection = {
  providerId: ProviderIdSchema.make("local"),
  providerModelId: ProviderModelIdSchema.make("local:test"),
  reasoningEffort: ReasoningEffortSchema.make("none"),
}

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
    const html = renderToStaticMarkup(
      <ActivityRail
        work={work()}
        width={100}
        waitsForGenerationProgress
        modelLoadActivity={new ModelSlotLoadingLocalModel({
          slotId: PRIMARY_SLOT_ID,
          selection,
          percentage: 42,
        })}
        modelRequestActivity={request()}
      />,
    )
    expect(htmlToText(html)).toBe('⠋ Loading model · 42%')
    expect(html).not.toContain('padding-left')
    expect(html).not.toContain('padding-top')
  })

  it('shows the low-memory stopped message without a spinner', () => {
    expect(text(
      <ActivityRail
        work={work()}
        width={100}
        waitsForGenerationProgress
        modelLoadActivity={new ModelSlotBlocked({
          slotId: PRIMARY_SLOT_ID,
          selection,
          reason: {
            _tag: "LocalModelStoppedLowMemory",
            error: {
              code: "low_memory",
              message: "internal detail",
              retryable: true,
            },
          },
        })}
        modelRequestActivity={request()}
      />,
    )).toBe("Model stopped · Low memory - close memory-intensive apps and try again")
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

  it('does not render completed work from client-local actor state', () => {
    expect(text(
      <ActivityRail
        work={work({ phase: 'worked', lastWorkMs: 5_000 })}
        width={100}
        waitsForGenerationProgress={false}
        modelLoadActivity={null}
        modelRequestActivity={null}
      />,
    )).toBe('')
  })
})
