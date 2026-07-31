import { renderToStaticMarkup } from 'react-dom/server'
import type { ComponentProps, ReactNode } from 'react'
import { describe, expect, it, vi } from 'vitest'
import { Option } from 'effect'
import { TextAttributes } from '@opentui/core'
import {
  ModelSlotConfiguredLocal,
  ModelInstanceIdSchema,
  ModelServingConfigurationIdSchema,
  PRIMARY_SLOT_ID,
  ProviderIdSchema,
  ProviderModelIdSchema,
  ReasoningEffortSchema,
  type DisplayActorWork,
  type DisplayModelRequestActivity,
} from '@magnitudedev/sdk'
vi.mock('../../hooks/use-theme', () => ({
  useTheme: () => ({
    primary: 'cyan',
    foreground: 'white',
    muted: 'gray',
  }),
}))
vi.mock('@opentui/react', async () => {
  const actual = await vi.importActual<typeof import('@opentui/react')>('@opentui/react')
  return {
    ...actual,
    useRenderer: () => ({ setMousePointer: () => {} }),
  }
})
vi.mock('../../components/button', () => ({
  Button: ({ children }: { children: ReactNode }) => <>{children}</>,
}))

const { ActivityRail: ActivityRailView } = await import('./activity-rail')
const ActivityRail = (
  props: Omit<ComponentProps<typeof ActivityRailView>, "onStopModel">,
) => <ActivityRailView {...props} onStopModel={() => {}} />

const htmlToText = (html: string): string =>
  html.replace(/<[^>]*>/g, '').replace(/\s+/g, ' ').trim()

const text = (node: ReactNode): string => htmlToText(renderToStaticMarkup(<>{node}</>))
const selection = {
  providerId: ProviderIdSchema.make("local"),
  providerModelId: ProviderModelIdSchema.make("test-configuration"),
  reasoningEffort: ReasoningEffortSchema.make("none"),
}
const instanceId = ModelInstanceIdSchema.make("test-instance")
const configurationId = ModelServingConfigurationIdSchema.make("test-configuration")
const localActivity = (
  lifecycle:
    | {
        readonly _tag: "Loading"
        readonly stage: "loading"
        readonly progress: Option.Option<number>
        readonly plannedAllocation: Option.Option<never>
      }
    | {
        readonly _tag: "Failed"
        readonly failure: { readonly code: string; readonly message: string; readonly retryable: boolean }
      },
) => new ModelSlotConfiguredLocal({
  slotId: PRIMARY_SLOT_ID,
  selection,
  descriptor: {
    providerId: selection.providerId,
    providerModelId: selection.providerModelId,
    displayName: "Local test",
  },
  availability: { _tag: "Available" },
  instance: Option.some({ id: instanceId, configurationId, lifecycle }),
  actions: lifecycle._tag === "Loading" ? ["Stop"] : [],
})

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
        modelLoadActivity={localActivity({
          _tag: "Loading",
          stage: "loading",
          progress: Option.some(0.42),
          plannedAllocation: Option.none(),
        })}
        modelRequestActivity={request()}
      />,
    )
    expect(htmlToText(html)).toBe('⠋ Loading model into memory · 42% Stop')
    expect(html).toContain('  Stop')
    expect(html).toContain(`attributes="${TextAttributes.DIM}"`)
    expect(html).not.toContain('padding-left')
    expect(html).not.toContain('padding-top')
  })

  it('shows the low-memory stopped message without a spinner', () => {
    expect(text(
      <ActivityRail
        work={work()}
        width={100}
        modelLoadActivity={localActivity({
          _tag: "Failed",
          failure: {
            code: "low_memory",
            message: "internal detail",
            retryable: true,
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
        work={work({ phase: 'waiting_for_model', activeSince: null })}
        width={100}
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
        work={work({ phase: 'waiting_for_model', activeSince: null })}
        width={100}
        modelLoadActivity={null}
        modelRequestActivity={request({
          completedTokens: 14_020,
          totalTokens: 14_300,
          cachedTokens: 13_200,
        })}
      />,
    )).toBe('⠋ Prefilling · 820 / 1.1k input tokens · Using 13.2k cached tokens')
    vi.restoreAllMocks()
  })

  it('drops trailing cached detail on narrow terminals', () => {
    vi.spyOn(Date, 'now').mockReturnValue(2_000)
    expect(text(
      <ActivityRail
        work={work()}
        width={60}
        modelLoadActivity={null}
        modelRequestActivity={request({
          completedTokens: 14_020,
          totalTokens: 14_300,
          cachedTokens: 13_200,
        })}
      />,
    )).toBe('⠋ Prefilling · 820 / 1.1k input tokens')
    vi.restoreAllMocks()
  })

  it('starts Working at response generation and retains thinking details', () => {
    vi.spyOn(Date, 'now').mockReturnValue(16_000)
    expect(text(
      <ActivityRail
        work={work({
          activeSince: 11_000,
          activity: { kind: 'thinking', message: 'Thinking' },
        })}
        width={100}
        modelLoadActivity={null}
        modelRequestActivity={null}
      />,
    )).toBe('● Working... 0:05 · Thinking')
    vi.restoreAllMocks()
  })

  it('adds the current productive interval to accumulated work', () => {
    vi.spyOn(Date, 'now').mockReturnValue(6_000)
    expect(text(
      <ActivityRail
        work={work({ activeSince: 4_000, accumulatedMs: 3_000 })}
        width={100}
        modelLoadActivity={null}
        modelRequestActivity={null}
      />,
    )).toBe('● Working... 0:05')
    vi.restoreAllMocks()
  })

  it('shows a generic model wait instead of a frozen Working timer', () => {
    vi.spyOn(Date, 'now').mockReturnValue(10_000)
    expect(text(
      <ActivityRail
        work={work({
          phase: 'waiting_for_model',
          activeSince: null,
          accumulatedMs: 3_000,
        })}
        width={100}
        modelLoadActivity={null}
        modelRequestActivity={null}
      />,
    )).toBe('⠋ Waiting for model · 0:03 worked')
    vi.restoreAllMocks()
  })

  it('shows a generic model wait without a zero duration before generation starts', () => {
    vi.spyOn(Date, 'now').mockReturnValue(10_000)
    expect(text(
      <ActivityRail
        work={work({
          phase: 'waiting_for_model',
          activeSince: null,
          accumulatedMs: 0,
        })}
        width={100}
        modelLoadActivity={null}
        modelRequestActivity={null}
      />,
    )).toBe('⠋ Waiting for model')
    vi.restoreAllMocks()
  })

  it('keeps the timer running when a worker overlaps a model wait', () => {
    vi.spyOn(Date, 'now').mockReturnValue(10_000)
    expect(text(
      <ActivityRail
        work={work({
          phase: 'waiting_for_model',
          activeSince: 8_000,
          accumulatedMs: 3_000,
          activeChildCount: 1,
        })}
        width={100}
        modelLoadActivity={null}
        modelRequestActivity={null}
      />,
    )).toBe('● Working... 0:05 · 1 worker running')
    vi.restoreAllMocks()
  })

  it('does not render completed work from client-local actor state', () => {
    expect(text(
      <ActivityRail
        work={work({ phase: 'worked', lastWorkMs: 5_000 })}
        width={100}
        modelLoadActivity={null}
        modelRequestActivity={null}
      />,
    )).toBe('')
  })
})
