import { renderToStaticMarkup } from 'react-dom/server'
import type { ComponentProps, ReactNode } from 'react'
import { describe, expect, it, vi } from 'vitest'
import { Option } from 'effect'
import {
  ModelSlotConfiguredLocal,
  ModelInstanceIdSchema,
  PRIMARY_SLOT_ID,
  ProviderIdSchema,
  ProviderModelIdSchema,
  ReasoningEffortSchema,
  type DisplayRootStatus,
} from '@magnitudedev/sdk'
import { defaultCliThemes } from '../../utils/theme'

vi.mock('../../hooks/use-theme', () => ({
  useTheme: () => defaultCliThemes.dark,
}))
vi.mock('@opentui/react', async () => ({
  ...await vi.importActual<typeof import('@opentui/react')>('@opentui/react'),
  useRenderer: () => ({ setMousePointer: () => {} }),
}))
vi.mock('../../components/button', () => ({
  Button: ({ children }: { children: ReactNode }) => <>{children}</>,
}))

const { ActivityRail: ActivityRailView } = await import('./activity-rail')
const ActivityRail = (
  props: Omit<ComponentProps<typeof ActivityRailView>, 'onStopModel'>,
) => <ActivityRailView {...props} onStopModel={() => {}} />

const text = (node: ReactNode): string => renderToStaticMarkup(<>{node}</>)
  .replace(/<[^>]*>/g, '')
  .replace(/\s+/g, ' ')
  .trim()

const working = (
  detail: Extract<DisplayRootStatus, { readonly _tag: 'Working' }>['detail'],
): DisplayRootStatus => ({
  _tag: 'Working',
  chainStartedAt: 1_000,
  detail,
  activeChildCount: 0,
})

const selection = {
  providerId: ProviderIdSchema.make('local'),
  providerModelId: ProviderModelIdSchema.make('test-configuration'),
  reasoningEffort: ReasoningEffortSchema.make('none'),
}
const instanceId = ModelInstanceIdSchema.make('test-instance')

const localActivity = (lifecycle: {
  readonly _tag: 'Requested'
} | {
  readonly _tag: 'Loading'
  readonly stage: 'loading'
  readonly progress: Option.Option<number>
  readonly plannedAllocation: Option.Option<never>
} | {
  readonly _tag: 'Failed'
  readonly failure: { readonly code: string; readonly message: string; readonly retryable: boolean }
}) => new ModelSlotConfiguredLocal({
  slotId: PRIMARY_SLOT_ID,
  selection,
  descriptor: {
    providerId: selection.providerId,
    providerModelId: selection.providerModelId,
    displayName: 'Local test',
    variantLabel: Option.none(),
  },
  availability: { _tag: 'Available' },
  residency: lifecycle._tag === 'Loading'
    ? { ...lifecycle, instanceId }
    : lifecycle,
  actions: lifecycle._tag === 'Loading' || lifecycle._tag === 'Requested' ? ['Stop'] : [],
})

describe('activity rail', () => {
  it('shows model loading in place of Working', () => {
    expect(text(<ActivityRail
      status={working({ _tag: 'NoDetail' })}
      width={100}
      modelLoadActivity={localActivity({
        _tag: 'Loading',
        stage: 'loading',
        progress: Option.some(0.42),
        plannedAllocation: Option.none(),
      })}
    />)).toBe('⠋ Loading model · 42% · Stop')
  })

  it('shows an admitted load request as loading rather than waiting', () => {
    expect(text(<ActivityRail
      status={working({ _tag: 'WaitingForModel', turnStartedAt: 1_000 })}
      width={100}
      modelLoadActivity={localActivity({ _tag: 'Requested' })}
    />)).toBe('⠋ Loading model · 0% · Stop')
  })

  it('shows the durable low-memory message without a spinner', () => {
    expect(text(<ActivityRail
      status={working({ _tag: 'NoDetail' })}
      width={100}
      modelLoadActivity={localActivity({
        _tag: 'Failed',
        failure: { code: 'low_memory', message: 'internal detail', retryable: true },
      })}
    />)).toBe('■ Model stopped · Low memory - close memory-intensive apps and try again')
  })

  it('renders live cached prefill within the Working grammar', () => {
    vi.spyOn(Date, 'now').mockReturnValue(5_000)
    expect(text(<ActivityRail
      status={working({
        _tag: 'Prefill',
        completedTokens: 14_020,
        totalTokens: 14_300,
        cachedTokens: 13_200,
      })}
      width={100}
      modelLoadActivity={null}
    />)).toBe('● Working · 0:04 · Prefill · 820 / 1.1k tokens · 13.2k cached')
    vi.restoreAllMocks()
  })

  it('renders no secondary detail immediately with the chain timer', () => {
    vi.spyOn(Date, 'now').mockReturnValue(2_000)
    expect(text(<ActivityRail
      status={working({ _tag: 'NoDetail' })}
      width={100}
      modelLoadActivity={null}
    />)).toBe('● Working · 0:01')
    vi.restoreAllMocks()
  })

  it('hides Waiting for model during the first second of a turn', () => {
    vi.spyOn(Date, 'now').mockReturnValue(1_500)
    expect(text(<ActivityRail
      status={working({ _tag: 'WaitingForModel', turnStartedAt: 1_000 })}
      width={100}
      modelLoadActivity={null}
    />)).toBe('● Working · 0:00')
    vi.restoreAllMocks()
  })

  it('shows Waiting for model after the turn-relative threshold', () => {
    vi.spyOn(Date, 'now').mockReturnValue(2_100)
    expect(text(<ActivityRail
      status={working({ _tag: 'WaitingForModel', turnStartedAt: 1_000 })}
      width={100}
      modelLoadActivity={null}
    />)).toBe('● Working · 0:01 · Waiting for model')
    vi.restoreAllMocks()
  })
})
