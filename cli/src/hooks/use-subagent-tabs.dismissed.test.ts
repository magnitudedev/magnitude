import { describe, expect, test } from 'bun:test'
import type { DisplayMessage } from '@magnitudedev/agent'
import { buildSubagentTabItem } from './use-subagent-tabs'

function filterDismissedIdleTabs(
  tabs: ReturnType<typeof buildSubagentTabItem>[],
  dismissedIdleForkIds: ReadonlySet<string>
) {
  return tabs.filter((tab) => !(tab.phase === 'idle' && dismissedIdleForkIds.has(tab.forkId)))
}

describe('dismissed idle tab persistence', () => {
  test('keeps dismissed idle tab hidden across unrelated display updates', () => {
    const tab = buildSubagentTabItem({
      forkId: 'fork-idle',
      meta: {
        agentId: 'agent-idle',
        name: 'Idle',
        activeSince: 1000,
        accumulatedActiveMs: 1000,
        completedAt: 2000,
        resumeCount: 0,
        toolCount: 0,
        toolCounts: {} as any,
        phase: 'idle',
      },
      messages: [] as DisplayMessage[],
      pendingDirect: { pending: false, since: null },
    })

    const visible = filterDismissedIdleTabs([tab], new Set(['fork-idle']))
    expect(visible).toEqual([])
  })

  test('auto-unhides when tab becomes active', () => {
    const tab = buildSubagentTabItem({
      forkId: 'fork-idle',
      meta: {
        agentId: 'agent-idle',
        name: 'Idle',
        activeSince: 1000,
        accumulatedActiveMs: 1000,
        completedAt: 2000,
        resumeCount: 0,
        toolCount: 0,
        toolCounts: {} as any,
        phase: 'idle',
      },
      messages: [] as DisplayMessage[],
      pendingDirect: { pending: true, since: 3000 },
    })

    const visible = filterDismissedIdleTabs([tab], new Set(['fork-idle']))
    expect(visible).toHaveLength(1)
    expect(visible[0]?.phase).toBe('active')
  })
})
