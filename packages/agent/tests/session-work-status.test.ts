import { describe, expect, it } from 'vitest'
import { Option } from 'effect'
import type { AgentLifecycleState } from '../src/projections/agent-lifecycle'
import { CompactionIdle, Compacting, PendingInjection } from '../src/projections/compaction'
import { TurnActive, TurnIdle, TurnWaitingForUser, type ForkTurnState } from '../src/projections/turn'
import { deriveSessionWorkStatus } from '../src/session-work-status'

const agents = (working: boolean): AgentLifecycleState => ({
  agents: working
    ? new Map([
        [
          'agent',
          {
            agentId: 'agent',
            forkId: 'fork',
            parentForkId: null,
            name: 'worker',
            role: 'engineer',
            context: '',
            mode: 'spawn',
            taskId: 'task',
            message: null,
            status: 'working',
            lastIdleReason: null,
          },
        ],
      ])
    : new Map(),
  agentByForkId: working ? new Map([['fork', 'agent']]) : new Map(),
  rootWork: {
    phase: 'idle',
    chainStartedAt: null,
    lastChainMs: 0,
    activity: null,
    activeChildCount: 0,
    _currentTurnId: null,
    _thinkingCharCount: null,
    _activeToolKey: null,
  },
})

const idleTurn = () =>
  new TurnIdle({
    completedTurns: 0,
    triggers: [],
    pendingInboundCommunications: [],
    parentForkId: null,
    connectionRetryCount: 0,
  })

const idleCompaction = () => new CompactionIdle({ contextLimitBlocked: false, shouldCompact: false })

const status = (overrides: {
  turn?: ForkTurnState
  compaction?: CompactionIdle | Compacting | PendingInjection
  workingAgent?: boolean
  detached?: number
}) =>
  deriveSessionWorkStatus({
    turns: new Map([[null, overrides.turn ?? idleTurn()]]),
    agents: agents(overrides.workingAgent ?? false),
    compactions: new Map([[null, overrides.compaction ?? idleCompaction()]]),
    detachedProcessCount: overrides.detached ?? 0,
  })

describe('deriveSessionWorkStatus', () => {
  it('is quiescent for an idle or waiting-for-user session', () => {
    expect(status({})._tag).toBe('Quiescent')
    expect(
      status({
        turn: new TurnWaitingForUser({
          ...idleTurn(),
        }),
      })._tag
    ).toBe('Quiescent')
  })

  it('recognizes active turns and queued delayed retry triggers', () => {
    expect(
      status({
        turn: new TurnActive({
          ...idleTurn(),
          turnId: 'turn',
          chainId: 'chain',
          toolCalls: [],
          triggeredByUser: true,
          requiresAdvisor: false,
        }),
      })._tag
    ).toBe('Working')
    expect(
      status({
        turn: new TurnIdle({
          ...idleTurn(),
          triggers: [
            {
              _tag: 'chain_continue',
              chainId: 'chain',
              notBefore: Option.some(Date.now() + 60_000),
            },
          ],
        }),
      })._tag
    ).toBe('Working')
  })

  it('recognizes worker, compaction policy, and live process work', () => {
    expect(status({ workingAgent: true })).toEqual({
      _tag: 'Working',
      workerCount: 1,
    })
    expect(
      status({
        compaction: new CompactionIdle({
          contextLimitBlocked: false,
          shouldCompact: true,
        }),
      })._tag
    ).toBe('Working')
    expect(status({ detached: 1 })._tag).toBe('Working')
  })
})
