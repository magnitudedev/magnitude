import { describe, expect, it } from 'vitest'
import { TurnActive, TurnIdle } from '../src/projections/turn'
import { buildObserverEscalationRunOptions } from '../src/workers/cortex'

const idleTurn = new TurnIdle({
  completedTurns: 0,
  triggers: [],
  pendingInboundCommunications: [],
  parentForkId: null,
  connectionRetryCount: 0,
})

function activeTurn(args: { turnId?: string; requiresAdvisor: boolean }) {
  return new TurnActive({
    completedTurns: 0,
    triggers: [],
    pendingInboundCommunications: [],
    parentForkId: null,
    connectionRetryCount: 0,
    turnId: args.turnId ?? 'turn-1',
    chainId: 'chain-1',
    toolCalls: [],
    triggeredByUser: false,
    requiresAdvisor: args.requiresAdvisor,
  })
}

describe('buildObserverEscalationRunOptions', () => {
  it('returns undefined when this turn did not claim an advisor-required escalation', () => {
    expect(buildObserverEscalationRunOptions(undefined, 'turn-1')).toBeUndefined()
    expect(buildObserverEscalationRunOptions(idleTurn, 'turn-1')).toBeUndefined()
    expect(buildObserverEscalationRunOptions(activeTurn({ requiresAdvisor: false }), 'turn-1')).toBeUndefined()
    expect(buildObserverEscalationRunOptions(activeTurn({ turnId: 'turn-2', requiresAdvisor: true }), 'turn-1')).toBeUndefined()
  })

  it('does not force message_advisor while Observer/Advisor escalation is disabled', () => {
    const options = buildObserverEscalationRunOptions(activeTurn({ requiresAdvisor: true }), 'turn-1')

    expect(options).toBeUndefined()
  })
})
