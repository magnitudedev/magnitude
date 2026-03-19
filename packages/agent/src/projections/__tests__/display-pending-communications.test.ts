import { describe, expect, it } from 'bun:test'
import { createAgentTestHarness } from '../../test-harness/harness'
import { DisplayProjection } from '../display'
import { WorkingStateProjection } from '../working-state'

const ts = (n: number) => 1_700_000_000_000 + n

describe('display pending communications promotion', () => {
  it('queues inbound pending, mirrors to display, promotes on turn_started, and clears pending', async () => {
    const harness = await createAgentTestHarness()
    try {
      await harness.send({
        type: 'agent_created',
        timestamp: ts(1),
        forkId: 'fork-1',
        parentForkId: null,
        agentId: 'agent-1',
        role: 'builder',
        name: 'Builder',
        message: 'ready',
        mode: 'manual',
      } as any)

      await harness.send({
        type: 'message_start',
        id: 'm1',
        timestamp: ts(2),
        forkId: null,
        turnId: 't-parent',
        dest: 'agent-1',
      } as any)
      await harness.send({
        type: 'message_chunk',
        id: 'm1',
        timestamp: ts(3),
        forkId: null,
        turnId: 't-parent',
        text: 'hello from parent',
      } as any)
      await harness.send({
        type: 'message_end',
        id: 'm1',
        timestamp: ts(4),
        forkId: null,
        turnId: 't-parent',
      } as any)

      const workingBefore = await harness.projectionFork(WorkingStateProjection.Tag, 'fork-1')
      expect(workingBefore.pendingInboundCommunications.length).toBe(1)

      const displayBefore = await harness.projectionFork(DisplayProjection.Tag, 'fork-1')
      expect(displayBefore.pendingInboundCommunications.length).toBe(1)
      const timelineBefore = displayBefore.messages.flatMap(m => m.type === 'think_block' ? m.steps : [])
      expect(
        timelineBefore.some(
          s => s.type === 'communication'
            && s.direction === 'from_agent'
            && (s as any).content.includes('hello from parent')
        )
      ).toBe(false)

      await harness.send({
        type: 'turn_started',
        timestamp: ts(5),
        forkId: 'fork-1',
        turnId: 't-sub-1',
        chainId: 'c1',
      } as any)

      const workingAfter = await harness.projectionFork(WorkingStateProjection.Tag, 'fork-1')
      expect(workingAfter.pendingInboundCommunications.length).toBe(0)

      const displayAfter = await harness.projectionFork(DisplayProjection.Tag, 'fork-1')
      expect(displayAfter.pendingInboundCommunications.length).toBe(0)

      const timelineAfter = displayAfter.messages.flatMap(m => m.type === 'think_block' ? m.steps : [])
      const inboundSteps = timelineAfter.filter(s => s.type === 'communication' && s.direction === 'from_agent')
      expect(inboundSteps.length).toBeGreaterThanOrEqual(1)
      const promotedHello = inboundSteps.find(s => (s as any).content.includes('hello from parent'))
      expect(promotedHello).toBeDefined()
    } finally {
      await harness.dispose()
    }
  })

  it('does not duplicate promoted inbound messages across subsequent turns and does not leak to root pending', async () => {
    const harness = await createAgentTestHarness()
    try {
      await harness.send({
        type: 'agent_created',
        timestamp: ts(11),
        forkId: 'fork-2',
        parentForkId: null,
        agentId: 'agent-2',
        role: 'builder',
        name: 'Builder2',
        message: 'ready',
        mode: 'manual',
      } as any)

      await harness.send({
        type: 'message_start',
        id: 'm2',
        timestamp: ts(12),
        forkId: null,
        turnId: 't-parent-2',
        dest: 'agent-2',
      } as any)
      await harness.send({
        type: 'message_chunk',
        id: 'm2',
        timestamp: ts(13),
        forkId: null,
        turnId: 't-parent-2',
        text: 'once only',
      } as any)
      await harness.send({
        type: 'message_end',
        id: 'm2',
        timestamp: ts(14),
        forkId: null,
        turnId: 't-parent-2',
      } as any)

      await harness.send({
        type: 'turn_started',
        timestamp: ts(15),
        forkId: 'fork-2',
        turnId: 't-sub-a',
        chainId: 'ca',
      } as any)

      await harness.send({
        type: 'turn_started',
        timestamp: ts(16),
        forkId: 'fork-2',
        turnId: 't-sub-b',
        chainId: 'cb',
      } as any)

      const display = await harness.projectionFork(DisplayProjection.Tag, 'fork-2')
      const allSteps = display.messages.flatMap(m => m.type === 'think_block' ? m.steps : [])
      const promoted = allSteps.filter(
        s => s.type === 'communication' && s.direction === 'from_agent' && (s as any).content.includes('once only')
      )
      expect(promoted.length).toBe(1)
      expect(display.pendingInboundCommunications.length).toBe(0)

      const rootDisplay = await harness.projectionFork(DisplayProjection.Tag, null)
      expect(rootDisplay.pendingInboundCommunications.length).toBe(0)
    } finally {
      await harness.dispose()
    }
  })
})