import { describe, expect, it } from 'bun:test'
import { createAgentTestHarness } from '../../test-harness/harness'
import { textParts } from '../../content'
import { createId } from '../../util/id'
import { DisplayProjection } from '../display'
import { TurnProjection } from '../turn'

const ts = (n: number) => 1_700_000_000_000 + n

describe('display pending communications promotion', () => {
  it('queues inbound pending, mirrors to display, promotes on turn_started, and clears pending', async () => {
    const harness = await createAgentTestHarness()
    const stopRouting = await harness.script.route({
      root: { xml: '<yield/>' },
      subagents: { xml: '<yield/>' },
    })

    try {
      await harness.send({
        type: 'task_created',
        forkId: null,
        taskId: 'task-1',
        title: 'Task 1',
        taskType: 'implement',
        parentId: null,
        timestamp: ts(1),
      } as any)
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
        taskId: 'task-1',
        context: '',
      } as any)
      await harness.send({
        type: 'task_assigned',
        forkId: null,
        taskId: 'task-1',
        assignee: 'builder',
        workerRole: 'builder',
        message: 'do it',
        workerInfo: { agentId: 'agent-1', forkId: 'fork-1', role: 'builder' },
        timestamp: ts(1),
      } as any)

      await harness.wait.agentCreated((e) => e.agentId === 'agent-1')

      await harness.send({
        type: 'message_start',
        id: 'm1',
        timestamp: ts(2),
        forkId: null,
        turnId: 't-parent',
        scope: 'task',
        taskId: 'task-1',
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

      await harness.send({
        type: 'turn_started',
        timestamp: ts(5),
        forkId: 'fork-1',
        turnId: 't-sub-1',
        chainId: 'c1',
      } as any)

      await harness.wait.until('pending inbound promoted/cleared', async () => {
        const display = await harness.projectionFork(DisplayProjection.Tag, 'fork-1')
        return display.pendingInboundCommunications.length === 0
      })

      const workingAfter = await harness.projectionFork(TurnProjection.Tag, 'fork-1')
      expect(workingAfter.pendingInboundCommunications.length).toBe(0)

      const displayAfter = await harness.projectionFork(DisplayProjection.Tag, 'fork-1')
      expect(displayAfter.pendingInboundCommunications.length).toBe(0)

      await harness.send({
        type: 'interrupt',
        forkId: 'fork-1',
      } as any)
    } finally {
      stopRouting()
      await harness.dispose()
    }
  })

  it('renders direct user→subagent message as user_message (not communication step)', async () => {
    const harness = await createAgentTestHarness()
    const stopRouting = await harness.script.route({
      root: { xml: '<yield/>' },
      subagents: { xml: '<yield/>' },
    })

    try {
      await harness.send({
        type: 'agent_created',
        timestamp: ts(8),
        forkId: 'fork-user',
        parentForkId: null,
        agentId: 'agent-user',
        role: 'builder',
        name: 'BuilderUser',
        message: '',
        mode: 'manual',
      } as any)

      const messageId = createId()
      await harness.send({
        type: 'user_message',
        messageId,
        timestamp: ts(9),
        forkId: 'fork-user',
        content: textParts('hello subagent'),
        mode: 'text',
        synthetic: false,
        taskMode: false,
        attachments: [],
      } as any)
      await harness.send({
        type: 'user_message_ready',
        messageId,
        forkId: 'fork-user',
        resolvedMentions: [],
      } as any)

      const display = await harness.projectionFork(DisplayProjection.Tag, 'fork-user')
      const userMessages = display.messages.filter(
        m => m.type === 'user_message' || m.type === 'queued_user_message'
      )
      expect(
        userMessages.some(m => (m as any).content.includes('hello subagent'))
      ).toBe(true)

      const communicationSteps = display.messages.flatMap(m => m.type === 'think_block' ? m.steps : [])
        .filter(s => s.type === 'communication' && (s as any).content?.includes('hello subagent'))
      expect(communicationSteps.length).toBe(0)
    } finally {
      stopRouting()
      await harness.dispose()
    }
  })

  it('does not duplicate promoted inbound messages across subsequent turns and does not leak to root pending', async () => {
    const harness = await createAgentTestHarness()
    const stopRouting = await harness.script.route({
      root: { xml: '<yield/>' },
      subagents: { xml: '<yield/>' },
    })

    try {
      await harness.send({
        type: 'task_created',
        forkId: null,
        taskId: 'task-2',
        title: 'Task 2',
        taskType: 'implement',
        parentId: null,
        timestamp: ts(11),
      } as any)
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
        taskId: 'task-2',
        context: '',
      } as any)
      await harness.send({
        type: 'task_assigned',
        forkId: null,
        taskId: 'task-2',
        assignee: 'builder',
        workerRole: 'builder',
        message: 'do it',
        workerInfo: { agentId: 'agent-2', forkId: 'fork-2', role: 'builder' },
        timestamp: ts(11),
      } as any)

      await harness.wait.agentCreated((e) => e.agentId === 'agent-2')

      await harness.send({
        type: 'message_start',
        id: 'm2',
        timestamp: ts(12),
        forkId: null,
        turnId: 't-parent-2',
        scope: 'task',
        taskId: 'task-2',
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

      await harness.wait.until('single promotion', async () => {
        const display = await harness.projectionFork(DisplayProjection.Tag, 'fork-2')
        const allSteps = display.messages.flatMap(m => m.type === 'think_block' ? m.steps : [])
        const promoted = allSteps.filter(
          s => s.type === 'communication' && s.direction === 'from_agent' && (s as any).content.includes('once only')
        )
        return promoted.length === 1
      })

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
      stopRouting()
      await harness.dispose()
    }
  })
})