import { describe, expect, it } from 'vitest'
import { Option } from 'effect'
import { forkIdToKey } from '@magnitudedev/protocol'
import { materializeDisplayActors, materializeDisplayTasks, materializeDisplayTimeline } from '../../src/display-view'

describe('display view materializer', () => {
  it('builds protocol timelines from ordinary metadata plus materialized messages', () => {
    const timeline = materializeDisplayTimeline(
      {
        mode: 'streaming',
        streamingMessageId: 'assistant-1',
      },
      [
        {
          id: 'assistant-1',
          type: 'assistant_message',
          content: 'hello',
          timestamp: 101
        }
      ],
      {
        start: 0,
        end: 1,
        totalCount: 1,
        hasMoreBefore: false,
        hasMoreAfter: false,
      },
      {
        mode: 'default',
        entries: [{
          kind: 'message',
          id: 'message:assistant-1',
          messageId: 'assistant-1',
          timestamp: 101,
          role: 'assistant',
          streaming: true,
          interrupted: false,
          nextMessageInterrupted: false,
        }],
        statusSlot: { kind: 'none' },
      },
    )

    expect(timeline).toEqual({
      mode: 'streaming',
      messages: {
        byId: {
          'assistant-1': {
            id: 'assistant-1',
            type: 'assistant_message',
            content: 'hello',
            timestamp: 101
          }
        },
        order: ['assistant-1'],
      },
      streamingMessageId: 'assistant-1',
      window: {
        start: 0,
        end: 1,
        totalCount: 1,
        hasMoreBefore: false,
        hasMoreAfter: false,
      },
      presentation: {
        mode: 'default',
        entries: [{
          kind: 'message',
          id: 'message:assistant-1',
          messageId: 'assistant-1',
          timestamp: 101,
          role: 'assistant',
          streaming: true,
          interrupted: false,
          nextMessageInterrupted: false,
        }],
        statusSlot: { kind: 'none' },
      },
    })
  })

  it('materializes task-linked worker actors with context tokens without live agent status', () => {
    const taskWorker = {
      orderedTaskIds: ['task-1'],
      rows: new Map([
        ['task-1', {
          taskId: 'task-1',
          title: 'Check worker tokens',
          status: 'pending' as const,
          parentId: Option.none<string>(),
          depth: 0,
          updatedAt: 101,
          assignee: {
            kind: 'worker' as const,
            role: 'engineer',
            agentId: 'agent-1',
            forkId: 'fork-1',
          },
          workerState: { status: 'unassigned' as const },
        }],
      ]),
      workerActivityByForkId: new Map(),
    }

    const idleRootWork = {
      phase: 'idle' as const,
      chainStartedAt: null,
      lastChainMs: 0,
      activity: null,
      activeChildCount: 0,
      _currentTurnId: null,
      _thinkingCharCount: null,
      _activeToolKey: null,
    }

    const actors = materializeDisplayActors(
      { agents: new Map(), agentByForkId: new Map(), rootWork: idleRootWork },
      taskWorker,
      { forks: new Map<string | null, { tokenEstimate: number }>([[null, { tokenEstimate: 1200 }], ['fork-1', { tokenEstimate: 4600 }]]) },
      { forks: new Map() },
    )
    const tasks = materializeDisplayTasks(taskWorker)
    const actorKey = forkIdToKey('fork-1')

    expect(actors[actorKey]).toMatchObject({
      kind: 'worker',
      name: 'Check worker tokens',
      role: 'engineer',
      taskId: 'task-1',
      context: { tokenEstimate: 4600, isCompacting: false },
    })
    expect(tasks.byId['task-1']?.assignee).toMatchObject({
      kind: 'actor',
      actorKey,
      taskState: 'assigned',
    })
  })
})
