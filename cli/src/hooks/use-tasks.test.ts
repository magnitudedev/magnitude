import { describe, expect, test } from 'bun:test'
import type { AgentStatusState } from '@magnitudedev/agent'
import type { TaskListItem, WorkerExecutionSnapshot } from '../components/chat/types'
import { deriveWorkerExecutionSnapshot } from './use-tasks'

function makeTask(overrides: Partial<TaskListItem> = {}): TaskListItem {
  return {
    taskId: 'task-1',
    title: 'Task',
    type: 'implement',
    status: 'pending',
    depth: 0,
    parentId: null,
    createdAt: 1_000,
    updatedAt: 1_000,
    completedAt: null,
    assignee: { kind: 'worker', agentId: 'builder-1', workerType: 'builder' },
    workerForkId: 'fork-1',
    workerExecution: null,
    ...overrides,
  }
}

function makeAgentStatus(status: 'working' | 'idle'): AgentStatusState {
  const agent = {
    agentId: 'builder-1',
    forkId: 'fork-1',
    parentForkId: null,
    name: 'builder',
    role: 'builder',
    context: '',
    mode: 'spawn' as const,
    taskId: 'task-1',
    message: null,
    status,
  }
  return {
    agents: new Map([['builder-1', agent]]),
    agentByForkId: new Map([['fork-1', 'builder-1']]),
  }
}

const idleDisplay: WorkerExecutionSnapshot = {
  state: 'idle',
  activeSince: null,
  accumulatedActiveMs: 83_000,
  completedAt: 83_000,
  resumeCount: 0,
}

describe('deriveWorkerExecutionSnapshot', () => {
  test('task state alone does not force worker state to working', () => {
    const task = makeTask({ status: 'working' })
    const result = deriveWorkerExecutionSnapshot({
      task,
      fromDisplay: idleDisplay,
      agentStatusState: null,
    })
    expect(result?.state).toBe('idle')
  })

  test('completed task + idle worker remains idle', () => {
    const task = makeTask({ status: 'completed', completedAt: 20_000 })
    const result = deriveWorkerExecutionSnapshot({
      task,
      fromDisplay: idleDisplay,
      agentStatusState: makeAgentStatus('idle'),
    })
    expect(result?.state).toBe('idle')
  })

  test('non-completed task with lifecycle-driven idle remains idle', () => {
    const task = makeTask({ status: 'pending' })
    const result = deriveWorkerExecutionSnapshot({
      task,
      fromDisplay: { ...idleDisplay, state: 'working' },
      agentStatusState: makeAgentStatus('idle'),
    })
    expect(result?.state).toBe('idle')
  })

  test('maps lifecycle working to working', () => {
    const task = makeTask()
    const fromDisplay: WorkerExecutionSnapshot = {
      state: 'idle',
      activeSince: 10_000,
      accumulatedActiveMs: 1_000,
      completedAt: null,
      resumeCount: 1,
    }

    expect(deriveWorkerExecutionSnapshot({
      task,
      fromDisplay,
      agentStatusState: makeAgentStatus('working'),
    })?.state).toBe('working')

    expect(deriveWorkerExecutionSnapshot({
      task,
      fromDisplay,
      agentStatusState: makeAgentStatus('working'),
    })?.state).toBe('working')
  })
})
