import { describe, expect, test } from 'vitest'
import type { TaskWorkerSnapshot } from '@magnitudedev/agent'
import { deriveTaskDisplayRow, deriveWorkerSlotDisplay } from '../components/chat/task-list/index'

function makeSnapshot(overrides: Partial<TaskWorkerSnapshot> = {}): TaskWorkerSnapshot {
  return {
    taskId: 'task-1',
    title: 'Task',
  
    parentId: null,
    depth: 0,
    status: 'pending',
    updatedAt: 1_000,
    assignee: {
      kind: 'worker',
      agentId: 'builder-1',
      forkId: 'fork-1',
      role: 'builder',
    },
    workerState: {
      status: 'idle',
      forkId: 'fork-1',
      accumulatedMs: 83_000,
      completedAt: 83_000,
      resumeCount: 0,
    },
    ...overrides,
  }
}

describe('deriveWorkerSlotDisplay', () => {
  test('returns null for unassigned task', () => {
    const snapshot = makeSnapshot({
      assignee: { kind: 'none' },
      workerState: { status: 'unassigned' },
    })
    expect(deriveWorkerSlotDisplay(snapshot)).toBeNull()
  })

  test('returns null for spawning without role', () => {
    const snapshot = makeSnapshot({
      assignee: { kind: 'none' },
      workerState: { status: 'spawning', toolCallId: 'tool-1', role: null },
    })
    expect(deriveWorkerSlotDisplay(snapshot)).toBeNull()
  })

  test('returns spawning slot when role is present', () => {
    const snapshot = makeSnapshot({
      assignee: { kind: 'none' },
      workerState: { status: 'spawning', toolCallId: 'tool-1', role: 'builder' },
    })
    const slot = deriveWorkerSlotDisplay(snapshot)
    expect(slot).toMatchObject({
      kind: 'worker',
      variant: 'spawning',
      label: '[builder]',
      icon: '+',
      ghostEligible: false,
    })
  })

  test('returns working slot with continuity key', () => {
    const snapshot = makeSnapshot({
      workerState: {
        status: 'working',
        forkId: 'fork-1',
        activeSince: 1000,
        accumulatedMs: 5000,
        resumeCount: 0,
      },
    })
    const slot = deriveWorkerSlotDisplay(snapshot)
    expect(slot).toMatchObject({
      kind: 'worker',
      variant: 'working',
      label: '[builder] builder-1',
      tone: 'active',
      continuityKey: 'fork-1',
      ghostEligible: true,
    })
  })

  test('returns idle slot', () => {
    const snapshot = makeSnapshot({
      workerState: {
        status: 'idle',
        forkId: 'fork-1',
        accumulatedMs: 10000,
        completedAt: 10000,
        resumeCount: 0,
      },
    })
    const slot = deriveWorkerSlotDisplay(snapshot)
    expect(slot).toMatchObject({
      kind: 'worker',
      variant: 'idle',
      tone: 'muted',
      ghostEligible: true,
    })
  })

  test('returns killing slot', () => {
    const snapshot = makeSnapshot({
      workerState: {
        status: 'killing',
        forkId: 'fork-1',
        toolCallId: 'kill-1',
      },
    })
    const slot = deriveWorkerSlotDisplay(snapshot)
    expect(slot).toMatchObject({
      kind: 'worker',
      variant: 'killing',
      icon: '✕',
      tone: 'danger',
      ghostEligible: true,
    })
  })

  test('returns user slot for user assignee', () => {
    const snapshot = makeSnapshot({
      assignee: { kind: 'user' },
      workerState: { status: 'unassigned' },
    })
    const slot = deriveWorkerSlotDisplay(snapshot)
    expect(slot).toMatchObject({
      kind: 'user',
      label: 'user',
      tone: 'warning',
    })
  })
})

describe('deriveTaskDisplayRow', () => {
  test('creates display row with worker slot', () => {
    const snapshot = makeSnapshot()
    const row = deriveTaskDisplayRow(snapshot)

    expect(row.kind).toBe('task')
    expect(row.taskId).toBe('task-1')
    expect(row.title).toBe('Task')
    expect(row.assignee.kind).toBe('worker')
  })

  test('creates display row with none slot for unassigned', () => {
    const snapshot = makeSnapshot({
      assignee: { kind: 'none' },
      workerState: { status: 'unassigned' },
    })
    const row = deriveTaskDisplayRow(snapshot)

    expect(row.assignee).toEqual({ kind: 'none' })
  })
})
