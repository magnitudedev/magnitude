import { describe, expect, test } from 'vitest'
import type { TaskListItem } from '../components/chat/task-list/index'
import {
  computeInheritedVisualStatusMap,
  getOwnVisualStatus,
} from './task-visual-status'

type TaskRow = Extract<TaskListItem, { kind: 'task' }>

const makeTask = (overrides: Partial<TaskRow> = {}): TaskRow => ({
  rowId: 'task:t-1',
  kind: 'task',
  taskId: 't-1',
  title: 'Task',

  status: 'pending',
  depth: 0,
  parentId: null,
  updatedAt: 2_000,
  workerSlot: null,
  ...overrides,
})

describe('getOwnVisualStatus', () => {
  test('returns completed for completed tasks', () => {
    expect(getOwnVisualStatus(makeTask({ status: 'completed' }))).toBe('completed')
  })

  test('returns pending for working tasks', () => {
    expect(getOwnVisualStatus(makeTask({ status: 'working' }))).toBe('pending')
  })

  test('returns pending for task with worker slot', () => {
    expect(
      getOwnVisualStatus(
        makeTask({
          status: 'pending',
          workerSlot: {
            kind: 'worker',
            variant: 'idle',
            label: '[builder] builder-1',
            icon: '●',
            tone: 'muted',
            interactiveForkId: 'fork-1',
            timer: { startedAt: 0, resumedAt: null },
            resumed: false,
            continuityKey: 'fork-1',
            ghostEligible: true,
          },
        }),
      ),
    ).toBe('pending')
  })

  test('returns pending for unassigned task', () => {
    expect(getOwnVisualStatus(makeTask({ status: 'pending', workerSlot: null }))).toBe('pending')
  })
})

describe('computeInheritedVisualStatusMap', () => {
  test('single task with no children keeps own status', () => {
    const tasks = [makeTask({ taskId: 'root', status: 'working' })]
    const result = computeInheritedVisualStatusMap(tasks)
    expect(result.get('root')).toBe('pending')
  })

  test('parent with working child remains pending', () => {
    const tasks = [
      makeTask({ taskId: 'parent', status: 'pending' }),
      makeTask({ taskId: 'child', depth: 1, parentId: 'parent', status: 'working' }),
    ]
    const result = computeInheritedVisualStatusMap(tasks)
    expect(result.get('parent')).toBe('pending')
  })

  test('parent with assigned child remains pending', () => {
    const tasks = [
      makeTask({ taskId: 'parent', status: 'pending' }),
      makeTask({
        taskId: 'child',
        depth: 1,
        parentId: 'parent',
        status: 'pending',
        workerSlot: {
          kind: 'worker',
          variant: 'idle',
          label: '[builder] builder-1',
          icon: '●',
          tone: 'muted',
          interactiveForkId: 'fork-1',
          timer: { startedAt: 0, resumedAt: null },
          resumed: false,
          continuityKey: 'fork-1',
          ghostEligible: true,
        },
      }),
    ]
    const result = computeInheritedVisualStatusMap(tasks)
    expect(result.get('parent')).toBe('pending')
  })

  test('parent with assigned and working children remains pending', () => {
    const tasks = [
      makeTask({ taskId: 'parent', status: 'pending' }),
      makeTask({
        taskId: 'assigned-child',
        depth: 1,
        parentId: 'parent',
        status: 'pending',
        workerSlot: {
          kind: 'worker',
          variant: 'idle',
          label: '[builder] builder-1',
          icon: '●',
          tone: 'muted',
          interactiveForkId: 'fork-1',
          timer: { startedAt: 0, resumedAt: null },
          resumed: false,
          continuityKey: 'fork-1',
          ghostEligible: true,
        },
      }),
      makeTask({ taskId: 'working-child', depth: 1, parentId: 'parent', status: 'working' }),
    ]
    const result = computeInheritedVisualStatusMap(tasks)
    expect(result.get('parent')).toBe('pending')
  })

  test('completed parent with working child stays completed', () => {
    const tasks = [
      makeTask({ taskId: 'parent', status: 'completed' }),
      makeTask({ taskId: 'child', depth: 1, parentId: 'parent', status: 'working' }),
    ]
    const result = computeInheritedVisualStatusMap(tasks)
    expect(result.get('parent')).toBe('completed')
  })

  test('completed child does not propagate upward', () => {
    const tasks = [
      makeTask({ taskId: 'parent', status: 'pending' }),
      makeTask({ taskId: 'child', depth: 1, parentId: 'parent', status: 'completed' }),
    ]
    const result = computeInheritedVisualStatusMap(tasks)
    expect(result.get('parent')).toBe('pending')
  })

  test('child with missing parent keeps own status and does not throw', () => {
    const tasks = [makeTask({ taskId: 'orphan', parentId: 'missing', depth: 1, status: 'working' })]
    const result = computeInheritedVisualStatusMap(tasks)
    expect(result.get('orphan')).toBe('pending')
  })

  test('grandchild working does not change grandparent from pending', () => {
    const tasks = [
      makeTask({ taskId: 'grandparent', status: 'pending' }),
      makeTask({ taskId: 'parent', depth: 1, parentId: 'grandparent', status: 'pending' }),
      makeTask({ taskId: 'child', depth: 2, parentId: 'parent', status: 'working' }),
    ]
    const result = computeInheritedVisualStatusMap(tasks)
    expect(result.get('parent')).toBe('pending')
    expect(result.get('grandparent')).toBe('pending')
  })

  test('all pending tree stays pending', () => {
    const tasks = [
      makeTask({ taskId: 'root', status: 'pending' }),
      makeTask({ taskId: 'child-a', depth: 1, parentId: 'root', status: 'pending' }),
      makeTask({ taskId: 'child-b', depth: 1, parentId: 'root', status: 'pending' }),
    ]
    const result = computeInheritedVisualStatusMap(tasks)
    expect(result.get('root')).toBe('pending')
    expect(result.get('child-a')).toBe('pending')
    expect(result.get('child-b')).toBe('pending')
  })
})