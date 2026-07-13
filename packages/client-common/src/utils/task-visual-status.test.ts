import { describe, expect, test } from 'vitest'
import type { TaskDisplayRow } from '@magnitudedev/sdk'
import { Option } from 'effect'
import {
  computeInheritedVisualStatusMap,
  getOwnVisualStatus,
} from './task-visual-status'

type TaskRow = TaskDisplayRow

const workerAssignee: TaskDisplayRow['assignee'] = {
  kind: 'actor',
  actorKey: 'fork-1',
  taskState: 'assigned',
  timer: Option.none(),
}

const makeTask = (overrides: Partial<TaskRow> = {}): TaskRow => ({
  rowId: 'task:t-1',
  kind: 'task',
  taskId: 't-1',
  title: 'Task',

  status: 'pending',
  depth: 0,
  parentId: Option.none(),
  updatedAt: 2_000,
  assignee: { kind: 'none' },
  ...overrides,
})

describe('getOwnVisualStatus', () => {
  test('returns completed for completed tasks', () => {
    expect(getOwnVisualStatus(makeTask({ status: 'completed' }))).toBe('completed')
  })

  test('returns pending for working tasks', () => {
    expect(getOwnVisualStatus(makeTask({ status: 'pending' }))).toBe('pending')
  })

  test('returns pending for task with worker slot', () => {
    expect(
      getOwnVisualStatus(
        makeTask({
          status: 'pending',
          assignee: workerAssignee,
        }),
      ),
    ).toBe('pending')
  })

  test('returns pending for unassigned task', () => {
    expect(getOwnVisualStatus(makeTask({ status: 'pending', assignee: { kind: 'none' } }))).toBe('pending')
  })
})

describe('computeInheritedVisualStatusMap', () => {
  test('single task with no children keeps own status', () => {
    const tasks = [makeTask({ taskId: 'root', status: 'pending' })]
    const result = computeInheritedVisualStatusMap(tasks)
    expect(result.get('root')).toBe('pending')
  })

  test('parent with working child remains pending', () => {
    const tasks = [
      makeTask({ taskId: 'parent', status: 'pending' }),
      makeTask({ taskId: 'child', depth: 1, parentId: Option.some('parent'), status: 'pending' }),
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
        parentId: Option.some('parent'),
        status: 'pending',
        assignee: workerAssignee,
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
        parentId: Option.some('parent'),
        status: 'pending',
        assignee: workerAssignee,
      }),
      makeTask({ taskId: 'working-child', depth: 1, parentId: Option.some('parent'), status: 'pending' }),
    ]
    const result = computeInheritedVisualStatusMap(tasks)
    expect(result.get('parent')).toBe('pending')
  })

  test('completed parent with working child stays completed', () => {
    const tasks = [
      makeTask({ taskId: 'parent', status: 'completed' }),
      makeTask({ taskId: 'child', depth: 1, parentId: Option.some('parent'), status: 'pending' }),
    ]
    const result = computeInheritedVisualStatusMap(tasks)
    expect(result.get('parent')).toBe('completed')
  })

  test('completed child does not propagate upward', () => {
    const tasks = [
      makeTask({ taskId: 'parent', status: 'pending' }),
      makeTask({ taskId: 'child', depth: 1, parentId: Option.some('parent'), status: 'completed' }),
    ]
    const result = computeInheritedVisualStatusMap(tasks)
    expect(result.get('parent')).toBe('pending')
  })

  test('child with missing parent keeps own status and does not throw', () => {
    const tasks = [makeTask({ taskId: 'orphan', parentId: Option.some('missing'), depth: 1, status: 'pending' })]
    const result = computeInheritedVisualStatusMap(tasks)
    expect(result.get('orphan')).toBe('pending')
  })

  test('grandchild working does not change grandparent from pending', () => {
    const tasks = [
      makeTask({ taskId: 'grandparent', status: 'pending' }),
      makeTask({ taskId: 'parent', depth: 1, parentId: Option.some('grandparent'), status: 'pending' }),
      makeTask({ taskId: 'child', depth: 2, parentId: Option.some('parent'), status: 'pending' }),
    ]
    const result = computeInheritedVisualStatusMap(tasks)
    expect(result.get('parent')).toBe('pending')
    expect(result.get('grandparent')).toBe('pending')
  })

  test('all pending tree stays pending', () => {
    const tasks = [
      makeTask({ taskId: 'root', status: 'pending' }),
      makeTask({ taskId: 'child-a', depth: 1, parentId: Option.some('root'), status: 'pending' }),
      makeTask({ taskId: 'child-b', depth: 1, parentId: Option.some('root'), status: 'pending' }),
    ]
    const result = computeInheritedVisualStatusMap(tasks)
    expect(result.get('root')).toBe('pending')
    expect(result.get('child-a')).toBe('pending')
    expect(result.get('child-b')).toBe('pending')
  })
})
