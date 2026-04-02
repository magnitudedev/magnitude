import { describe, expect, test } from 'bun:test'
import type { TaskListItem } from '../components/chat/types'
import {
  computeInheritedVisualStatusMap,
  getOwnVisualStatus,
} from './task-visual-status'

const makeTask = (overrides: Partial<TaskListItem> = {}): TaskListItem => ({
  taskId: 't-1',
  title: 'Task',
  type: 'implement',
  status: 'pending',
  depth: 0,
  parentId: null,
  createdAt: 1_000,
  updatedAt: 2_000,
  completedAt: null,
  assignee: { kind: 'lead' },
  workerForkId: null,
  ...overrides,
})

describe('getOwnVisualStatus', () => {
  test('returns completed for completed tasks', () => {
    expect(getOwnVisualStatus(makeTask({ status: 'completed', completedAt: 10_000 }))).toBe('completed')
  })

  test('returns completed for archived tasks', () => {
    expect(getOwnVisualStatus(makeTask({ status: 'archived' }))).toBe('completed')
  })

  test('returns working for working tasks', () => {
    expect(getOwnVisualStatus(makeTask({ status: 'working' }))).toBe('working')
  })

  test('returns assigned for worker with fork id', () => {
    expect(
      getOwnVisualStatus(
        makeTask({
          status: 'pending',
          assignee: { kind: 'worker', agentId: 'builder-1', workerType: 'builder' },
          workerForkId: 'fork-1',
        }),
      ),
    ).toBe('assigned')
  })

  test('returns pending when worker is present but fork id is missing', () => {
    expect(
      getOwnVisualStatus(
        makeTask({
          status: 'pending',
          assignee: { kind: 'worker', agentId: 'builder-1', workerType: 'builder' },
          workerForkId: null,
        }),
      ),
    ).toBe('pending')
  })

  test('returns pending for unassigned lead task', () => {
    expect(getOwnVisualStatus(makeTask({ status: 'pending', assignee: { kind: 'lead' } }))).toBe('pending')
  })
})

describe('computeInheritedVisualStatusMap', () => {
  test('single task with no children keeps own status', () => {
    const tasks = [makeTask({ taskId: 'root', status: 'working' })]
    const result = computeInheritedVisualStatusMap(tasks)
    expect(result.get('root')).toBe('working')
  })

  test('parent with working child becomes working', () => {
    const tasks = [
      makeTask({ taskId: 'parent', status: 'pending' }),
      makeTask({ taskId: 'child', depth: 1, parentId: 'parent', status: 'working' }),
    ]
    const result = computeInheritedVisualStatusMap(tasks)
    expect(result.get('parent')).toBe('working')
  })

  test('parent with assigned child becomes assigned', () => {
    const tasks = [
      makeTask({ taskId: 'parent', status: 'pending' }),
      makeTask({
        taskId: 'child',
        depth: 1,
        parentId: 'parent',
        status: 'pending',
        assignee: { kind: 'worker', agentId: 'builder-1', workerType: 'builder' },
        workerForkId: 'fork-1',
      }),
    ]
    const result = computeInheritedVisualStatusMap(tasks)
    expect(result.get('parent')).toBe('assigned')
  })

  test('parent with assigned and working children becomes working', () => {
    const tasks = [
      makeTask({ taskId: 'parent', status: 'pending' }),
      makeTask({
        taskId: 'assigned-child',
        depth: 1,
        parentId: 'parent',
        status: 'pending',
        assignee: { kind: 'worker', agentId: 'builder-1', workerType: 'builder' },
        workerForkId: 'fork-1',
      }),
      makeTask({ taskId: 'working-child', depth: 1, parentId: 'parent', status: 'working' }),
    ]
    const result = computeInheritedVisualStatusMap(tasks)
    expect(result.get('parent')).toBe('working')
  })

  test('completed parent with working child stays completed', () => {
    const tasks = [
      makeTask({ taskId: 'parent', status: 'completed', completedAt: 10_000 }),
      makeTask({ taskId: 'child', depth: 1, parentId: 'parent', status: 'working' }),
    ]
    const result = computeInheritedVisualStatusMap(tasks)
    expect(result.get('parent')).toBe('completed')
  })

  test('completed child does not propagate upward', () => {
    const tasks = [
      makeTask({ taskId: 'parent', status: 'pending' }),
      makeTask({ taskId: 'child', depth: 1, parentId: 'parent', status: 'completed', completedAt: 10_000 }),
    ]
    const result = computeInheritedVisualStatusMap(tasks)
    expect(result.get('parent')).toBe('pending')
  })

  test('archived summary row is excluded from inheritance', () => {
    const tasks = [
      makeTask({ taskId: 'parent', status: 'pending' }),
      makeTask({ taskId: '__archived__parent', depth: 1, parentId: 'parent', status: 'archived' }),
      makeTask({ taskId: 'child', depth: 2, parentId: '__archived__parent', status: 'working' }),
    ]
    const result = computeInheritedVisualStatusMap(tasks)
    expect(result.get('__archived__parent')).toBe('completed')
    expect(result.get('parent')).toBe('pending')
  })

  test('child with missing parent keeps own status and does not throw', () => {
    const tasks = [makeTask({ taskId: 'orphan', parentId: 'missing', depth: 1, status: 'working' })]
    const result = computeInheritedVisualStatusMap(tasks)
    expect(result.get('orphan')).toBe('working')
  })

  test('grandchild working propagates to grandparent', () => {
    const tasks = [
      makeTask({ taskId: 'grandparent', status: 'pending' }),
      makeTask({ taskId: 'parent', depth: 1, parentId: 'grandparent', status: 'pending' }),
      makeTask({ taskId: 'child', depth: 2, parentId: 'parent', status: 'working' }),
    ]
    const result = computeInheritedVisualStatusMap(tasks)
    expect(result.get('parent')).toBe('working')
    expect(result.get('grandparent')).toBe('working')
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
