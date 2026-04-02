import { describe, expect, test } from 'bun:test'
import type { TaskListItem } from '../components/chat/types'
import { buildRootSummaries, findOwningRootIndex } from './task-tree'

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

describe('buildRootSummaries', () => {
  test('single root with children has correct range and progress', () => {
    const tasks = [
      makeTask({ taskId: 'root', status: 'pending' }),
      makeTask({ taskId: 'child-1', depth: 1, parentId: 'root', status: 'completed', completedAt: 10_000 }),
      makeTask({ taskId: 'child-2', depth: 1, parentId: 'root', status: 'pending' }),
    ]

    const summaries = buildRootSummaries(tasks)
    expect(summaries).toHaveLength(1)
    expect(summaries[0]).toEqual({
      task: tasks[0],
      startIndex: 0,
      endIndex: 3,
      completed: 1,
      active: 0,
      total: 3,
    })
  })

  test('multiple roots have exclusive endIndex boundaries', () => {
    const tasks = [
      makeTask({ taskId: 'root-a' }),
      makeTask({ taskId: 'a-child', depth: 1, parentId: 'root-a' }),
      makeTask({ taskId: 'root-b' }),
      makeTask({ taskId: 'b-child', depth: 1, parentId: 'root-b' }),
      makeTask({ taskId: 'root-c' }),
    ]

    const summaries = buildRootSummaries(tasks)
    expect(summaries).toHaveLength(3)
    expect(summaries[0]?.startIndex).toBe(0)
    expect(summaries[0]?.endIndex).toBe(2)
    expect(summaries[1]?.startIndex).toBe(2)
    expect(summaries[1]?.endIndex).toBe(4)
    expect(summaries[2]?.startIndex).toBe(4)
    expect(summaries[2]?.endIndex).toBe(5)
  })

  test('archived summary rows are excluded from roots and progress', () => {
    const tasks = [
      makeTask({ taskId: '__archived____root', title: '2 archived tasks', type: 'archived', status: 'archived', depth: 0 }),
      makeTask({ taskId: 'archived-child', status: 'archived', depth: 1, parentId: '__archived____root' }),
      makeTask({ taskId: 'root-a', status: 'pending', depth: 0 }),
      makeTask({ taskId: '__archived__root-a', title: '1 archived task', type: 'archived', status: 'archived', depth: 1, parentId: 'root-a' }),
      makeTask({ taskId: 'child-live', status: 'completed', completedAt: 10_000, depth: 1, parentId: 'root-a' }),
    ]

    const summaries = buildRootSummaries(tasks)
    expect(summaries).toHaveLength(1)
    expect(summaries[0]?.task.taskId).toBe('root-a')
    expect(summaries[0]?.completed).toBe(1)
    expect(summaries[0]?.total).toBe(2)
  })

  test('empty list returns empty array', () => {
    expect(buildRootSummaries([])).toEqual([])
  })
})

describe('findOwningRootIndex', () => {
  test('child at depth 1 finds parent root', () => {
    const tasks = [
      makeTask({ taskId: 'root', depth: 0 }),
      makeTask({ taskId: 'child', depth: 1, parentId: 'root' }),
    ]
    expect(findOwningRootIndex(tasks, 1)).toBe(0)
  })

  test('deep child finds correct root', () => {
    const tasks = [
      makeTask({ taskId: 'root-a', depth: 0 }),
      makeTask({ taskId: 'a-child', depth: 1, parentId: 'root-a' }),
      makeTask({ taskId: 'root-b', depth: 0 }),
      makeTask({ taskId: 'b-child', depth: 1, parentId: 'root-b' }),
      makeTask({ taskId: 'b-grandchild', depth: 2, parentId: 'b-child' }),
    ]
    expect(findOwningRootIndex(tasks, 4)).toBe(2)
  })

  test('archived summary root row is skipped while finding real root', () => {
    const tasks = [
      makeTask({ taskId: 'root-a', depth: 0 }),
      makeTask({ taskId: '__archived__root-a', depth: 0, status: 'archived', type: 'archived' }),
      makeTask({ taskId: 'archived-child', depth: 1, parentId: '__archived__root-a', status: 'archived' }),
    ]
    expect(findOwningRootIndex(tasks, 2)).toBe(0)
  })

  test('index 0 as root returns 0', () => {
    const tasks = [makeTask({ taskId: 'root', depth: 0 })]
    expect(findOwningRootIndex(tasks, 0)).toBe(0)
  })

  test('no root found returns null', () => {
    const tasks = [makeTask({ taskId: 'child', depth: 1, parentId: 'missing-root' })]
    expect(findOwningRootIndex(tasks, 0)).toBeNull()
  })
})
