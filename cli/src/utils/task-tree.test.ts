import { describe, expect, test } from 'vitest'
import type { TaskDisplayRow } from '../components/chat/task-list/index'
import { buildRootSummaries, findOwningRootIndex } from './task-tree'

type TaskRow = TaskDisplayRow

const makeTask = (overrides: Partial<TaskRow> = {}): TaskRow => ({
  rowId: 'task:t-1',
  kind: 'task',
  taskId: 't-1',
  title: 'Task',
  taskType: 'implement',
  status: 'pending',
  depth: 0,
  parentId: null,
  updatedAt: 2_000,
  workerSlot: null,
  ...overrides,
})

describe('buildRootSummaries', () => {
  test('single root with children has correct range and progress', () => {
    const tasks = [
      makeTask({ taskId: 'root', status: 'pending' }),
      makeTask({ taskId: 'child-1', depth: 1, parentId: 'root', status: 'completed',  }),
      makeTask({ taskId: 'child-2', depth: 1, parentId: 'root', status: 'pending' }),
    ]

    const summaries = buildRootSummaries(tasks)
    expect(summaries).toHaveLength(1)
    expect(summaries[0]).toEqual({
      task: tasks[0],
      startIndex: 0,
      endIndex: 3,
      completed: 1,
      active: 2,
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

  test('index 0 as root returns 0', () => {
    const tasks = [makeTask({ taskId: 'root', depth: 0 })]
    expect(findOwningRootIndex(tasks, 0)).toBe(0)
  })

  test('no root found returns null', () => {
    const tasks = [makeTask({ taskId: 'child', depth: 1, parentId: 'missing-root' })]
    expect(findOwningRootIndex(tasks, 0)).toBeNull()
  })
})
