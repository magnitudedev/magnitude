import { describe, expect, test } from 'bun:test'
import { flattenTaskTree, type TaskGraphState } from '../utils/task-tree'

const makeState = (tasks: TaskGraphState['tasks']): TaskGraphState => ({
  tasks,
  rootTaskIds: [],
})

describe('flattenTaskTree', () => {
  test('sorts roots by createdAt then task id', () => {
    const state = makeState(new Map([
      ['b', { id: 'b', title: 'B', taskType: 'implement', parentId: null, childIds: [], assignee: null, worker: null, status: 'pending', createdAt: 2, updatedAt: 2, completedAt: null }],
      ['a', { id: 'a', title: 'A', taskType: 'implement', parentId: null, childIds: [], assignee: null, worker: null, status: 'pending', createdAt: 2, updatedAt: 2, completedAt: null }],
    ]) as any)

    expect(flattenTaskTree(state).map(t => t.taskId)).toEqual(['a', 'b'])
  })

  test('applies depth for nested tasks', () => {
    const state = makeState(new Map([
      ['root', { id: 'root', title: 'Root', taskType: 'feature', parentId: null, childIds: ['child'], assignee: null, worker: null, status: 'pending', createdAt: 1, updatedAt: 1, completedAt: null }],
      ['child', { id: 'child', title: 'Child', taskType: 'implement', parentId: 'root', childIds: [], assignee: null, worker: null, status: 'pending', createdAt: 2, updatedAt: 2, completedAt: null }],
    ]) as any)

    const rows = flattenTaskTree(state)
    expect(rows.find(r => r.taskId === 'root')?.depth).toBe(0)
    expect(rows.find(r => r.taskId === 'child')?.depth).toBe(1)
  })
})
