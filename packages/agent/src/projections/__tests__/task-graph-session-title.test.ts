import { describe, expect, it } from 'bun:test'
import type { TaskGraphState } from '../task-graph'
import { getSessionTitleFromTaskGraph } from '../task-graph'

const baseState = (): TaskGraphState => ({
  tasks: new Map(),
  rootTaskIds: [],
})

describe('getSessionTitleFromTaskGraph', () => {
  it('returns null when there are no root tasks', () => {
    expect(getSessionTitleFromTaskGraph(baseState())).toBeNull()
  })

  it('returns the first root task title', () => {
    const state: TaskGraphState = {
      tasks: new Map([
        ['t1', { id: 't1', title: 'First root', taskType: 'implement', parentId: null, childIds: [], assignee: null, worker: null, status: 'pending', createdAt: 1, updatedAt: 1, completedAt: null }],
        ['t2', { id: 't2', title: 'Second root', taskType: 'implement', parentId: null, childIds: [], assignee: null, worker: null, status: 'pending', createdAt: 2, updatedAt: 2, completedAt: null }],
      ]),
      rootTaskIds: ['t1', 't2'],
    }

    expect(getSessionTitleFromTaskGraph(state)).toBe('First root')
  })

  it('reflects selected root renames and ignores later roots', () => {
    const state: TaskGraphState = {
      tasks: new Map([
        ['t1', { id: 't1', title: 'Renamed first root', taskType: 'implement', parentId: null, childIds: [], assignee: null, worker: null, status: 'pending', createdAt: 1, updatedAt: 3, completedAt: null }],
        ['t2', { id: 't2', title: 'Renamed second root', taskType: 'implement', parentId: null, childIds: [], assignee: null, worker: null, status: 'pending', createdAt: 2, updatedAt: 4, completedAt: null }],
        ['c1', { id: 'c1', title: 'Subtask', taskType: 'implement', parentId: 't1', childIds: [], assignee: null, worker: null, status: 'pending', createdAt: 5, updatedAt: 5, completedAt: null }],
      ]),
      rootTaskIds: ['t1', 't2'],
    }

    expect(getSessionTitleFromTaskGraph(state)).toBe('Renamed first root')
  })
})
