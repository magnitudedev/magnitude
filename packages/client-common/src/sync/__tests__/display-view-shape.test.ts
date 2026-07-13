import { describe, expect, it } from 'vitest'
import {
  INITIAL_ROOT_PAGE_SIZE,
  INCREMENTAL_ROOT_PAGE_SIZE,
  WORKER_TIMELINE_LIMIT,
  ceilToPageMultiple,
  displayShapeFor,
  timelineTail,
} from '../display-view-shape'

describe('display view shape helpers', () => {
  it('builds a root-only default shape', () => {
    expect(displayShapeFor(INITIAL_ROOT_PAGE_SIZE, [])).toEqual({
      timelines: {
        root: timelineTail(INITIAL_ROOT_PAGE_SIZE),
      },
    })
  })

  it('adds expanded worker timelines without changing the root limit', () => {
    expect(displayShapeFor(600, ['worker-a', 'worker/b'])).toEqual({
      timelines: {
        root: timelineTail(600),
        'worker-a': timelineTail(WORKER_TIMELINE_LIMIT),
        'worker/b': timelineTail(WORKER_TIMELINE_LIMIT),
      },
    })
  })

  it('quantizes declarations up to page multiples with a one-page floor', () => {
    expect(ceilToPageMultiple(0)).toBe(INCREMENTAL_ROOT_PAGE_SIZE)
    expect(ceilToPageMultiple(1)).toBe(INCREMENTAL_ROOT_PAGE_SIZE)
    expect(ceilToPageMultiple(INCREMENTAL_ROOT_PAGE_SIZE)).toBe(INCREMENTAL_ROOT_PAGE_SIZE)
    expect(ceilToPageMultiple(INCREMENTAL_ROOT_PAGE_SIZE + 1)).toBe(INCREMENTAL_ROOT_PAGE_SIZE * 2)
    expect(ceilToPageMultiple(110)).toBe(150)
  })

  it('is idempotent — a quantized value maps to itself', () => {
    for (const value of [50, 100, 150, 500]) {
      expect(ceilToPageMultiple(ceilToPageMultiple(value))).toBe(ceilToPageMultiple(value))
    }
  })
})
