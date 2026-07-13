import { forkIdToKey, type DisplayTimelinePresentationMode, type DisplayViewShape } from '@magnitudedev/sdk'

export const INITIAL_ROOT_PAGE_SIZE = 50
export const INCREMENTAL_ROOT_PAGE_SIZE = 50
export const WORKER_TIMELINE_LIMIT = 200

export const EMPTY_DISPLAY_VIEW_SHAPE: DisplayViewShape = { timelines: {} }

export const timelineTail = (
  limit: number,
  presentation: DisplayTimelinePresentationMode = 'default',
): DisplayViewShape['timelines'][string] => ({
  kind: 'tail',
  limit,
  live: true,
  presentation,
})

export const displayShapeFor = (
  rootLimit: number,
  requestedWorkerForkIds: readonly string[],
  presentation: DisplayTimelinePresentationMode = 'default',
): DisplayViewShape => {
  const timelines: Record<string, DisplayViewShape['timelines'][string]> = {
    root: timelineTail(rootLimit, presentation),
  }

  for (const forkId of requestedWorkerForkIds) {
    timelines[forkIdToKey(forkId)] = timelineTail(WORKER_TIMELINE_LIMIT, presentation)
  }

  return { timelines }
}

/**
 * Quantize a raw entry need to a page multiple. Declarations are a step
 * function of the viewport anchor: the declared limit only changes when a
 * page boundary is crossed, which is the hysteresis that keeps scroll
 * wiggle from producing shape-request churn.
 */
export const ceilToPageMultiple = (value: number): number =>
  Math.max(
    INCREMENTAL_ROOT_PAGE_SIZE,
    Math.ceil(value / INCREMENTAL_ROOT_PAGE_SIZE) * INCREMENTAL_ROOT_PAGE_SIZE,
  )
