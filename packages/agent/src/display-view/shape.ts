import type {
  DisplayTimelinePresentationMode,
  DisplayTimelineWindowShape,
  DisplayViewShape
} from '@magnitudedev/protocol'

export const DEFAULT_TIMELINE_WINDOW_LIMIT = 300

export type { DisplayTimelineWindowShape, DisplayViewShape }

export const defaultDisplayViewShape: DisplayViewShape = {
  timelines: {
    root: {
      kind: 'tail',
      limit: DEFAULT_TIMELINE_WINDOW_LIMIT,
      live: true,
      presentation: 'default',
    }
  }
}

export const timelineTail = (
  limit = DEFAULT_TIMELINE_WINDOW_LIMIT,
  live = true,
  presentation: DisplayTimelinePresentationMode = 'default',
): DisplayTimelineWindowShape => ({
  kind: 'tail',
  limit,
  live,
  presentation,
})

export const timelineRange = (
  start: number,
  limit: number,
  live = false,
  presentation: DisplayTimelinePresentationMode = 'default',
): DisplayTimelineWindowShape => ({
  kind: 'range',
  start,
  limit,
  live,
  presentation,
})
