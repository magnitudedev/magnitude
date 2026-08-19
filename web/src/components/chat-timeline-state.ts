import type { TimelineStatus } from "@magnitudedev/client-common"

export interface ChatTimelinePlaceholderState {
  readonly isSessionLoading: boolean
  readonly isEmpty: boolean
}

/** Visible timeline content always outranks connection and empty placeholders. */
export function chatTimelinePlaceholderState(
  selectedSessionId: string | null,
  timelineStatus: TimelineStatus,
  entryCount: number,
): ChatTimelinePlaceholderState {
  if (entryCount > 0) {
    return { isSessionLoading: false, isEmpty: false }
  }
  return {
    isSessionLoading:
      selectedSessionId !== null && timelineStatus._tag === "pending",
    isEmpty:
      selectedSessionId === null || timelineStatus._tag === "empty",
  }
}
