import type { DisplayState } from "@magnitudedev/sdk"

export const EMPTY_DISPLAY_STATE: DisplayState = {
  session: { sessionId: "", title: null, cwd: "" },
  timelines: {},
  actors: {},
  agents: {},
  tasks: { byId: {}, order: [], summary: { totalCount: 0, completedCount: 0, incompleteCount: 0 } },
}
