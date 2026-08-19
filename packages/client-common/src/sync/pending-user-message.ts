import type { DisplaySpeculator, SpeculativeDisplayHandle } from "./display-view-store"
import { appendMessageToTimeline, emptyTimeline } from "./display-view-store"
import { INITIAL_ROOT_PAGE_SIZE, timelineTail } from "./display-view-shape"
import type { DisplayAttachment } from "@magnitudedev/sdk"

export interface PendingUserMessageProjection {
  readonly messageId: string
  readonly content: string
  readonly taskMode: boolean
  readonly activeSessionId: string | null
  readonly draftSessionId: string
  readonly cwd: string
  readonly attachments?: readonly DisplayAttachment[]
}

/**
 * Present one exact pending SendMessage/CreateSession intent over the current
 * authoritative display. The server-authored message with the same identity
 * acknowledges and replaces this projection.
 */
export function presentPendingUserMessage(
  speculator: DisplaySpeculator,
  pending: PendingUserMessageProjection,
): SpeculativeDisplayHandle {
  return speculator.mutate(
    {
      owner: `send:${pending.messageId}`,
      label: "send-message",
      acknowledgedBy: (accepted) =>
        accepted.state.timelines.root?.messages.byId[pending.messageId] !== undefined,
    },
    (draft) => {
      if (!pending.activeSessionId && !draft.state.session.sessionId) {
        draft.state.session = {
          sessionId: pending.draftSessionId,
          title: null,
          cwd: pending.cwd,
        }
      }

      draft.shape.timelines.root ??= timelineTail(INITIAL_ROOT_PAGE_SIZE)
      draft.state.timelines.root ??= emptyTimeline()
      const messageType = draft.state.timelines.root.mode === "streaming"
        ? "queued_user_message"
        : "user_message"
      draft.state.timelines.root = appendMessageToTimeline(draft.state.timelines.root, {
        id: pending.messageId,
        type: messageType,
        content: pending.content,
        timestamp: Date.now(),
        taskMode: pending.taskMode,
        attachments: [...(pending.attachments ?? [])],
      })
    },
  )
}
