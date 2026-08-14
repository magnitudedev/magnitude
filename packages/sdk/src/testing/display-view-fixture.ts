import type {
  DisplayMessage,
  DisplayRootStatus,
  DisplayTimelineEntry,
  DisplayViewShape,
  DisplayViewSnapshot,
} from "@magnitudedev/acn-protocol"

export interface DisplayViewSnapshotFixtureOptions {
  readonly shape: DisplayViewShape
  readonly session: DisplayViewSnapshot["state"]["session"]
  readonly status?: DisplayRootStatus
  readonly messages?: readonly DisplayMessage[]
  readonly messageOrder?: DisplayViewSnapshot["state"]["timelines"]["root"]["messages"]["order"]
  readonly entries?: readonly DisplayTimelineEntry[]
  readonly mode?: DisplayViewSnapshot["state"]["timelines"]["root"]["mode"]
  readonly streamingMessageId?: DisplayViewSnapshot["state"]["timelines"]["root"]["streamingMessageId"]
}

export function makeDisplayViewSnapshotFixture(
  options: DisplayViewSnapshotFixtureOptions,
): DisplayViewSnapshot {
  const messages = options.messages ?? []
  const messageOrder = options.messageOrder ?? messages.map((message) => message.id)
  const mode = options.mode ?? "idle"
  return {
    shape: options.shape,
    state: {
      session: options.session,
      timelines: {
        root: {
          mode,
          messages: {
            byId: Object.fromEntries(messages.map((message) => [message.id, message])),
            order: [...messageOrder],
          },
          streamingMessageId: options.streamingMessageId === undefined
            ? (mode === "streaming" ? messageOrder.at(-1) ?? null : null)
            : options.streamingMessageId,
          window: {
            start: 0,
            end: messageOrder.length,
            totalCount: messageOrder.length,
            hasMoreBefore: false,
            hasMoreAfter: false,
          },
          presentation: {
            mode: "default",
            entries: [...(options.entries ?? [])],
            statusSlot: { kind: "none" },
          },
        },
      },
      actors: {
        root: {
          kind: "root",
          name: "Magnitude",
          role: "leader",
          parentActorKey: null,
          taskId: null,
          context: { tokenEstimate: 0, isCompacting: false },
          status: options.status ?? { _tag: "Idle" },
        },
      },
      agents: {},
      tasks: {
        byId: {},
        order: [],
        summary: { totalCount: 0, completedCount: 0, incompleteCount: 0 },
      },
    },
  }
}
