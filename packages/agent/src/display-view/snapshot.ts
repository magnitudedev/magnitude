import { Effect } from 'effect'
import { AmbientServiceTag, Projection } from '@magnitudedev/event-core'
import {
  forkKeyToForkId,
  type DisplayMessage,
  type DisplayState,
  type DisplayTimeline,
  type DisplayTimelinePresentationMode,
  type DisplayViewShape,
  type DisplayViewSnapshot,
  type ForkKey,
} from '@magnitudedev/protocol'
import { SessionOptionsAmbient } from '../ambient/session-ambient'
import { ChatTitleProjection } from '../projections/chat-title'
import { AgentLifecycleProjection } from '../projections/agent-lifecycle'
import { TaskAssignmentProjection } from '../projections/task-assignment'
import { SessionContextProjection } from '../projections/session-context'
import { WindowProjection } from '../window'
import { CompactionProjection } from '../projections/compaction'
import { DisplayTimelineProjection } from '../display/timeline-projection'
import type { DisplayTimelineState } from '../display/types'
import { materializeDisplayTimeline } from './materializer'
import { buildDisplayTimelinePresentation } from './timeline-presentation'
import {
  materializeDisplayActors,
  materializeDisplayAgents,
  materializeDisplaySession,
  materializeDisplayTasks,
} from './semantic'
import { type DisplayTimelineWindowShape } from './shape'

const windowBounds = (
  shape: DisplayTimelineWindowShape,
  totalCount: number
): { readonly start: number; readonly end: number } => {
  switch (shape.kind) {
    case 'tail': {
      const start = Math.max(0, totalCount - shape.limit)
      return { start, end: totalCount }
    }
    case 'range': {
      const start = Math.max(0, shape.start)
      const end = Math.min(totalCount, start + shape.limit)
      return { start, end }
    }
  }
}

const toDisplayTimeline = (
  forkId: string | null,
  fork: Pick<
    DisplayTimelineState,
    'mode' | 'streamingMessageId'
  >,
  messages: readonly DisplayMessage[],
  window: {
    readonly start: number
    readonly end: number
    readonly totalCount: number
  },
  presentationMode: DisplayTimelinePresentationMode,
): DisplayTimeline =>
  materializeDisplayTimeline(
    {
      mode: fork.mode,
      streamingMessageId: fork.streamingMessageId,
    },
    messages,
    {
      ...window,
      hasMoreBefore: window.start > 0,
      hasMoreAfter: window.end < window.totalCount,
    },
    buildDisplayTimelinePresentation({
      scope: forkId === null ? 'root' : 'fork',
      mode: presentationMode,
      timelineMode: fork.mode,
      streamingMessageId: fork.streamingMessageId,
      messages,
      window: {
        ...window,
        hasMoreBefore: window.start > 0,
        hasMoreAfter: window.end < window.totalCount,
      },
    }),
  )

export const buildDisplayViewSnapshot = (
  shape: DisplayViewShape
) =>
  Effect.gen(function* () {
    const timeline = yield* Projection.consumer.read(DisplayTimelineProjection)
    const sessionContext = yield* Projection.consumer.read(SessionContextProjection)
    const chatTitle = yield* Projection.consumer.read(ChatTitleProjection)
    const agentStatus = yield* Projection.consumer.read(AgentLifecycleProjection)
    const taskWorker = yield* Projection.consumer.read(TaskAssignmentProjection)
    const windowState = yield* Projection.consumer.read(WindowProjection)
    const compaction = yield* Projection.consumer.read(CompactionProjection)
    const ambient = yield* AmbientServiceTag
    const sessionOptions = ambient.getValue(SessionOptionsAmbient)

    const timelines: Record<string, DisplayTimeline> = {}
    const acceptedTimelines: Record<string, typeof shape.timelines[string]> = {}

    for (const [forkKey, timelineShape] of Object.entries(shape.timelines)) {
      const forkId = forkKeyToForkId(forkKey as ForkKey)
      const fork = timeline.state.forks.get(forkId)
      if (!fork) continue

      const messages = timeline.addressed.forFork(forkId).messages
      const window = timelineShape.kind === 'tail'
        ? messages.resolveTailWindow(fork.messages, timelineShape.limit)
        : messages.resolveRangeWindow(fork.messages, timelineShape.start, timelineShape.limit)
      const visibleMessages = yield* messages.readWindow(window)
      const bounds = windowBounds(timelineShape, fork.messages.totalCount)

      acceptedTimelines[forkKey] = timelineShape
      timelines[forkKey] = toDisplayTimeline(
        forkId,
        fork,
        visibleMessages,
        {
          ...bounds,
          totalCount: fork.messages.totalCount,
        },
        timelineShape.presentation,
      )
    }

    const acceptedShape: DisplayViewShape = { timelines: acceptedTimelines }

    const state: DisplayState = {
      session: materializeDisplaySession({
        sessionId: sessionOptions.sessionId ?? '',
        title: chatTitle.state.chatName,
        cwd: sessionContext.state.context?.cwd ?? '',
      }),
      timelines,
      actors: materializeDisplayActors(
        agentStatus.state,
        taskWorker.state,
        windowState.state,
        compaction.state
      ),
      agents: materializeDisplayAgents(agentStatus.state),
      tasks: materializeDisplayTasks(taskWorker.state),
    }

    return { shape: acceptedShape, state }
  })
