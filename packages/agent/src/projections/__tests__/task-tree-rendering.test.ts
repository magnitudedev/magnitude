
import { describe, expect, it } from 'bun:test'
import { Effect, Layer } from 'effect'
import {
  ProjectionBusTag,
  makeProjectionBusLayer,
  FrameworkErrorPubSubLive,
  FrameworkErrorReporterLive,
} from '@magnitudedev/event-core'
import type { AppEvent } from '../../events'
import { AgentStatusProjection } from '../agent-status'
import { MemoryProjection, type ForkMemoryState } from '../memory'
import { SubagentActivityProjection } from '../subagent-activity'
import { CanonicalTurnProjection } from '../canonical-turn'
import { UserPresenceProjection } from '../user-presence'
import { OutboundMessagesProjection } from '../outbound-messages'
import { UserMessageResolutionProjection } from '../user-message-resolution'
import { TaskGraphProjection } from '../task-graph'
import { formatInbox } from '../../inbox/render'

const ts = (n: number) => 1_700_200_000_000 + n

const makeRuntimeLayer = () => {
  const projectionBusLayer = Layer.provideMerge(
    makeProjectionBusLayer<AppEvent>(),
    Layer.provide(FrameworkErrorReporterLive, FrameworkErrorPubSubLive),
  )

  return Layer.mergeAll(
    FrameworkErrorPubSubLive,
    Layer.provide(FrameworkErrorReporterLive, FrameworkErrorPubSubLive),
    projectionBusLayer,
    Layer.provide(AgentStatusProjection.Layer, projectionBusLayer),
    Layer.provide(SubagentActivityProjection.Layer, projectionBusLayer),
    Layer.provide(CanonicalTurnProjection.Layer, projectionBusLayer),
    Layer.provide(UserPresenceProjection.Layer, projectionBusLayer),
    Layer.provide(OutboundMessagesProjection.Layer, projectionBusLayer),
    Layer.provide(UserMessageResolutionProjection.Layer, projectionBusLayer),
    Layer.provide(TaskGraphProjection.Layer, projectionBusLayer),
    Layer.provide(MemoryProjection.Layer, projectionBusLayer),
  )
}

const runEvents = async (events: AppEvent[]) => {
  const runtimeLayer = makeRuntimeLayer()

  const program = Effect.gen(function* () {
    const bus = yield* ProjectionBusTag<AppEvent>()
    const projection = yield* MemoryProjection.Tag

    for (const event of events) {
      yield* bus.processEvent(event as any)
    }

    return yield* projection.getFork(null)
  })

  return Effect.runPromise(program.pipe(Effect.provide(runtimeLayer)) as any) as Promise<ForkMemoryState>
}

const getLastInbox = (fork: ForkMemoryState) => {
  const inbox = [...fork.messages].reverse().find(m => m.type === 'inbox')
  expect(inbox).toBeTruthy()
  return inbox as Extract<ForkMemoryState['messages'][number], { type: 'inbox' }>
}

const inboxText = (inbox: Extract<ForkMemoryState['messages'][number], { type: 'inbox' }>) =>
  formatInbox({
    results: inbox.results,
    timeline: inbox.timeline,
    timezone: null,
  })
    .filter((p): p is { type: 'text', text: string } => p.type === 'text')
    .map(p => p.text)
    .join('')

describe('task tree rendering mechanics', () => {
  it('single task created: emits task_update(created) and one task_tree_view', async () => {
    const rootFork = await runEvents([
      {
        type: 'session_initialized',
        timestamp: ts(0),
        sessionId: 's1',
        cwd: '/tmp',
        model: 'test',
        mode: 'interactive',
        approvalMode: 'on-request',
      } as any,
      {
        type: 'task_created',
        timestamp: ts(1),
        forkId: null,
        taskId: 'root-1',
        title: 'Root 1',

        parentId: null,
      } as any,
      {
        type: 'turn_started',
        timestamp: ts(2),
        turnId: 'turn-1',
        forkId: null,
        strategyId: 'lead',
        chainId: null,
      } as any,
    ])

    const inbox = getLastInbox(rootFork)
    const taskUpdates = inbox.timeline.filter(e => e.kind === 'task_update')
    expect(taskUpdates.some((e: any) => e.action === 'created' && e.taskId === 'root-1')).toBe(true)

    const treeViews = inbox.timeline.filter(e => e.kind === 'task_tree_view')
    expect(treeViews.length).toBe(1)
    expect((treeViews[0] as any).renderedTree).toContain('[pending] implement: Root 1 (root-1)')
  })

  it('multiple tasks under same root: dirty markers dedupe to one root and render full tree once', async () => {
    const rootFork = await runEvents([
      {
        type: 'session_initialized',
        timestamp: ts(0),
        sessionId: 's1',
        cwd: '/tmp',
        model: 'test',
        mode: 'interactive',
        approvalMode: 'on-request',
      } as any,
      {
        type: 'task_created',
        timestamp: ts(1),
        forkId: null,
        taskId: 'root-1',
        title: 'Root 1',

        parentId: null,
      } as any,
      {
        type: 'task_created',
        timestamp: ts(2),
        forkId: null,
        taskId: 'child-a',
        title: 'Child A',

        parentId: 'root-1',
      } as any,
      {
        type: 'task_created',
        timestamp: ts(3),
        forkId: null,
        taskId: 'child-b',
        title: 'Child B',

        parentId: 'root-1',
      } as any,
      {
        type: 'turn_started',
        timestamp: ts(4),
        turnId: 'turn-1',
        forkId: null,
        strategyId: 'lead',
        chainId: null,
      } as any,
    ])

    const inbox = getLastInbox(rootFork)
    const treeViews = inbox.timeline.filter(e => e.kind === 'task_tree_view')
    expect(treeViews.length).toBe(1)
    const rendered = (treeViews[0] as any).renderedTree as string
    expect(rendered.match(/\[pending\] feature: Root 1 \(root-1\)/g)?.length ?? 0).toBe(1)
    expect(rendered).toContain('[pending] implement: Child A (child-a)')
    expect(rendered).toContain('[pending] review: Child B (child-b)')
  })

  it('tasks under different roots: flush renders both roots', async () => {
    const rootFork = await runEvents([
      {
        type: 'session_initialized',
        timestamp: ts(0),
        sessionId: 's1',
        cwd: '/tmp',
        model: 'test',
        mode: 'interactive',
        approvalMode: 'on-request',
      } as any,
      {
        type: 'task_created',
        timestamp: ts(1),
        forkId: null,
        taskId: 'root-a',
        title: 'Root A',

        parentId: null,
      } as any,
      {
        type: 'task_created',
        timestamp: ts(2),
        forkId: null,
        taskId: 'root-b',
        title: 'Root B',

        parentId: null,
      } as any,
      {
        type: 'task_created',
        timestamp: ts(3),
        forkId: null,
        taskId: 'a-child',
        title: 'A Child',

        parentId: 'root-a',
      } as any,
      {
        type: 'task_created',
        timestamp: ts(4),
        forkId: null,
        taskId: 'b-child',
        title: 'B Child',

        parentId: 'root-b',
      } as any,
      {
        type: 'turn_started',
        timestamp: ts(5),
        turnId: 'turn-1',
        forkId: null,
        strategyId: 'lead',
        chainId: null,
      } as any,
    ])

    const inbox = getLastInbox(rootFork)
    const treeViews = inbox.timeline.filter(e => e.kind === 'task_tree_view')
    expect(treeViews.length).toBe(1)
    const rendered = (treeViews[0] as any).renderedTree as string
    expect(rendered).toContain('[pending] feature: Root A (root-a)')
    expect(rendered).toContain('[pending] bug: Root B (root-b)')
  })

  it('no task signals: no task_tree_view and no <task_tree> in formatted inbox', async () => {
    const rootFork = await runEvents([
      {
        type: 'session_initialized',
        timestamp: ts(0),
        sessionId: 's1',
        cwd: '/tmp',
        model: 'test',
        mode: 'interactive',
        approvalMode: 'on-request',
      } as any,
      {
        type: 'turn_started',
        timestamp: ts(1),
        turnId: 'turn-1',
        forkId: null,
        strategyId: 'lead',
        chainId: null,
      } as any,
    ])

    const inbox = [...rootFork.messages].reverse().find(m => m.type === 'inbox')
    if (!inbox) {
      expect(inbox).toBeUndefined()
      return
    }

    expect(inbox.timeline.some(e => e.kind === 'task_tree_view')).toBe(false)
    expect(inboxText(inbox as any)).not.toContain('<task_tree>')
  })

  it('duplicate dirty markers for same task: deduplicates and renders once', async () => {
    const rootFork = await runEvents([
      {
        type: 'session_initialized',
        timestamp: ts(0),
        sessionId: 's1',
        cwd: '/tmp',
        model: 'test',
        mode: 'interactive',
        approvalMode: 'on-request',
      } as any,
      {
        type: 'task_created',
        timestamp: ts(1),
        forkId: null,
        taskId: 'task-dup',
        title: 'Task Dup',

        parentId: null,
      } as any,
      {
        type: 'task_updated',
        timestamp: ts(2),
        forkId: null,
        taskId: 'task-dup',
        patch: { status: 'completed' },
      } as any,
      {
        type: 'turn_started',
        timestamp: ts(3),
        turnId: 'turn-1',
        forkId: null,
        strategyId: 'lead',
        chainId: null,
      } as any,
    ])

    const inbox = getLastInbox(rootFork)
    const treeViews = inbox.timeline.filter(e => e.kind === 'task_tree_view')
    expect(treeViews.length).toBe(1)
    const rendered = (treeViews[0] as any).renderedTree as string
    expect(rendered.match(/\(task-dup\)/g)?.length ?? 0).toBe(1)
  })

  it('task created then completed in same turn: one tree render with final state', async () => {
    const rootFork = await runEvents([
      {
        type: 'session_initialized',
        timestamp: ts(0),
        sessionId: 's1',
        cwd: '/tmp',
        model: 'test',
        mode: 'interactive',
        approvalMode: 'on-request',
      } as any,
      {
        type: 'task_created',
        timestamp: ts(1),
        forkId: null,
        taskId: 'task-final',
        title: 'Task Final',

        parentId: null,
      } as any,
      {
        type: 'task_updated',
        timestamp: ts(2),
        forkId: null,
        taskId: 'task-final',
        patch: { status: 'completed' },
      } as any,
      {
        type: 'turn_started',
        timestamp: ts(3),
        turnId: 'turn-1',
        forkId: null,
        strategyId: 'lead',
        chainId: null,
      } as any,
    ])

    const inbox = getLastInbox(rootFork)
    const treeViews = inbox.timeline.filter(e => e.kind === 'task_tree_view')
    expect(treeViews.length).toBe(1)
    expect((treeViews[0] as any).renderedTree).toContain('[done] implement: Task Final (task-final)')
  })

  it('shows assigned worker role for non-user worker tasks', async () => {
    const rootFork = await runEvents([
      {
        type: 'session_initialized',
        timestamp: ts(0),
        sessionId: 's1',
        cwd: '/tmp',
        model: 'test',
        mode: 'interactive',
        approvalMode: 'on-request',
      } as any,
      {
        type: 'task_created',
        timestamp: ts(1),
        forkId: null,
        taskId: 'diag-1',
        title: 'Root cause analysis',

        parentId: null,
      } as any,
      {
        type: 'task_assigned',
        timestamp: ts(2),
        forkId: null,
        taskId: 'diag-1',
        assignee: 'debugger',
        workerInfo: { agentId: 'agent-1', forkId: 'fork-1', role: 'debugger', message: null },
      } as any,
      {
        type: 'turn_started',
        timestamp: ts(3),
        turnId: 'turn-1',
        forkId: null,
        strategyId: 'lead',
        chainId: null,
      } as any,
    ])

    const inbox = getLastInbox(rootFork)
    const treeViews = inbox.timeline.filter(e => e.kind === 'task_tree_view')
    expect(treeViews.length).toBe(1)
    expect((treeViews[0] as any).renderedTree).toContain(
      '[pending] diagnose: Root cause analysis (diag-1, assigned: debugger)',
    )
  })

  it('task_tree_dirty entries are excluded from chronological timeline render', async () => {
    const rootFork = await runEvents([
      {
        type: 'session_initialized',
        timestamp: ts(0),
        sessionId: 's1',
        cwd: '/tmp',
        model: 'test',
        mode: 'interactive',
        approvalMode: 'on-request',
      } as any,
      {
        type: 'task_created',
        timestamp: ts(1),
        forkId: null,
        taskId: 'task-hidden',
        title: 'Task Hidden',

        parentId: null,
      } as any,
      {
        type: 'turn_started',
        timestamp: ts(2),
        turnId: 'turn-1',
        forkId: null,
        strategyId: 'lead',
        chainId: null,
      } as any,
    ])

    const inbox = getLastInbox(rootFork)
    expect(inbox.timeline.some(e => e.kind === 'task_tree_dirty')).toBe(false)
    const rendered = inboxText(inbox)
    expect(rendered).toContain('<task_tree>')
    expect(rendered).not.toContain('task_tree_dirty')
  })

  it('cancel-only mutation emits task_update(cancelled) even when task_tree_view is empty', async () => {
    const rootFork = await runEvents([
      {
        type: 'session_initialized',
        timestamp: ts(0),
        sessionId: 's1',
        cwd: '/tmp',
        model: 'test',
        mode: 'interactive',
        approvalMode: 'on-request',
      } as any,
      {
        type: 'task_created',
        timestamp: ts(1),
        forkId: null,
        taskId: 'task-cancel',
        title: 'Task Cancel',

        parentId: null,
      } as any,
      {
        type: 'turn_started',
        timestamp: ts(2),
        turnId: 'turn-init',
        forkId: null,
        strategyId: 'lead',
        chainId: null,
      } as any,
      {
        type: 'task_cancelled',
        timestamp: ts(3),
        forkId: null,
        taskId: 'task-cancel',
        cancelledSubtree: ['task-cancel'],
      } as any,
      {
        type: 'turn_started',
        timestamp: ts(4),
        turnId: 'turn-cancel',
        forkId: null,
        strategyId: 'lead',
        chainId: null,
      } as any,
    ])

    const inbox = getLastInbox(rootFork)
    expect(inbox.timeline.some((e: any) => e.kind === 'task_update' && e.action === 'cancelled' && e.taskId === 'task-cancel')).toBe(true)
    expect(inbox.timeline.some(e => e.kind === 'task_tree_view')).toBe(false)
  })

  it('cancel-only after assistant turn still appends inbox on next turn_started', async () => {
    const rootFork = await runEvents([
      {
        type: 'session_initialized',
        timestamp: ts(0),
        sessionId: 's1',
        cwd: '/tmp',
        model: 'test',
        mode: 'interactive',
        approvalMode: 'on-request',
      } as any,
      {
        type: 'task_created',
        timestamp: ts(1),
        forkId: null,
        taskId: 'task-regression',
        title: 'Task Regression',

        parentId: null,
      } as any,
      {
        type: 'turn_started',
        timestamp: ts(2),
        turnId: 'turn-1',
        forkId: null,
        strategyId: 'lead',
        chainId: null,
      } as any,
      {
        type: 'turn_completed',
        timestamp: ts(3),
        turnId: 'turn-1',
        forkId: null,
        strategyId: 'lead',
        result: { success: true, errors: [], oneshotLivenessTriggered: false },
      } as any,
      {
        type: 'task_cancelled',
        timestamp: ts(4),
        forkId: null,
        taskId: 'task-regression',
        cancelledSubtree: ['task-regression'],
      } as any,
      {
        type: 'turn_started',
        timestamp: ts(5),
        turnId: 'turn-2',
        forkId: null,
        strategyId: 'lead',
        chainId: null,
      } as any,
    ])

    const lastMessage = rootFork.messages[rootFork.messages.length - 1]
    expect(lastMessage?.type).toBe('inbox')
    const inbox = lastMessage as Extract<ForkMemoryState['messages'][number], { type: 'inbox' }>
    expect(inbox.timeline.some((e: any) => e.kind === 'task_update' && e.action === 'cancelled')).toBe(true)
  })
})
