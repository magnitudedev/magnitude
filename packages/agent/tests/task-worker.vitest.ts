import { describe, expect, it } from '@effect/vitest'
import { Effect } from 'effect'
import { TestHarness, TestHarnessLive } from '../src/test-harness/harness'
import { TaskWorkerProjection } from '../src/projections/task-worker'
import { ToolStateProjection, type ToolStateProjectionState } from '../src/projections/tool-state'
import type { ToolHandle } from '../src/tools/tool-handle'

const ts = (n: number) => 1_700_400_000_000 + n

describe('TaskWorkerProjection', () => {
  it.live('derives spawning role from ToolInputReady even when only id was streamed beforehand', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness

      yield* h.send({
        type: 'task_created',
        timestamp: ts(1),
        forkId: null,
        taskId: 'task-1',
        title: 'Build it',
        parentId: null,
      } as any)
      yield* h.send({
        type: 'tool_event',
        timestamp: ts(2),
        forkId: null,
        turnId: 'turn-1',
        toolCallId: 'spawn-1',
        toolKey: 'spawnWorker',
        event: { _tag: 'ToolInputStarted' },
      } as any)
      yield* h.send({
        type: 'tool_event',
        timestamp: ts(2.5),
        forkId: null,
        turnId: 'turn-1',
        toolCallId: 'spawn-1',
        toolKey: 'spawnWorker',
        event: { _tag: 'ToolInputFieldValue', field: 'id', value: 'task-1' },
      } as any)
      yield* h.send({
        type: 'tool_event',
        timestamp: ts(3),
        forkId: null,
        turnId: 'turn-1',
        toolCallId: 'spawn-1',
        toolKey: 'spawnWorker',
        event: {
          _tag: 'ToolInputReady',
          input: { id: 'task-1', role: 'builder', message: 'do it' },
        },
      } as any)
      yield* h.send({
        type: 'agent_created',
        timestamp: ts(4),
        forkId: 'fork-worker',
        parentForkId: null,
        agentId: 'agent-1',
        name: 'Builder',
        role: 'builder',
        context: '',
        mode: 'spawn',
        taskId: 'task-1',
        message: 'do it',
      } as any)
      yield* h.send({
        type: 'task_assigned',
        timestamp: ts(5),
        forkId: null,
        taskId: 'task-1',
        assignee: 'lead',
        workerInfo: {
          agentId: 'agent-1',
          forkId: 'fork-worker',
          role: 'builder',
        },
        message: 'do it',
      } as any)
      yield* h.send({
        type: 'turn_started',
        timestamp: ts(6),
        forkId: 'fork-worker',
        turnId: 'turn-worker',
        chainId: 'chain-worker',
      } as any)

      const state = yield* h.projection(TaskWorkerProjection.Tag)

      expect(state.orderedTaskIds).toEqual(['task-1'])
      expect(state.snapshots.get('task-1')).toEqual({
        taskId: 'task-1',
        title: 'Build it',

        status: 'pending',
        parentId: null,
        depth: 0,
        updatedAt: ts(5),
        assignee: {
          kind: 'worker',
          role: 'builder',
          agentId: 'agent-1',
          forkId: 'fork-worker',
        },
        workerState: {
          status: 'spawning',
          toolCallId: 'spawn-1',
          role: 'builder',
        },
      })
    }).pipe(Effect.provide(TestHarnessLive({ workers: { turnController: false } })))
  )

  it.live('derives killing for an assigned worker with an active killWorker tool', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness

      yield* h.send({
        type: 'task_created',
        timestamp: ts(1),
        forkId: null,
        taskId: 'task-1',
        title: 'Build it',

        parentId: null,
      } as any)
      yield* h.send({
        type: 'agent_created',
        timestamp: ts(2),
        forkId: 'fork-worker',
        parentForkId: null,
        agentId: 'agent-1',
        name: 'Builder',
        role: 'builder',
        context: '',
        mode: 'spawn',
        taskId: 'task-1',
        message: 'do it',
      } as any)
      yield* h.send({
        type: 'task_assigned',
        timestamp: ts(3),
        forkId: null,
        taskId: 'task-1',
        assignee: 'lead',
        workerInfo: {
          agentId: 'agent-1',
          forkId: 'fork-worker',
          role: 'builder',
        },
        message: 'do it',
      } as any)
      yield* h.send({
        type: 'tool_event',
        timestamp: ts(4),
        forkId: null,
        turnId: 'turn-1',
        toolCallId: 'kill-1',
        toolKey: 'killWorker',
        event: { _tag: 'ToolInputStarted' },
      } as any)
      yield* h.send({
        type: 'tool_event',
        timestamp: ts(4.5),
        forkId: null,
        turnId: 'turn-1',
        toolCallId: 'kill-1',
        toolKey: 'killWorker',
        event: { _tag: 'ToolInputFieldValue', field: 'id', value: 'task-1' },
      } as any)
      yield* h.send({
        type: 'tool_event',
        timestamp: ts(5),
        forkId: null,
        turnId: 'turn-1',
        toolCallId: 'kill-1',
        toolKey: 'killWorker',
        event: { _tag: 'ToolInputReady', input: { id: 'task-1' } },
      } as any)

      const state = yield* h.projection(TaskWorkerProjection.Tag)

      expect(state.snapshots.get('task-1')?.workerState).toEqual({
        status: 'killing',
        forkId: 'fork-worker',
        toolCallId: 'kill-1',
      })
    }).pipe(Effect.provide(TestHarnessLive({ workers: { turnController: false } })))
  )

  it.live('spawning takes precedence over killing when both tools are active', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness

      yield* h.send({
        type: 'task_created',
        timestamp: ts(1),
        forkId: null,
        taskId: 'task-1',
        title: 'Build it',

        parentId: null,
      } as any)
      yield* h.send({
        type: 'agent_created',
        timestamp: ts(2),
        forkId: 'fork-worker',
        parentForkId: null,
        agentId: 'agent-1',
        name: 'Builder',
        role: 'builder',
        context: '',
        mode: 'spawn',
        taskId: 'task-1',
        message: 'do it',
      } as any)
      yield* h.send({
        type: 'task_assigned',
        timestamp: ts(3),
        forkId: null,
        taskId: 'task-1',
        assignee: 'lead',
        workerInfo: {
          agentId: 'agent-1',
          forkId: 'fork-worker',
          role: 'builder',
        },
        message: 'do it',
      } as any)

      yield* h.send({
        type: 'tool_event',
        timestamp: ts(4),
        forkId: null,
        turnId: 'turn-1',
        toolCallId: 'kill-1',
        toolKey: 'killWorker',
        event: { _tag: 'ToolInputStarted' },
      } as any)
      yield* h.send({
        type: 'tool_event',
        timestamp: ts(4.5),
        forkId: null,
        turnId: 'turn-1',
        toolCallId: 'kill-1',
        toolKey: 'killWorker',
        event: { _tag: 'ToolInputFieldValue', field: 'id', value: 'task-1' },
      } as any)
      yield* h.send({
        type: 'tool_event',
        timestamp: ts(5),
        forkId: null,
        turnId: 'turn-1',
        toolCallId: 'kill-1',
        toolKey: 'killWorker',
        event: { _tag: 'ToolInputReady', input: { id: 'task-1' } },
      } as any)

      yield* h.send({
        type: 'tool_event',
        timestamp: ts(6),
        forkId: null,
        turnId: 'turn-2',
        toolCallId: 'spawn-1',
        toolKey: 'spawnWorker',
        event: { _tag: 'ToolInputStarted' },
      } as any)
      yield* h.send({
        type: 'tool_event',
        timestamp: ts(6.5),
        forkId: null,
        turnId: 'turn-2',
        toolCallId: 'spawn-1',
        toolKey: 'spawnWorker',
        event: { _tag: 'ToolInputFieldValue', field: 'id', value: 'task-1' },
      } as any)
      yield* h.send({
        type: 'tool_event',
        timestamp: ts(7),
        forkId: null,
        turnId: 'turn-2',
        toolCallId: 'spawn-1',
        toolKey: 'spawnWorker',
        event: {
          _tag: 'ToolInputReady',
          input: { id: 'task-1', role: 'builder', message: 'replace it' },
        },
      } as any)

      const state = yield* h.projection(TaskWorkerProjection.Tag)

      expect(state.snapshots.get('task-1')?.workerState).toEqual({
        status: 'spawning',
        toolCallId: 'spawn-1',
        role: 'builder',
      })
    }).pipe(Effect.provide(TestHarnessLive({ workers: { turnController: false } })))
  )

  it.live('derives working then idle from agent status and activity history after tools settle', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness

      yield* h.send({
        type: 'task_created',
        timestamp: ts(1),
        forkId: null,
        taskId: 'task-1',
        title: 'Build it',

        parentId: null,
      } as any)
      yield* h.send({
        type: 'agent_created',
        timestamp: ts(2),
        forkId: 'fork-worker',
        parentForkId: null,
        agentId: 'agent-1',
        name: 'Builder',
        role: 'builder',
        context: '',
        mode: 'spawn',
        taskId: 'task-1',
        message: 'do it',
      } as any)
      yield* h.send({
        type: 'task_assigned',
        timestamp: ts(3),
        forkId: null,
        taskId: 'task-1',
        assignee: 'lead',
        workerInfo: {
          agentId: 'agent-1',
          forkId: 'fork-worker',
          role: 'builder',
        },
        message: 'do it',
      } as any)
      yield* h.send({
        type: 'turn_started',
        timestamp: ts(4),
        forkId: 'fork-worker',
        turnId: 'turn-worker',
        chainId: 'chain-worker',
      } as any)
      yield* h.send({
        type: 'turn_completed',
        timestamp: ts(10),
        forkId: 'fork-worker',
        turnId: 'turn-worker',
        chainId: 'chain-worker',
        strategyId: 'xml-act',
        result: {
          success: true,
          turnDecision: 'yield',
          errors: [],
        },
        inputTokens: null,
        outputTokens: null,
        cacheReadTokens: null,
        cacheWriteTokens: null,
        providerId: null,
        modelId: null,
      } as any)

      const state = yield* h.projection(TaskWorkerProjection.Tag)
      const workerState = state.snapshots.get('task-1')?.workerState
      expect(workerState).toMatchObject({
        status: 'idle',
        forkId: 'fork-worker',
        accumulatedMs: 6,
        resumeCount: 0,
      })
      // Note: completedAt may be affected by test infrastructure events (turn_unexpected_error)
      // The important thing is the accumulated time is correct
    }).pipe(Effect.provide(TestHarnessLive({ workers: { turnController: false } })))
  )

  it.live('returns unassigned after kill tool settles and assignment clears', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness

      yield* h.send({
        type: 'task_created',
        timestamp: ts(1),
        forkId: null,
        taskId: 'task-1',
        title: 'Build it',

        parentId: null,
      } as any)
      yield* h.send({
        type: 'agent_created',
        timestamp: ts(2),
        forkId: 'fork-worker',
        parentForkId: null,
        agentId: 'agent-1',
        name: 'Builder',
        role: 'builder',
        context: '',
        mode: 'spawn',
        taskId: 'task-1',
        message: 'do it',
      } as any)
      yield* h.send({
        type: 'task_assigned',
        timestamp: ts(3),
        forkId: null,
        taskId: 'task-1',
        assignee: 'lead',
        workerInfo: {
          agentId: 'agent-1',
          forkId: 'fork-worker',
          role: 'builder',
        },
        message: 'do it',
      } as any)
      yield* h.send({
        type: 'tool_event',
        timestamp: ts(4),
        forkId: null,
        turnId: 'turn-1',
        toolCallId: 'kill-1',
        toolKey: 'killWorker',
        event: { _tag: 'ToolInputStarted' },
      } as any)
      yield* h.send({
        type: 'tool_event',
        timestamp: ts(5),
        forkId: null,
        turnId: 'turn-1',
        toolCallId: 'kill-1',
        toolKey: 'killWorker',
        event: { _tag: 'ToolInputReady', input: { id: 'task-1' } },
      } as any)
      yield* h.send({
        type: 'agent_killed',
        timestamp: ts(6),
        forkId: 'fork-worker',
        agentId: 'agent-1',
        reason: 'killed by lead',
      } as any)
      yield* h.send({
        type: 'task_assigned',
        timestamp: ts(7),
        forkId: null,
        taskId: 'task-1',
        assignee: 'lead',
        workerInfo: null,
        message: null,
      } as any)
      yield* h.send({
        type: 'tool_event',
        timestamp: ts(8),
        forkId: null,
        turnId: 'turn-1',
        toolCallId: 'kill-1',
        toolKey: 'killWorker',
        event: { _tag: 'ToolExecutionEnded', output: { id: 'task-1' } },
      } as any)

      const state = yield* h.projection(TaskWorkerProjection.Tag)

      expect(state.snapshots.get('task-1')?.workerState).toEqual({ status: 'unassigned' })
    }).pipe(Effect.provide(TestHarnessLive({ workers: { turnController: false } })))
  )
})
