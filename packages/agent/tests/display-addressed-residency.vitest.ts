import { describe, expect, it } from 'vitest'
import { Addressed, EventEngine } from '@magnitudedev/event-core'
import { Effect, Layer, Option } from 'effect'
import type { ModelAttemptFailureSnapshot, ProviderToolCallId, ToolCallId } from '@magnitudedev/ai'
import type { AppEvent } from '../src/events'
import { toToolKeyErased } from '../src/tools/toolkits'
import { DisplayTimelineProjection } from '../src/display'
import { AgentRoutingProjection } from '../src/projections/agent-routing'
import { AgentLifecycleProjection } from '../src/projections/agent-lifecycle'
import { GoalProjection } from '../src/projections/goal'
import { HarnessStateProjection } from '../src/projections/harness-state'
import { TurnProjection } from '../src/projections/turn'
import { UserMessageResolutionProjection } from '../src/projections/user-message-resolution'
import { makeCountingAddressedEntryStore } from './helpers/counting-addressed-store'

const connectionFailureSnapshot: ModelAttemptFailureSnapshot = {
  phase: 'stream',
  tag: 'StreamOperationalFailure',
  detailTag: 'BodyReadFailure',
  message: 'connection failed',
  call: { provider: 'test', model: 'role/leader', method: 'POST', url: 'http://test' },
  responseStatus: null,
  progress: null,
  retryable: true,
  retryAfterMs: null,
}

const displayMessageSegmentAddress = (forkId: string | null, segmentIndex = 0) =>
  Effect.gen(function* () {
    const timeline = yield* DisplayTimelineProjection.Tag
    const fork = yield* timeline.getFork(forkId)
    const segment = fork.messages.segments[segmentIndex]
    if (!segment) {
      return yield* Effect.dieMessage(`missing display message segment ${segmentIndex} for ${forkId ?? 'root'}`)
    }
    return segment.address
  })

const displayMessageAddress = (forkId: string | null, messageId: string) =>
  Effect.gen(function* () {
    const timeline = yield* DisplayTimelineProjection.Tag
    const fork = yield* timeline.getFork(forkId)
    const address = timeline.addressed.forFork(forkId).messages.resolveAddressForItem(fork.messages, messageId)
    if (Option.isNone(address)) {
      return yield* Effect.dieMessage(`missing display message ${messageId} for ${forkId ?? 'root'}`)
    }
    return address.value
  })

const activeStreamingMessageAddress = (forkId: string | null) =>
  Effect.gen(function* () {
    const timeline = yield* DisplayTimelineProjection.Tag
    const fork = yield* timeline.getFork(forkId)
    if (!fork.streamingMessageId) {
      return yield* Effect.dieMessage(`missing active streaming message for ${forkId ?? 'root'}`)
    }
    return yield* displayMessageAddress(forkId, fork.streamingMessageId)
  })

const readAllMessages = (forkId: string | null) =>
  Effect.gen(function* () {
    const timeline = yield* DisplayTimelineProjection.Tag
    const fork = yield* timeline.getFork(forkId)
    return yield* timeline.addressed.forFork(forkId).messages.readAll(fork.messages)
  })

const makeTestAgent = (name: string) =>
  EventEngine.make<AppEvent>()({
    name,
    schemaVersion: 'test',
    projections: [
      AgentLifecycleProjection,
      AgentRoutingProjection,
      UserMessageResolutionProjection,
      GoalProjection,
      HarnessStateProjection,
      TurnProjection,
      DisplayTimelineProjection,
    ],
    workers: [],
    expose: {
      state: {
        timeline: DisplayTimelineProjection,
      }
    }
  })

/**
 * Display addressed residency tests.
 *
 * The owning projection (DisplayTimelineProjection) never calls pin/unpin.
 * Residency is exactly the pinned set: the writer pin holds what the latest
 * write transaction wrote (streaming rewrites hit memory), consumer reads
 * are pinned by the framework, and everything else lives in the store and
 * loads on access.
 */
describe('display addressed residency', () => {
  it('streaming assistant segments stay resident across chunk events without store reloads', async () => {
    const fixture = await Effect.runPromise(makeCountingAddressedEntryStore)
    const TestAgent = makeTestAgent('DisplayAutoPinStreamingAgent')

    const client = await TestAgent.createClient(
      Layer.succeed(Addressed.AddressedEntryStore, fixture.store)
    )

    try {
      await client.send({ type: 'turn_started', forkId: null, turnId: 'turn-1', chainId: 'chain-1' })
      await client.send({ type: 'message_start', forkId: null, turnId: 'turn-1', id: 'provider-message-1', destination: { kind: 'user' } })
      const streamingAddress = await client.runEffect(activeStreamingMessageAddress(null))
      expect(await Effect.runPromise(fixture.loadCount('DisplayTimeline/messages', streamingAddress))).toBe(0)

      // Multiple chunks — the writer pin keeps the segment resident across commits
      await client.send({ type: 'message_chunk', forkId: null, turnId: 'turn-1', id: 'provider-message-1', text: 'hello' })
      await client.send({ type: 'message_chunk', forkId: null, turnId: 'turn-1', id: 'provider-message-1', text: ' world' })
      await client.send({ type: 'message_chunk', forkId: null, turnId: 'turn-1', id: 'provider-message-1', text: '!' })

      // The streaming segment should not have been loaded from the store —
      // it was created in memory by the write and stayed writer-pinned.
      expect(await Effect.runPromise(fixture.loadCount('DisplayTimeline/messages', streamingAddress))).toBe(0)

      await client.send({ type: 'message_end', forkId: null, turnId: 'turn-1', id: 'provider-message-1' })

      // readAll reads from the resident table — the writer pin kept it resident
      const messages = await client.runEffect(readAllMessages(null))
      expect(messages).toHaveLength(1)
      expect(messages[0]).toMatchObject({ type: 'assistant_message', content: 'hello world!' })
      expect(await Effect.runPromise(fixture.loadCount('DisplayTimeline/messages', streamingAddress))).toBe(0)
    } finally {
      await client.dispose()
    }
  })

  it('active thinking chunks do not refetch older offloaded segments', async () => {
    const fixture = await Effect.runPromise(makeCountingAddressedEntryStore)
    const TestAgent = makeTestAgent('DisplayAutoPinThinkingAgent')

    const client = await TestAgent.createClient(
      Layer.succeed(Addressed.AddressedEntryStore, fixture.store)
    )

    try {
      // Fill the first segment with 50 skill activations
      for (let index = 0; index < 50; index += 1) {
        await client.send({
          type: 'skill_activated',
          forkId: null,
          skillName: 'seed',
          skillPath: '/seed',
          source: 'user',
          message: `old ${index}`,
        })
      }

      await client.send({ type: 'turn_started', forkId: null, turnId: 'turn-thinking', chainId: 'chain-thinking' })
      await client.send({ type: 'thinking_chunk', forkId: null, turnId: 'turn-thinking', text: 'alpha' })

      const oldSegmentAddress = await client.runEffect(displayMessageSegmentAddress(null, 0))
      const activeSegmentAddress = await client.runEffect(displayMessageSegmentAddress(null, 1))
      const oldLoadsAfterFirstChunk = await Effect.runPromise(
        fixture.loadCount('DisplayTimeline/messages', oldSegmentAddress)
      )
      const activeLoadsAfterFirstChunk = await Effect.runPromise(
        fixture.loadCount('DisplayTimeline/messages', activeSegmentAddress)
      )

      // More thinking chunks — both segments stay resident (writer pin + read retention)
      await client.send({ type: 'wake', forkId: null })
      await client.send({ type: 'thinking_chunk', forkId: null, turnId: 'turn-thinking', text: ' beta' })

      expect(await Effect.runPromise(fixture.loadCount('DisplayTimeline/messages', oldSegmentAddress)))
        .toBe(oldLoadsAfterFirstChunk)
      expect(await Effect.runPromise(fixture.loadCount('DisplayTimeline/messages', activeSegmentAddress)))
        .toBe(activeLoadsAfterFirstChunk)

      const messages = await client.runEffect(readAllMessages(null))
      expect(messages).toHaveLength(51)
      expect(messages[50]).toMatchObject({ type: 'thinking', content: 'alpha beta' })
    } finally {
      await client.dispose()
    }
  })

  it('active communication chunks do not refetch older offloaded worker segments', async () => {
    const fixture = await Effect.runPromise(makeCountingAddressedEntryStore)
    const TestAgent = makeTestAgent('DisplayAutoPinCommunicationAgent')

    const client = await TestAgent.createClient(
      Layer.succeed(Addressed.AddressedEntryStore, fixture.store)
    )

    try {
      await client.send({
        type: 'agent_created',
        forkId: 'worker-a',
        parentForkId: null,
        agentId: 'agent-worker-a',
        name: 'Worker A',
        role: 'engineer',
        context: 'worker context',
        mode: 'spawn',
        taskId: 'task-worker-a',
        message: 'starting work',
      })

      for (let index = 0; index < 49; index += 1) {
        await client.send({
          type: 'skill_activated',
          forkId: 'worker-a',
          skillName: 'seed',
          skillPath: '/seed',
          source: 'user',
          message: `old ${index}`,
        })
      }

      await client.send({
        type: 'message_start',
        forkId: null,
        turnId: 'root-turn',
        id: 'communication-message-1',
        destination: { kind: 'worker', agentId: 'agent-worker-a' },
      })

      const workerOldSegmentAddress = await client.runEffect(displayMessageSegmentAddress('worker-a', 0))
      const workerActiveSegmentAddress = await client.runEffect(displayMessageSegmentAddress('worker-a', 1))
      const oldLoadsAfterStart = await Effect.runPromise(
        fixture.loadCount('DisplayTimeline/messages', workerOldSegmentAddress)
      )
      const activeLoadsAfterStart = await Effect.runPromise(
        fixture.loadCount('DisplayTimeline/messages', workerActiveSegmentAddress)
      )

      // Communication chunks arrive on root forkId but affect the worker timeline
      await client.send({ type: 'message_chunk', forkId: null, turnId: 'root-turn', id: 'communication-message-1', text: 'hello' })
      await client.send({ type: 'wake', forkId: null })
      await client.send({ type: 'message_chunk', forkId: null, turnId: 'root-turn', id: 'communication-message-1', text: ' worker' })

      expect(await Effect.runPromise(fixture.loadCount('DisplayTimeline/messages', workerOldSegmentAddress)))
        .toBe(oldLoadsAfterStart)
      expect(await Effect.runPromise(fixture.loadCount('DisplayTimeline/messages', workerActiveSegmentAddress)))
        .toBe(activeLoadsAfterStart)

      const messages = await client.runEffect(readAllMessages('worker-a'))
      expect(messages).toHaveLength(51)
      expect(messages[50]).toMatchObject({
        type: 'agent_communication',
        streamId: 'communication-message-1',
        content: 'hello worker',
        status: 'streaming',
      })
    } finally {
      await client.dispose()
    }
  })

  it('killed worker forks evict to the store and read back on demand', async () => {
    const fixture = await Effect.runPromise(makeCountingAddressedEntryStore)
    const TestAgent = makeTestAgent('DisplayKilledWorkerAgent')

    const client = await TestAgent.createClient(
      Layer.succeed(Addressed.AddressedEntryStore, fixture.store)
    )

    try {
      await client.send({
        type: 'agent_created',
        forkId: 'worker-a',
        parentForkId: null,
        agentId: 'agent-worker-a',
        name: 'Worker A',
        role: 'engineer',
        context: 'worker context',
        mode: 'spawn',
        taskId: 'task-worker-a',
        message: 'starting work',
      })
      await client.send({ type: 'turn_started', forkId: 'worker-a', turnId: 'worker-turn-1', chainId: 'worker-chain-1' })
      await client.send({ type: 'message_start', forkId: 'worker-a', turnId: 'worker-turn-1', id: 'provider-worker-message-1', destination: { kind: 'user' } })

      const workerStreamingAddress = await client.runEffect(activeStreamingMessageAddress('worker-a'))
      const loadsBeforeKill = await Effect.runPromise(
        fixture.loadCount('DisplayTimeline/messages', workerStreamingAddress)
      )

      await client.send({
        type: 'agent_killed',
        forkId: 'worker-a',
        parentForkId: null,
        agentId: 'agent-worker-a',
        reason: 'test kill',
      })

      const result = await client.runEffect(
        Effect.gen(function* () {
          const timeline = yield* DisplayTimelineProjection.Tag
          const fork = yield* timeline.getFork('worker-a')
          const messages = yield* timeline.addressed.forFork('worker-a').messages.readAll(fork.messages)
          return { fork, messages }
        })
      )

      expect(result.fork.streamingMessageId).toBeNull()
      expect(result.fork.mode).toBe('idle')
      expect(result.messages).toMatchObject([
        { type: 'agent_communication', content: 'starting work' },
        { type: 'assistant_message', content: '' },
      ])
      // The kill's parent-fork step rotated the writer pin, evicting the
      // worker's segment to the store; readAll loads it back on demand.
      expect(await Effect.runPromise(fixture.loadCount('DisplayTimeline/messages', workerStreamingAddress)))
        .toBe(loadsBeforeKill + 1)
    } finally {
      await client.dispose()
    }
  })

  it('rejected visible tools keep their segment writer-pinned', async () => {
    const fixture = await Effect.runPromise(makeCountingAddressedEntryStore)
    const TestAgent = makeTestAgent('DisplayAutoPinToolRejectionAgent')

    const client = await TestAgent.createClient(
      Layer.succeed(Addressed.AddressedEntryStore, fixture.store)
    )

    try {
      const toolCallId = 'tool-1' as ToolCallId
      const providerToolCallId = 'tool-1' as ProviderToolCallId

      await client.send({ type: 'turn_started', forkId: null, turnId: 'turn-1', chainId: 'chain-1' })
      await client.send({
        type: 'tool_event',
        forkId: null,
        turnId: 'turn-1',
        toolCallId,
        providerToolCallId,
        toolKey: toToolKeyErased('fileTree'),
        event: {
          _tag: 'ToolInputStarted',
          toolCallId,
          providerToolCallId,
          toolName: 'tree',
          toolKey: toToolKeyErased('fileTree'),
        },
      })

      const toolAddress = await client.runEffect(displayMessageAddress(null, toolCallId))
      // Segment created in memory by the write, held by the writer pin — no store loads
      expect(await Effect.runPromise(fixture.loadCount('DisplayTimeline/messages', toolAddress))).toBe(0)

      await client.send({
        type: 'tool_event',
        forkId: null,
        turnId: 'turn-1',
        toolCallId,
        providerToolCallId,
        toolKey: toToolKeyErased('fileTree'),
        event: {
          _tag: 'ToolInputRejected',
          toolCallId,
          providerToolCallId,
          toolName: 'tree',
          toolKey: toToolKeyErased('fileTree'),
          issue: { path: ['path'], message: 'invalid path' },
        },
      })

      const messages = await client.runEffect(readAllMessages(null))
      expect(messages.some((m) => m.type === 'tool' && m.id === toolCallId)).toBe(true)
      // Still resident — the writer pin kept it
      expect(await Effect.runPromise(fixture.loadCount('DisplayTimeline/messages', toolAddress))).toBe(0)
    } finally {
      await client.dispose()
    }
  })

  it('cancelled turns keep their tool segment writer-pinned', async () => {
    const fixture = await Effect.runPromise(makeCountingAddressedEntryStore)
    const TestAgent = makeTestAgent('DisplayAutoPinToolCancellationAgent')

    const client = await TestAgent.createClient(
      Layer.succeed(Addressed.AddressedEntryStore, fixture.store)
    )

    try {
      const toolCallId = 'tool-cancelled' as ToolCallId
      const providerToolCallId = 'tool-cancelled' as ProviderToolCallId

      await client.send({ type: 'turn_started', forkId: null, turnId: 'turn-1', chainId: 'chain-1' })
      await client.send({
        type: 'tool_event',
        forkId: null,
        turnId: 'turn-1',
        toolCallId,
        providerToolCallId,
        toolKey: toToolKeyErased('fileTree'),
        event: {
          _tag: 'ToolInputStarted',
          toolCallId,
          providerToolCallId,
          toolName: 'tree',
          toolKey: toToolKeyErased('fileTree'),
        },
      })

      const toolAddress = await client.runEffect(displayMessageAddress(null, toolCallId))
      expect(await Effect.runPromise(fixture.loadCount('DisplayTimeline/messages', toolAddress))).toBe(0)

      await client.send({
        type: 'turn_outcome',
        forkId: null,
        turnId: 'turn-1',
        chainId: 'chain-1',
        strategyId: 'native',
        outcome: {
          _tag: 'Cancelled',
          reason: { _tag: 'UserInterrupt' },
          requestId: null,
        },
        commitPolicy: { _tag: 'commitCleanTurn' },
        inputTokens: null,
        outputTokens: null,
        cacheReadTokens: null,
        cacheWriteTokens: null,
        cost: null,
        providerId: 'test',
        modelId: 'role/leader',
      })

      const messages = await client.runEffect(readAllMessages(null))
      expect(messages.some((m) => m.type === 'tool' && m.id === toolCallId)).toBe(true)
      expect(await Effect.runPromise(fixture.loadCount('DisplayTimeline/messages', toolAddress))).toBe(0)
    } finally {
      await client.dispose()
    }
  })

  it('connection failures keep their tool segment writer-pinned', async () => {
    const fixture = await Effect.runPromise(makeCountingAddressedEntryStore)
    const TestAgent = makeTestAgent('DisplayAutoPinConnectionFailureAgent')

    const client = await TestAgent.createClient(
      Layer.succeed(Addressed.AddressedEntryStore, fixture.store)
    )

    try {
      const toolCallId = 'tool-connection-failure' as ToolCallId
      const providerToolCallId = 'tool-connection-failure' as ProviderToolCallId

      await client.send({ type: 'turn_started', forkId: null, turnId: 'turn-1', chainId: 'chain-1' })
      await client.send({
        type: 'tool_event',
        forkId: null,
        turnId: 'turn-1',
        toolCallId,
        providerToolCallId,
        toolKey: toToolKeyErased('fileTree'),
        event: {
          _tag: 'ToolInputStarted',
          toolCallId,
          providerToolCallId,
          toolName: 'tree',
          toolKey: toToolKeyErased('fileTree'),
        },
      })

      const toolAddress = await client.runEffect(displayMessageAddress(null, toolCallId))
      expect(await Effect.runPromise(fixture.loadCount('DisplayTimeline/messages', toolAddress))).toBe(0)

      await client.send({
        type: 'turn_outcome',
        forkId: null,
        turnId: 'turn-1',
        chainId: 'chain-1',
        strategyId: 'native',
        outcome: {
          _tag: 'ConnectionFailure',
          detail: { _tag: 'ModelAttemptFailure', failure: connectionFailureSnapshot },
          requestId: null,
        },
        commitPolicy: { _tag: 'discardPartialAssistant' },
        inputTokens: null,
        outputTokens: null,
        cacheReadTokens: null,
        cacheWriteTokens: null,
        cost: null,
        providerId: 'test',
        modelId: 'role/leader',
      })

      const messages = await client.runEffect(readAllMessages(null))
      expect(messages.some((m) => m.type === 'tool' && m.id === toolCallId)).toBe(true)
      expect(messages.some((m) => m.type === 'status_indicator' && m.message === 'Connection issue: retrying')).toBe(true)
      expect(await Effect.runPromise(fixture.loadCount('DisplayTimeline/messages', toolAddress))).toBe(0)
    } finally {
      await client.dispose()
    }
  })
})
