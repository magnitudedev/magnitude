import { describe, expect, test } from 'bun:test'
import { withHarness } from '../src/test-harness'
import { CompactionProjection } from '../src/projections/compaction'
import { WorkingStateProjection, shouldTrigger } from '../src/projections/working-state'
import { MemoryProjection } from '../src/projections/memory'
import type { AppEvent, SessionContext } from '../src/events'
import { SYSTEM_PROMPT_TOKENS } from '../src/generated/system-prompt-size'
import { CHARS_PER_TOKEN } from '../src/constants'
import { textParts } from '../src/content'

const root = null

const baseContext = (): SessionContext => ({
  cwd: '/tmp/project',
  workspacePath: '/tmp/test-workspace',
  platform: 'macos',
  shell: '/bin/zsh',
  timezone: 'UTC',
  username: 'tester',
  fullName: null,
  git: null,
  folderStructure: '.',
  agentsFile: null,
  skills: null,
  userMemory: null,
})

const mkTurnCompleted = (overrides: Partial<Extract<AppEvent, { type: 'turn_completed' }>> = {}): Extract<AppEvent, { type: 'turn_completed' }> => ({
  type: 'turn_completed',
  forkId: root,
  turnId: 'turn-1',
  chainId: 'chain-1',
  strategyId: 'xml-act',
  responseParts: [],
  toolCalls: [],
  observedResults: [],

  inputTokens: null,
  outputTokens: null,
  cacheReadTokens: null,
  cacheWriteTokens: null,
  result: { success: true, turnDecision: 'yield' },
  ...overrides,
})

describe('Compaction', async () => {
  describe('Projection state transitions', async () => {
    test('initial token estimate includes system prompt tokens', async () =>
      withHarness(async (h) => {
        const state = await h.projectionFork(CompactionProjection.Tag, root)
        expect(state.tokenEstimate).toBeGreaterThanOrEqual(SYSTEM_PROMPT_TOKENS)
        expect(state.shouldCompact).toBe(false)
      }))

    test('user_message increments token estimate', async () =>
      withHarness(async (h) => {
        const before = await h.projectionFork(CompactionProjection.Tag, root)

        await h.send({
          type: 'user_message',
          forkId: root,
          content: textParts('A'.repeat(300)),
          attachments: [],
          mode: 'text',
          synthetic: false,
          taskMode: false,
        })

        const after = await h.projectionFork(CompactionProjection.Tag, root)
        expect(after.tokenEstimate).toBe(before.tokenEstimate + Math.ceil(300 / CHARS_PER_TOKEN))
      }))

    test('turn_completed with inputTokens resets estimate baseline', async () =>
      withHarness(async (h) => {
        await h.send({
          type: 'user_message',
          forkId: root,
          content: textParts('hello'),
          attachments: [],
          mode: 'text',
          synthetic: false,
          taskMode: false,
        })

        await h.send(mkTurnCompleted({
          inputTokens: 1234,
          responseParts: [{ type: 'text', content: 'A'.repeat(300) }],
        }))

        const state = await h.projectionFork(CompactionProjection.Tag, root)
        expect(state.tokenEstimate).toBe(1234 + Math.ceil(300 / CHARS_PER_TOKEN))
      }))

    test('turn_completed without inputTokens adds to estimate', async () =>
      withHarness(async (h) => {
        const before = await h.projectionFork(CompactionProjection.Tag, root)

        await h.send(mkTurnCompleted({
          inputTokens: null,
          responseParts: [{ type: 'text', content: 'B'.repeat(150) }],
        }))

        const after = await h.projectionFork(CompactionProjection.Tag, root)
        expect(after.tokenEstimate).toBe(before.tokenEstimate + Math.ceil(150 / CHARS_PER_TOKEN))
      }))

    test('compaction_started sets isCompacting', async () =>
      withHarness(async (h) => {
        await h.send({
          type: 'compaction_started',
          forkId: root,
          compactedMessageCount: 0,
        })

        const state = await h.projectionFork(CompactionProjection.Tag, root)
        expect(state.isCompacting).toBe(true)
      }))

    test('compaction_ready sets pendingFinalization and pending payload', async () =>
      withHarness(async (h) => {
        await h.send({
          type: 'compaction_started',
          forkId: root,
          compactedMessageCount: 0,
        })

        await h.send({
          type: 'compaction_ready',
          forkId: root,
          summary: 'short summary',
          compactedMessageCount: 2,
          originalTokenEstimate: 500,
          refreshedContext: null,
        })

        const state = await h.projectionFork(CompactionProjection.Tag, root)
        expect(state.isCompacting).toBe(true)
        expect(state.pendingFinalization).toBe(true)
        expect(state.pendingCompactionData?.summary).toBe('short summary')
      }))

    test('compaction_completed clears flags and subtracts tokensSaved', async () =>
      withHarness(async (h) => {
        await h.send({
          type: 'compaction_started',
          forkId: root,
          compactedMessageCount: 0,
        })
        await h.send({
          type: 'compaction_ready',
          forkId: root,
          summary: 'summary',
          compactedMessageCount: 1,
          originalTokenEstimate: 500,
          refreshedContext: null,
        })
        const before = await h.projectionFork(CompactionProjection.Tag, root)

        await h.send({
          type: 'compaction_completed',
          forkId: root,
          summary: 'summary',
          compactedMessageCount: 1,
          tokensSaved: 50,
          preservedVariables: [],
          refreshedContext: null,
        })

        const after = await h.projectionFork(CompactionProjection.Tag, root)
        expect(after.tokenEstimate).toBe(Math.max(0, before.tokenEstimate - 50))
        expect(after.isCompacting).toBe(false)
        expect(after.pendingFinalization).toBe(false)
        expect(after.contextLimitBlocked).toBe(false)
        expect(after.pendingCompactionData).toBeNull()
      }))

    test('compaction_failed clears flags', async () =>
      withHarness(async (h) => {
        await h.send({
          type: 'compaction_started',
          forkId: root,
          compactedMessageCount: 0,
        })
        await h.send({
          type: 'compaction_ready',
          forkId: root,
          summary: 'summary',
          compactedMessageCount: 1,
          originalTokenEstimate: 500,
          refreshedContext: null,
        })

        await h.send({
          type: 'compaction_failed',
          forkId: root,
          error: 'failure',
        })

        const state = await h.projectionFork(CompactionProjection.Tag, root)
        expect(state.isCompacting).toBe(false)
        expect(state.pendingFinalization).toBe(false)
        expect(state.contextLimitBlocked).toBe(false)
        expect(state.pendingCompactionData).toBeNull()
      }))
  })

  describe('Context limit behavior', async () => {
    test('context_limit_hit forces contextLimitBlocked true', async () =>
      withHarness(async (h) => {
        await h.send({ type: 'context_limit_hit', forkId: root, error: 'cap hit' })
        const state = await h.projectionFork(CompactionProjection.Tag, root)
        expect(state.contextLimitBlocked).toBe(true)
      }))

    test('context_limit_hit when not compacting keeps state.shouldCompact unchanged', async () =>
      withHarness(async (h) => {
        const before = await h.projectionFork(CompactionProjection.Tag, root)
        await h.send({ type: 'context_limit_hit', forkId: root, error: 'cap hit' })
        const state = await h.projectionFork(CompactionProjection.Tag, root)
        expect(state.shouldCompact).toBe(before.shouldCompact)
      }))
  })

  describe('Working state gating', async () => {
    test('compactionPending blocks shouldTrigger', async () =>
      withHarness(async (h) => {
        await h.send({
          type: 'user_message',
          forkId: root,
          content: textParts('wake up'),
          attachments: [],
          mode: 'text',
          synthetic: false,
          taskMode: false,
        })

        await h.send({
          type: 'compaction_ready',
          forkId: root,
          summary: 'summary',
          compactedMessageCount: 1,
          originalTokenEstimate: 10,
          refreshedContext: null,
        })

        const after = await h.projectionFork(WorkingStateProjection.Tag, root)
        expect(after.compactionPending).toBe(true)
        expect(shouldTrigger(after)).toBe(false)
      }))

    test('contextLimitBlocked blocks shouldTrigger', async () =>
      withHarness(async (h) => {
        await h.send({
          type: 'user_message',
          forkId: root,
          content: textParts('wake up'),
          attachments: [],
          mode: 'text',
          synthetic: false,
          taskMode: false,
        })
        await h.send({ type: 'context_limit_hit', forkId: root, error: 'cap hit' })

        const state = await h.projectionFork(WorkingStateProjection.Tag, root)
        expect(state.contextLimitBlocked).toBe(true)
        expect(shouldTrigger(state)).toBe(false)
      }))

    test('compaction_completed unblocks working state', async () =>
      withHarness(async (h) => {
        await h.send({ type: 'context_limit_hit', forkId: root, error: 'cap hit' })
        await h.send({
          type: 'compaction_ready',
          forkId: root,
          summary: 'summary',
          compactedMessageCount: 1,
          originalTokenEstimate: 10,
          refreshedContext: null,
        })
        await h.send({
          type: 'compaction_completed',
          forkId: root,
          summary: 'summary',
          compactedMessageCount: 1,
          tokensSaved: 5,
          preservedVariables: [],
          refreshedContext: null,
        })

        const state = await h.projectionFork(WorkingStateProjection.Tag, root)
        expect(state.contextLimitBlocked).toBe(false)
        expect(state.compactionPending).toBe(false)
      }))

    test('compaction_failed unblocks working state', async () =>
      withHarness(async (h) => {
        await h.send({ type: 'context_limit_hit', forkId: root, error: 'cap hit' })
        await h.send({
          type: 'compaction_ready',
          forkId: root,
          summary: 'summary',
          compactedMessageCount: 1,
          originalTokenEstimate: 10,
          refreshedContext: null,
        })
        await h.send({
          type: 'compaction_failed',
          forkId: root,
          error: 'failure',
        })

        const state = await h.projectionFork(WorkingStateProjection.Tag, root)
        expect(state.contextLimitBlocked).toBe(false)
        expect(state.compactionPending).toBe(true)
      }))
  })

  describe('Memory rewrite', async () => {
    test('compaction_completed rewrites messages with summary', async () =>
      withHarness(async (h) => {
        await h.send({
          type: 'user_message',
          forkId: root,
          content: textParts('first message'),
          attachments: [],
          mode: 'text',
          synthetic: false,
          taskMode: false,
        })
        await h.send({
          type: 'user_message',
          forkId: root,
          content: textParts('second message'),
          attachments: [],
          mode: 'text',
          synthetic: false,
          taskMode: false,
        })

        await h.send({
          type: 'compaction_completed',
          forkId: root,
          summary: 'compacted summary',
          compactedMessageCount: 2,
          tokensSaved: 20,
          preservedVariables: [],
          refreshedContext: null,
        })

        const memory = await h.projectionFork(MemoryProjection.Tag, root)
        expect(memory.messages[1]?.type).toBe('compacted')
        if (memory.messages[1]?.type === 'compacted') {
          expect(memory.messages[1].content[0]?.type).toBe('text')
        }
      }))

    test('refreshedContext replaces session context', async () =>
      withHarness(async (h) => {
        const refreshed: SessionContext = {
          ...baseContext(),
          cwd: '/tmp/new-cwd',
        }

        await h.send({
          type: 'compaction_completed',
          forkId: root,
          summary: 'summary',
          compactedMessageCount: 0,
          tokensSaved: 20,
          preservedVariables: [],
          refreshedContext: refreshed,
        })

        const memory = await h.projectionFork(MemoryProjection.Tag, root)
        expect(memory.messages[0]?.type).toBe('session_context')
        if (memory.messages[0]?.type === 'session_context') {
          const text = memory.messages[0].content.map((p) => (p.type === 'text' ? p.text : '')).join('')
          expect(text.includes('/tmp/new-cwd')).toBe(true)
        }
      }))
  })

  describe('Full lifecycle with compaction worker', async () => {
    test('context_limit_hit triggers worker and emits compaction_ready', async () =>
      withHarness(
        { workers: { compaction: true }, model: { completeResponse: 'worker summary' } },
        async (h) => {
          await h.send({
            type: 'user_message',
            forkId: root,
            content: textParts('X'.repeat(60_000)),
            attachments: [],
            mode: 'text',
            synthetic: false,
            taskMode: false,
          })

          await h.compaction.trigger(root)
          const ready = await h.compaction.waitReady(root)
          expect(ready.summary.length).toBeGreaterThan(0)
        },
      ))
  })
})