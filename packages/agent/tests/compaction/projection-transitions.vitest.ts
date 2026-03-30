import { describe, it } from '@effect/vitest'
import { Effect } from 'effect'
import { expect } from 'vitest'
import { CHARS_PER_TOKEN } from '../../src/constants'
import { getAgentDefinition } from '../../src/agents'
import { renderSystemPrompt } from '../../src/prompts/system-prompt'
import { TestHarness, TestHarnessLive } from '../../src/test-harness/harness'
import {
  ROOT_FORK_ID,
  estimateTokens,
  getCompaction,
  mkCompactionCompleted,
  mkCompactionFailed,
  mkCompactionReady,
  mkCompactionStarted,
  mkContextLimitHit,
  mkTurnCompleted,
  mkUserMessage,
} from './helpers'

const leadSystemPromptTokens = Math.ceil(renderSystemPrompt(getAgentDefinition('lead')).length / CHARS_PER_TOKEN)

describe('compaction/projection-transitions', () => {
  it.effect('initial token estimate includes system prompt tokens', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness
      const state = yield* getCompaction(h, ROOT_FORK_ID)
      expect(state.tokenEstimate).toBeGreaterThanOrEqual(leadSystemPromptTokens)
      expect(state.shouldCompact).toBe(false)
    }).pipe(Effect.provide(TestHarnessLive())))

  it.effect('user_message increments token estimate', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness
      const before = yield* getCompaction(h)
      yield* h.send(mkUserMessage({ text: 'A'.repeat(300) }))
      const after = yield* getCompaction(h)
      expect(after.tokenEstimate).toBe(before.tokenEstimate + estimateTokens('A'.repeat(300)))
    }).pipe(Effect.provide(TestHarnessLive())))

  it.effect('turn_completed with inputTokens resets estimate baseline', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness
      yield* h.send(mkUserMessage({ text: 'hello' }))
      yield* h.send(mkTurnCompleted({
        inputTokens: 1234,
        responseParts: [{ type: 'text', content: 'A'.repeat(300) }],
      }))
      const state = yield* getCompaction(h)
      expect(state.tokenEstimate).toBe(1234 + estimateTokens('A'.repeat(300)))
    }).pipe(Effect.provide(TestHarnessLive())))

  it.effect('turn_completed without inputTokens accumulates estimate', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness
      const before = yield* getCompaction(h)
      yield* h.send(mkTurnCompleted({
        inputTokens: null,
        responseParts: [{ type: 'text', content: 'B'.repeat(150) }],
      }))
      const after = yield* getCompaction(h)
      expect(after.tokenEstimate).toBe(before.tokenEstimate + estimateTokens('B'.repeat(150)))
    }).pipe(Effect.provide(TestHarnessLive())))

  it.effect('compaction_started sets isCompacting and leaves pendingFinalization false', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness
      yield* h.send(mkCompactionStarted())
      const state = yield* getCompaction(h)
      expect(state.isCompacting).toBe(true)
      expect(state.pendingFinalization).toBe(false)
    }).pipe(Effect.provide(TestHarnessLive())))

  it.effect('compaction_ready sets pendingFinalization and pending payload', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness
      yield* h.send(mkCompactionStarted())
      yield* h.send(mkCompactionReady({
        summary: 'short summary',
        compactedMessageCount: 2,
        originalTokenEstimate: 500,
      }))
      const state = yield* getCompaction(h)
      expect(state.isCompacting).toBe(true)
      expect(state.pendingFinalization).toBe(true)
      expect(state.pendingCompactionData?.summary).toBe('short summary')
      expect(state.pendingCompactionData?.compactedMessageCount).toBe(2)
      expect(state.pendingCompactionData?.originalTokenEstimate).toBe(500)
    }).pipe(Effect.provide(TestHarnessLive())))

  it.effect('compaction_completed clears flags and subtracts tokensSaved', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness
      yield* h.send(mkCompactionStarted())
      yield* h.send(mkCompactionReady())
      const before = yield* getCompaction(h)
      yield* h.send(mkCompactionCompleted({ tokensSaved: 50 }))
      const after = yield* getCompaction(h)
      expect(after.tokenEstimate).toBe(Math.max(0, before.tokenEstimate - 50))
      expect(after.isCompacting).toBe(false)
      expect(after.pendingFinalization).toBe(false)
      expect(after.contextLimitBlocked).toBe(false)
      expect(after.pendingCompactionData).toBeNull()
    }).pipe(Effect.provide(TestHarnessLive())))

  it.effect('compaction_failed clears lifecycle flags', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness
      yield* h.send(mkCompactionStarted())
      yield* h.send(mkCompactionReady())
      yield* h.send(mkCompactionFailed())
      const state = yield* getCompaction(h)
      expect(state.isCompacting).toBe(false)
      expect(state.pendingFinalization).toBe(false)
      expect(state.contextLimitBlocked).toBe(false)
      expect(state.pendingCompactionData).toBeNull()
    }).pipe(Effect.provide(TestHarnessLive())))

  it.effect('context_limit_hit forces contextLimitBlocked true', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness
      yield* h.send(mkContextLimitHit())
      const state = yield* getCompaction(h)
      expect(state.contextLimitBlocked).toBe(true)
    }).pipe(Effect.provide(TestHarnessLive())))

  it.effect('context_limit_hit when idle does not mutate shouldCompact directly', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness
      const before = yield* getCompaction(h)
      yield* h.send(mkContextLimitHit())
      const after = yield* getCompaction(h)
      expect(after.shouldCompact).toBe(before.shouldCompact)
    }).pipe(Effect.provide(TestHarnessLive())))
})