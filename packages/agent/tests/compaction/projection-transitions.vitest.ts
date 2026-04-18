import { describe, it } from '@effect/vitest'
import { Effect } from 'effect'
import { expect } from 'vitest'
import { CHARS_PER_TOKEN } from '../../src/constants'
import { getAgentDefinition, getAgentSlot } from '../../src/agents'
import { renderSystemPrompt } from '../../src/prompts/system-prompt'
import { buildResolvedToolSet } from '../../src/tools/resolved-toolset'
import type { ConfigState } from '../../src/ambient/config-ambient'
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
  mkTurnStarted,
  mkUserMessage,
} from './helpers'

// Create a mock config state for testing
const mockConfigState: ConfigState = {
  bySlot: {
    lead: { providerId: 'openai', modelId: 'gpt-4', hardCap: 100000, softCap: 80000 },
    worker: { providerId: 'openai', modelId: 'gpt-4', hardCap: 100000, softCap: 80000 },
  },
}

const leadDef = getAgentDefinition('lead')
const leadToolSet = buildResolvedToolSet(leadDef, mockConfigState, getAgentSlot('lead'))
const leadSystemPromptTokens = Math.ceil(renderSystemPrompt(leadDef, new Map(), leadToolSet).length / CHARS_PER_TOKEN)

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
      yield* h.send(mkTurnStarted({ turnId: 'turn-compaction-a', chainId: 'chain-compaction' }))
      yield* h.send(mkTurnCompleted({
        turnId: 'turn-compaction-a',
        chainId: 'chain-compaction',
        inputTokens: 1234,
      }))
      const state = yield* getCompaction(h)
      expect(state.tokenEstimate).toBe(1234)
    }).pipe(Effect.provide(TestHarnessLive())))

  it.effect('turn_completed without inputTokens preserves estimate when no canonical output is present', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness
      const before = yield* getCompaction(h)
      yield* h.send(mkTurnStarted({ turnId: 'turn-compaction-b', chainId: 'chain-compaction' }))
      yield* h.send(mkTurnCompleted({
        turnId: 'turn-compaction-b',
        chainId: 'chain-compaction',
        inputTokens: null,
      }))
      const after = yield* getCompaction(h)
      expect(after.tokenEstimate).toBe(before.tokenEstimate)
    }).pipe(Effect.provide(TestHarnessLive())))

  it.effect('compaction_started sets isCompacting and leaves pendingFinalization false', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness
      yield* h.send(mkCompactionStarted())
      const state = yield* getCompaction(h)
      expect(state._tag !== 'idle').toBe(true)
      expect(state._tag === 'pendingFinalization').toBe(false)
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
      expect(state._tag !== 'idle').toBe(true)
      expect(state._tag === 'pendingFinalization').toBe(true)
      if (state._tag !== 'pendingFinalization') {
        throw new Error('expected pendingFinalization')
      }
      expect(state.summary).toBe('short summary')
      expect(state.compactedMessageCount).toBe(2)
      expect(state.originalTokenEstimate).toBe(500)
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
      expect(after._tag !== 'idle').toBe(false)
      expect(after._tag === 'pendingFinalization').toBe(false)
      expect(after.contextLimitBlocked).toBe(false)
    }).pipe(Effect.provide(TestHarnessLive())))

  it.effect('compaction_failed clears lifecycle flags', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness
      yield* h.send(mkCompactionStarted())
      yield* h.send(mkCompactionReady())
      yield* h.send(mkCompactionFailed())
      const state = yield* getCompaction(h)
      expect(state._tag !== 'idle').toBe(false)
      expect(state._tag === 'pendingFinalization').toBe(false)
      expect(state.contextLimitBlocked).toBe(false)
    }).pipe(Effect.provide(TestHarnessLive())))

  it.effect('context_limit_hit forces contextLimitBlocked true', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness
      yield* h.send(mkContextLimitHit())
      const state = yield* getCompaction(h)
      expect(state.contextLimitBlocked).toBe(true)
    }).pipe(Effect.provide(TestHarnessLive())))

  it.effect('context_limit_hit when idle sets shouldCompact true', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness
      const before = yield* getCompaction(h)
      expect(before.shouldCompact).toBe(false)
      yield* h.send(mkContextLimitHit())
      const after = yield* getCompaction(h)
      expect(after.shouldCompact).toBe(true)
    }).pipe(Effect.provide(TestHarnessLive())))
})