import { describe, expect, test } from 'bun:test'
import { withHarness } from '../../test-harness/harness'
import { ContextUsageProjection } from '../context-usage'

describe('ContextUsageProjection', () => {

  test('compaction_completed reduces retained tokens', async () => {
    await withHarness(async (h) => {
      await h.send({
        type: 'turn_completed',
        forkId: null,
        turnId: 't1',
        chainId: 'c1',
        strategyId: 'xml-act',
        responseParts: [],
        toolCalls: [],
        observedResults: [],
        result: { success: true, turnDecision: 'idle' },
        inputTokens: 2000,
        outputTokens: 10,
        cacheReadTokens: 0,
        cacheWriteTokens: 0,
        providerId: 'openai',
        modelId: 'gpt-5',
      })

      await h.send({
        type: 'compaction_completed',
        forkId: null,
        summary: 'summary',
        compactedMessageCount: 1,
        tokensSaved: 500,
        preservedVariables: [],
        refreshedContext: null,
      })

      const state = await h.projectionFork(ContextUsageProjection.Tag, null)
      expect(state.retainedTokens).toBe(1500)
    })
  })

  test('provider inputTokens does not reset retained total downward between turns', async () => {
    await withHarness(async (h) => {
      await h.send({
        type: 'turn_completed',
        forkId: null,
        turnId: 't1',
        chainId: 'c1',
        strategyId: 'xml-act',
        responseParts: [],
        toolCalls: [],
        observedResults: [],
        result: { success: true, turnDecision: 'idle' },
        inputTokens: 2000,
        outputTokens: 10,
        cacheReadTokens: 0,
        cacheWriteTokens: 0,
        providerId: 'openai',
        modelId: 'gpt-5',
      })

      const afterFirstCompletion = await h.projectionFork(ContextUsageProjection.Tag, null)
      expect(afterFirstCompletion.retainedTokens).toBe(2000)
      expect(afterFirstCompletion.source).toBe('provider')

      await h.send({
        type: 'turn_completed',
        forkId: null,
        turnId: 't2',
        chainId: 'c1',
        strategyId: 'xml-act',
        responseParts: [],
        toolCalls: [],
        observedResults: [],
        result: { success: true, turnDecision: 'idle' },
        inputTokens: 144,
        outputTokens: 10,
        cacheReadTokens: 0,
        cacheWriteTokens: 0,
        providerId: 'openai',
        modelId: 'gpt-5',
      })

      const state = await h.projectionFork(ContextUsageProjection.Tag, null)
      expect(state.retainedTokens).toBe(2000)
      expect(state.source).toBe('provider')
    })
  })
})
