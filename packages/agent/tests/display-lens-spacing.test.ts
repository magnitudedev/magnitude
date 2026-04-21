import { describe, expect, it } from '@effect/vitest'
import { Effect } from 'effect'
import { TestHarness, TestHarnessLive } from '../src/test-harness/harness'
import { DisplayProjection } from '../src/projections/display'

const ts = (n: number) => 1_700_000_000_000 + n

describe('display lens spacing', () => {
  it.live('should keep all lens content in one thinking step without extra spaces', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness

      // Start a turn
      yield* h.send({
        type: 'turn_started',
        timestamp: ts(1),
        forkId: null,
        turnId: 'turn-1',
        chainId: 'chain-1',
      } as any)

      // Start first lens (turn)
      yield* h.send({
        type: 'lens_start',
        timestamp: ts(2),
        forkId: null,
        turnId: 'turn-1',
        lens: 'turn',
      } as any)

      // Add content to first lens
      yield* h.send({
        type: 'lens_chunk',
        timestamp: ts(3),
        forkId: null,
        turnId: 'turn-1',
        lens: 'turn',
        text: 'First lens content',
      } as any)

      // Start second lens (skills)
      yield* h.send({
        type: 'lens_start',
        timestamp: ts(4),
        forkId: null,
        turnId: 'turn-1',
        lens: 'skills',
      } as any)

      // Add content to second lens (with newline prefix from parser)
      yield* h.send({
        type: 'lens_chunk',
        timestamp: ts(5),
        forkId: null,
        turnId: 'turn-1',
        lens: 'skills',
        text: '\nSecond lens content',
      } as any)

      // Check the display state
      const display = yield* h.projectionFork(DisplayProjection.Tag, null)
      
      // Find the think block
      const thinkBlock = display.messages.find(m => m.type === 'think_block')
      expect(thinkBlock).toBeDefined()
      expect(thinkBlock?.type).toBe('think_block')
      
      if (thinkBlock && thinkBlock.type === 'think_block') {
        // Should have exactly 1 thinking step (all lens content in one step)
        const thinkingSteps = thinkBlock.steps.filter(s => s.type === 'thinking')
        expect(thinkingSteps.length).toBe(1)
        
        // Content should have newline between lenses, no extra space
        const step = thinkingSteps[0]
        expect(step.content).toBe('First lens content\nSecond lens content')
      }
    }).pipe(Effect.provide(TestHarnessLive()))
  )

  it.live('should not add space between lens chunks', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness

      // Start a turn
      yield* h.send({
        type: 'turn_started',
        timestamp: ts(1),
        forkId: null,
        turnId: 'turn-1',
        chainId: 'chain-1',
      } as any)

      // Start first lens
      yield* h.send({
        type: 'lens_start',
        timestamp: ts(2),
        forkId: null,
        turnId: 'turn-1',
        lens: 'turn',
      } as any)

      // Content for first lens
      yield* h.send({
        type: 'lens_chunk',
        timestamp: ts(3),
        forkId: null,
        turnId: 'turn-1',
        lens: 'turn',
        text: 'Alignment reasoning',
      } as any)

      // Start second lens
      yield* h.send({
        type: 'lens_start',
        timestamp: ts(4),
        forkId: null,
        turnId: 'turn-1',
        lens: 'tasks',
      } as any)

      // Content for second lens
      yield* h.send({
        type: 'lens_chunk',
        timestamp: ts(5),
        forkId: null,
        turnId: 'turn-1',
        lens: 'tasks',
        text: '\nTasks reasoning',
      } as any)

      const display = yield* h.projectionFork(DisplayProjection.Tag, null)
      const thinkBlock = display.messages.find(m => m.type === 'think_block')
      
      if (thinkBlock && thinkBlock.type === 'think_block') {
        const thinkingSteps = thinkBlock.steps.filter(s => s.type === 'thinking')
        expect(thinkingSteps.length).toBe(1)
        // No extra space — just the newline from parser
        expect(thinkingSteps[0].content).toBe('Alignment reasoning\nTasks reasoning')
      }
    }).pipe(Effect.provide(TestHarnessLive()))
  )
})