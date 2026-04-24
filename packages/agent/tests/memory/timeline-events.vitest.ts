import { describe, expect, it } from '@effect/vitest'
import { Effect } from 'effect'
import { TestHarness, TestHarnessLive } from '../../src/test-harness/harness'
import { getView } from '../../src/projections/memory'
import { getRootMemory, inboxMessages, lastInboxMessage, getRenderedUserText } from './helpers'

describe('memory/timeline-events', () => {
  it.live('subagent activity lifecycle (created → activity → idle) appears in timeline', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness
      const subforkId = 'fork-sub-activity'

      yield* h.send({
        type: 'agent_created',
        forkId: subforkId,
        parentForkId: null,
        agentId: 'builder-auth',
        name: 'builder-auth',
        role: 'builder',
        context: '',
        mode: 'spawn',
        taskId: 'task-sub-1',
        message: 'do work',
      })

      yield* h.send({ type: 'turn_started', forkId: subforkId, turnId: 'sub-turn-1', chainId: 'sub-chain-1' })
      yield* h.send({ type: 'message_start', forkId: subforkId, turnId: 'sub-turn-1', id: 'm1', destination: { kind: 'parent' } })
      yield* h.send({ type: 'message_chunk', forkId: subforkId, turnId: 'sub-turn-1', id: 'm1', text: 'working on auth flow' })
      yield* h.send({ type: 'message_end', forkId: subforkId, turnId: 'sub-turn-1', id: 'm1' })
      yield* h.send({
        type: 'turn_outcome',

        forkId: subforkId,
        turnId: 'sub-turn-1',
        chainId: 'sub-chain-1',
        strategyId: 'xml-act',


        outcome: { _tag: 'Completed', completion: { yieldTarget: 'user', feedback: [] } },
        inputTokens: null,
        outputTokens: null,
        cacheReadTokens: null,
        cacheWriteTokens: null,
        providerId: null,
        modelId: null,
      })

      yield* h.send({
        type: 'agent_killed',
        forkId: subforkId,
        parentForkId: null,
        agentId: 'builder-auth',
        reason: 'test cleanup',
      })

      yield* h.send({ type: 'turn_started', forkId: null, turnId: 'root-turn-1', chainId: 'root-chain-1' })

      const memory = yield* getRootMemory(h)
      const inbox = lastInboxMessage(memory)
      expect(inbox?.type).toBe('inbox')
      if (inbox?.type === 'inbox') {
        const blocks = inbox.timeline.filter(t => t.kind === 'agent_block')
        expect(blocks.length).toBeGreaterThan(0)
      }

      const rendered = yield* getRenderedUserText(h)
      expect(rendered).toContain('<agent id="builder-auth"')
      expect(rendered).toContain('status="idle"')
      expect(rendered).toContain('<magnitude:yield_user/>')
    }).pipe(Effect.provide(TestHarnessLive()))
  )

  it.live('user presence changes are deferred then injected at flush', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness

      yield* h.send({ type: 'window_focus_changed', forkId: null, focused: false })
      yield* h.send({ type: 'turn_started', forkId: null, turnId: 't-presence-1', chainId: 'c-presence-1' })

      const after = yield* getRootMemory(h)
      const timeline = inboxMessages(after).flatMap(m => m.type === 'inbox' ? m.timeline : [])
      expect(timeline.some(t => t.kind === 'user_presence')).toBe(true)

      const rendered = yield* getRenderedUserText(h)
      expect(rendered).toContain('<user-presence>')
    }).pipe(Effect.provide(TestHarnessLive()))
  )

  it.live('turn error entries are injected and rendered in results', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness

      yield* h.send({ type: 'turn_started', forkId: null, turnId: 't-rem-1', chainId: 'c-rem-1' })
      yield* h.send({
        type: 'turn_outcome',

        forkId: null,
        turnId: 't-rem-1',
        chainId: 'c-rem-1',
        strategyId: 'xml-act',


        outcome: {
          _tag: 'Completed',
          completion: {
            yieldTarget: 'user',
            feedback: [{ _tag: 'InvalidMessageDestination', destination: 'unknown', message: 'follow up on deployment checks' }],
          },
        },
        inputTokens: null,
        outputTokens: null,
        cacheReadTokens: null,
        cacheWriteTokens: null,
        providerId: null,
        modelId: null,
      })
      yield* h.send({ type: 'turn_started', forkId: null, turnId: 't-rem-2', chainId: 'c-rem-1' })

      const memory = yield* getRootMemory(h)
      const inbox = lastInboxMessage(memory)
      expect(inbox?.type).toBe('inbox')
      if (inbox?.type === 'inbox') {
        expect(inbox.outcomes.some(r => r.kind === 'error' && r.message.includes('follow up on deployment checks'))).toBe(true)
      }

      const rendered = yield* getRenderedUserText(h)
      expect(rendered).toContain('<error>follow up on deployment checks</error>')
    }).pipe(Effect.provide(TestHarnessLive()))
  )

  it.live('observations during turn append a new inbox message (append-only)', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness

      yield* h.send({ type: 'turn_started', forkId: null, turnId: 't-obs-1', chainId: 'c-obs-1' })
      yield* h.send({
        type: 'observations_captured',
        forkId: null,
        turnId: 't-obs-1',
        parts: [
          { type: 'text', text: 'saw important output' },
          { type: 'image', base64: 'aGVsbG8=', mediaType: 'image/png', width: 10, height: 10 },
        ],
      })

      const memory = yield* getRootMemory(h)
      const allInbox = inboxMessages(memory)
      expect(allInbox.length).toBeGreaterThan(0)
      const last = allInbox[allInbox.length - 1]
      const prev = allInbox.length > 1 ? allInbox[allInbox.length - 2] : undefined

      expect(last?.type).toBe('inbox')
      if (last?.type === 'inbox') {
        expect(last.timeline.some(t => t.kind === 'observation')).toBe(true)
      }
      if (prev && prev.type === 'inbox') {
        expect(prev.timeline.some(t => t.kind === 'observation')).toBe(false)
      }

      const rendered = getView(memory.messages, 'UTC', 'agent')
        .filter(m => m.role === 'user')
        .flatMap(m => m.content)
      const hasObservationText = rendered.some(p => p.type === 'text' && p.text.includes('saw important output'))
      const hasImage = rendered.some(p => p.type === 'image')
      expect(hasObservationText).toBe(true)
      expect(hasImage).toBe(true)
    }).pipe(Effect.provide(TestHarnessLive()))
  )
})
